"""
Mesh Node implementation for the distributed causal consistency system.
Each node maintains connections to other nodes and propagates updates through gossip.
"""
import asyncio
import json
import logging
import random
import time
import uuid
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, field

from causal_clock import VectorClock, CausalEvent, CausalConsistencyChecker
from crdt_store import CRDTStore
from gossip_protocol import GossipProtocol, GossipMessage
from optimized_vector_clock import (
    CompressedVectorClock,
    TriangularMatrixClock,
    HybridOptimizedClock
)


@dataclass
class PeerInfo:
    """Information about a peer node in the mesh."""
    node_id: str
    host: str
    port: int
    last_seen: float = field(default_factory=time.time)
    is_alive: bool = True


class MeshNode:
    """
    A node in the causal mesh network.
    Handles causal consistency, CRDT operations, and peer-to-peer communication.
    """

    def __init__(self, node_id: str, host: str = "localhost", port: int = 8000,
                 use_optimized_clock: bool = None, optimization_threshold: int = 50):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.optimization_threshold = optimization_threshold

        # Smart optimization: auto-enable based on network size
        self.use_optimized_clock = use_optimized_clock
        if use_optimized_clock is None:
            self.use_optimized_clock = False  # Start with traditional, upgrade later

        # Causal consistency components - choose optimized or traditional clock
        if self.use_optimized_clock:
            try:
                self.vector_clock = HybridOptimizedClock()
                self.triangular_clock = TriangularMatrixClock()
            except ImportError:
                # Fallback if numpy/scipy not available
                self.vector_clock = VectorClock()
                self.triangular_clock = None
                self.use_optimized_clock = False
        else:
            self.vector_clock = VectorClock()
            self.triangular_clock = None

        self.event_log: Dict[str, CausalEvent] = {}
        self.applied_events: Set[str] = set()

        # Data storage
        self.crdt_store = CRDTStore(node_id)

        # Networking and peers
        self.peers: Dict[str, PeerInfo] = {}
        self.gossip_protocol = GossipProtocol(node_id)

        # Event handling
        self.event_handlers: Dict[str, Callable] = {}
        self.consistency_checker = CausalConsistencyChecker()

        # Async components
        self.server = None
        self.running = False

        # Configuration
        self.gossip_interval = 2.0  # seconds
        self.failure_detector_interval = 5.0  # seconds
        self.max_peers = 10

        # Logging
        self.logger = logging.getLogger(f"MeshNode-{node_id}")

    def _should_upgrade_to_optimized(self) -> bool:
        """Check if should upgrade to optimized version"""
        peer_count = len([p for p in self.peers.values() if p.is_alive])
        event_count = len(self.applied_events)

        # Upgrade conditions: node count > threshold OR event count > 10K
        return (peer_count >= self.optimization_threshold or
                event_count >= 10000) and not self.use_optimized_clock

    def _upgrade_to_optimized_clock(self):
        """Dynamically upgrade to optimized version"""
        try:
            self.logger.info("Upgrading to optimized vector clock due to scale")

            # Save current state
            old_clock_state = {}
            if hasattr(self.vector_clock, 'clocks'):
                old_clock_state = self.vector_clock.clocks.copy()

            # Create optimized version
            self.vector_clock = HybridOptimizedClock()
            self.triangular_clock = TriangularMatrixClock()

            # Migrate state
            for node_id, clock_val in old_clock_state.items():
                for _ in range(clock_val):
                    self.vector_clock = self.vector_clock.tick(node_id)
                    if self.triangular_clock:
                        self.triangular_clock = self.triangular_clock.tick(node_id)

            self.use_optimized_clock = True

        except ImportError as e:
            self.logger.warning(f"Cannot upgrade to optimized clock: {e}")

    def _check_optimization_upgrade(self):
        """Periodically check if optimization upgrade is needed"""
        if self._should_upgrade_to_optimized():
            self._upgrade_to_optimized_clock()

    async def start(self) -> None:
        """Start the mesh node server and background tasks."""
        self.running = True

        # Start the server
        self.server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port
        )

        self.logger.info(f"Node {self.node_id} started on {self.host}:{self.port}")

        # Start background tasks
        asyncio.create_task(self._gossip_loop())
        asyncio.create_task(self._failure_detector_loop())

        await self.server.start_serving()

    async def stop(self) -> None:
        """Stop the mesh node and cleanup resources."""
        self.running = False

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        self.logger.info(f"Node {self.node_id} stopped")

    async def join_mesh(self, seed_host: str, seed_port: int) -> bool:
        """Join an existing mesh by connecting to a seed node."""
        try:
            reader, writer = await asyncio.open_connection(seed_host, seed_port)

            # Send join request
            join_message = {
                'type': 'join_request',
                'node_id': self.node_id,
                'host': self.host,
                'port': self.port
            }

            writer.write(json.dumps(join_message).encode() + b'\n')
            await writer.drain()

            # Wait for response
            response_data = await reader.readline()
            response = json.loads(response_data.decode().strip())

            if response['type'] == 'join_response' and response['success']:
                # Add peers from response
                for peer_data in response['peers']:
                    peer_info = PeerInfo(
                        node_id=peer_data['node_id'],
                        host=peer_data['host'],
                        port=peer_data['port']
                    )
                    self.peers[peer_info.node_id] = peer_info

                self.logger.info(f"Successfully joined mesh with {len(self.peers)} peers")
                return True

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            self.logger.error(f"Failed to join mesh: {e}")

        return False

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming connections from other nodes."""
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break

                message = json.loads(data.decode().strip())
                await self._process_message(message, writer)

        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_message(self, message: Dict[str, Any], writer: asyncio.StreamWriter) -> None:
        """Process incoming messages from other nodes."""
        msg_type = message.get('type')

        if msg_type == 'join_request':
            await self._handle_join_request(message, writer)
        elif msg_type == 'gossip':
            await self._handle_gossip_message(message)
        elif msg_type == 'causal_event':
            await self._handle_causal_event(message)
        elif msg_type == 'ping':
            await self._handle_ping(message, writer)
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_join_request(self, message: Dict[str, Any], writer: asyncio.StreamWriter) -> None:
        """Handle a node requesting to join the mesh."""
        requester_id = message['node_id']
        requester_host = message['host']
        requester_port = message['port']

        # Add the requester as a peer
        peer_info = PeerInfo(
            node_id=requester_id,
            host=requester_host,
            port=requester_port
        )
        self.peers[requester_id] = peer_info

        # Send response with current peer list
        response = {
            'type': 'join_response',
            'success': True,
            'peers': [
                {
                    'node_id': peer.node_id,
                    'host': peer.host,
                    'port': peer.port
                }
                for peer in self.peers.values()
                if peer.node_id != requester_id
            ]
        }

        writer.write(json.dumps(response).encode() + b'\n')
        await writer.drain()

        self.logger.info(f"Node {requester_id} joined the mesh")

    async def _handle_gossip_message(self, message: Dict[str, Any]) -> None:
        """Handle gossip messages for state synchronization."""
        gossip_msg = GossipMessage.from_dict(message['gossip_data'])
        await self.gossip_protocol.handle_gossip(gossip_msg, self)

    async def _handle_causal_event(self, message: Dict[str, Any]) -> None:
        """Handle causal events from other nodes."""
        event = CausalEvent.from_dict(message['event'])
        await self._process_causal_event(event)

    async def _handle_ping(self, message: Dict[str, Any], writer: asyncio.StreamWriter) -> None:
        """Handle ping messages for failure detection."""
        response = {
            'type': 'pong',
            'node_id': self.node_id,
            'timestamp': time.time()
        }

        writer.write(json.dumps(response).encode() + b'\n')
        await writer.drain()

    async def _gossip_loop(self) -> None:
        """Background task for periodic gossip."""
        while self.running:
            try:
                await self._do_gossip_round()
                await asyncio.sleep(self.gossip_interval)
            except Exception as e:
                self.logger.error(f"Error in gossip loop: {e}")

    async def _do_gossip_round(self) -> None:
        """Perform one round of gossip with random peers."""
        if not self.peers:
            return

        # Select random peers for gossip
        num_peers = min(3, len(self.peers))  # Gossip with up to 3 peers
        selected_peers = random.sample(list(self.peers.values()), num_peers)

        for peer in selected_peers:
            if peer.is_alive:
                await self._gossip_with_peer(peer)

    async def _gossip_with_peer(self, peer: PeerInfo) -> None:
        """Gossip with a specific peer."""
        try:
            gossip_msg = self.gossip_protocol.create_gossip_message(self)

            message = {
                'type': 'gossip',
                'from_node': self.node_id,
                'gossip_data': gossip_msg.to_dict()
            }

            reader, writer = await asyncio.open_connection(peer.host, peer.port)
            writer.write(json.dumps(message).encode() + b'\n')
            await writer.drain()

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            self.logger.warning(f"Failed to gossip with {peer.node_id}: {e}")
            peer.is_alive = False

    async def _failure_detector_loop(self) -> None:
        """Background task for failure detection."""
        while self.running:
            try:
                await self._check_peer_health()
                await asyncio.sleep(self.failure_detector_interval)
            except Exception as e:
                self.logger.error(f"Error in failure detector: {e}")

    async def _check_peer_health(self) -> None:
        """Check health of all peers."""
        for peer in list(self.peers.values()):
            if time.time() - peer.last_seen > self.failure_detector_interval * 2:
                await self._ping_peer(peer)

    async def _ping_peer(self, peer: PeerInfo) -> None:
        """Ping a specific peer to check if it's alive."""
        try:
            reader, writer = await asyncio.open_connection(peer.host, peer.port)

            ping_message = {
                'type': 'ping',
                'from_node': self.node_id,
                'timestamp': time.time()
            }

            writer.write(json.dumps(ping_message).encode() + b'\n')
            await writer.drain()

            # Wait for pong with timeout
            response_data = await asyncio.wait_for(reader.readline(), timeout=2.0)
            response = json.loads(response_data.decode().strip())

            if response['type'] == 'pong':
                peer.last_seen = time.time()
                peer.is_alive = True

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            self.logger.warning(f"Peer {peer.node_id} appears to be down: {e}")
            peer.is_alive = False

    # Public API for causal operations

    async def causal_put(self, key: str, value: Any) -> str:
        """Put a key-value pair with causal consistency."""
        # Check if we should upgrade to optimized version
        self._check_optimization_upgrade()

        # Create causal event with optimized or traditional clock
        if self.use_optimized_clock:
            self.vector_clock = self.vector_clock.tick(self.node_id)
            # Also update triangular matrix for enhanced causality tracking
            if self.triangular_clock:
                self.triangular_clock = self.triangular_clock.tick(self.node_id)

            # For now, create a traditional VectorClock for event compatibility
            traditional_clock = VectorClock({self.node_id: self.triangular_clock.get_clock(self.node_id)})
        else:
            self.vector_clock = self.vector_clock.tick(self.node_id)
            traditional_clock = self.vector_clock

        event_id = f"{self.node_id}:{uuid.uuid4().hex[:8]}"

        event = CausalEvent(
            event_id=event_id,
            node_id=self.node_id,
            timestamp=traditional_clock,
            operation='put',
            key=key,
            value=value
        )

        # Apply locally
        await self._process_causal_event(event)

        # Propagate to peers
        await self._broadcast_event(event)

        return event_id

    async def causal_get(self, key: str) -> Any:
        """Get a value with causal consistency guarantees."""
        return self.crdt_store.get(key)

    async def _process_causal_event(self, event: CausalEvent) -> None:
        """Process a causal event, applying it if dependencies are satisfied."""
        self.event_log[event.event_id] = event

        # Check if we can apply this event
        if self.consistency_checker.can_apply_event(
            event, self.vector_clock,
            {eid: e for eid, e in self.event_log.items() if eid in self.applied_events}
        ):
            await self._apply_event(event)

    async def _apply_event(self, event: CausalEvent) -> None:
        """Apply a causal event to the local state."""
        if event.event_id in self.applied_events:
            return

        # Update vector clock - handle both optimized and traditional versions
        if self.use_optimized_clock:
            # For optimized clock, we need to extract values and update
            if isinstance(event.timestamp, VectorClock):
                # Create a temporary optimized clock from the event timestamp
                temp_optimized = HybridOptimizedClock()
                for node_id, clock_val in event.timestamp.clocks.items():
                    for _ in range(clock_val):
                        temp_optimized = temp_optimized.tick(node_id)

                # Merge with current optimized clock
                self.vector_clock = self.vector_clock.merge(temp_optimized)

            # Also update triangular matrix if available
            if self.triangular_clock and event.node_id != self.node_id:
                self.triangular_clock = self.triangular_clock.tick(event.node_id)
        else:
            self.vector_clock = self.vector_clock.merge(event.timestamp)

        # Apply the operation
        if event.operation == 'put':
            self.crdt_store.set_lww(event.key, event.value)
        elif event.operation == 'add_to_set':
            self.crdt_store.add_to_set(event.key, event.value)
        elif event.operation == 'remove_from_set':
            self.crdt_store.remove_from_set(event.key, event.value)

        self.applied_events.add(event.event_id)
        self.logger.debug(f"Applied event {event.event_id}: {event.operation} {event.key}")

        # Try to apply any pending events that might now be ready
        await self._try_apply_pending_events()

    async def _try_apply_pending_events(self) -> None:
        """Try to apply events that were waiting for dependencies."""
        applied_any = True
        while applied_any:
            applied_any = False
            for event_id, event in self.event_log.items():
                if event_id not in self.applied_events:
                    if self.consistency_checker.can_apply_event(
                        event, self.vector_clock,
                        {eid: e for eid, e in self.event_log.items() if eid in self.applied_events}
                    ):
                        await self._apply_event(event)
                        applied_any = True
                        break

    async def _broadcast_event(self, event: CausalEvent) -> None:
        """Broadcast a causal event to all peers."""
        message = {
            'type': 'causal_event',
            'from_node': self.node_id,
            'event': event.to_dict()
        }

        # Send to all alive peers
        for peer in self.peers.values():
            if peer.is_alive:
                asyncio.create_task(self._send_event_to_peer(peer, message))

    async def _send_event_to_peer(self, peer: PeerInfo, message: Dict[str, Any]) -> None:
        """Send an event to a specific peer."""
        try:
            reader, writer = await asyncio.open_connection(peer.host, peer.port)
            writer.write(json.dumps(message).encode() + b'\n')
            await writer.drain()

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            self.logger.warning(f"Failed to send event to {peer.node_id}: {e}")
            peer.is_alive = False

    def get_status(self) -> Dict[str, Any]:
        """Get current node status for monitoring."""
        status = {
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'peers': len([p for p in self.peers.values() if p.is_alive]),
            'total_events': len(self.event_log),
            'applied_events': len(self.applied_events),
            'keys': self.crdt_store.keys(),
            'optimized_clock_enabled': self.use_optimized_clock
        }

        # Add clock information based on type
        if self.use_optimized_clock:
            if hasattr(self.vector_clock, 'get_performance_stats'):
                status['hybrid_clock_stats'] = self.vector_clock.get_performance_stats()

            if self.triangular_clock:
                status['triangular_matrix_stats'] = self.triangular_clock.memory_usage()
        else:
            status['vector_clock'] = self.vector_clock.to_dict()

        return status

    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get detailed optimization metrics for performance analysis."""
        if not self.use_optimized_clock:
            return {'optimization_enabled': False}

        metrics = {
            'optimization_enabled': True,
            'hybrid_clock_performance': self.vector_clock.get_performance_stats() if hasattr(self.vector_clock, 'get_performance_stats') else {},
            'triangular_matrix_efficiency': self.triangular_clock.memory_usage() if self.triangular_clock else {},
        }

        # Calculate estimated memory savings
        traditional_size = len(self.applied_events) * 50  # 估算每個事件50字節
        if self.triangular_clock:
            optimized_size = self.triangular_clock.memory_usage().get('triangular_elements', 0) * 4  # 4字節每元素
            if traditional_size > 0:
                metrics['memory_savings'] = {
                    'traditional_estimated_bytes': traditional_size,
                    'optimized_bytes': optimized_size,
                    'savings_percent': max(0, (traditional_size - optimized_size) / traditional_size * 100)
                }

        return metrics