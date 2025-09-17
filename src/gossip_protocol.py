"""
Gossip Protocol implementation for efficient state propagation in the mesh.
Implements anti-entropy and rumor-mongering patterns for distributed state sync.
"""
import random
import time
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class GossipMessage:
    """A gossip message containing state updates and metadata."""

    sender_id: str
    timestamp: float
    message_type: str  # 'state_sync', 'rumor', 'anti_entropy'
    vector_clock: Dict[str, int]
    payload: Dict[str, Any]
    hop_count: int = 0
    ttl: int = 10  # Time to live for rumor spreading

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sender_id': self.sender_id,
            'timestamp': self.timestamp,
            'message_type': self.message_type,
            'vector_clock': self.vector_clock,
            'payload': self.payload,
            'hop_count': self.hop_count,
            'ttl': self.ttl
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GossipMessage':
        """Create from dictionary."""
        return cls(
            sender_id=data['sender_id'],
            timestamp=data['timestamp'],
            message_type=data['message_type'],
            vector_clock=data['vector_clock'],
            payload=data['payload'],
            hop_count=data.get('hop_count', 0),
            ttl=data.get('ttl', 10)
        )

    def increment_hop(self) -> 'GossipMessage':
        """Create a new message with incremented hop count and decremented TTL."""
        return GossipMessage(
            sender_id=self.sender_id,
            timestamp=self.timestamp,
            message_type=self.message_type,
            vector_clock=self.vector_clock,
            payload=self.payload,
            hop_count=self.hop_count + 1,
            ttl=self.ttl - 1
        )


@dataclass
class RumorState:
    """State tracking for rumor spreading."""

    message_id: str
    received_from: Set[str] = field(default_factory=set)
    sent_to: Set[str] = field(default_factory=set)
    first_seen: float = field(default_factory=time.time)
    last_propagated: float = field(default_factory=time.time)
    propagation_count: int = 0


class GossipProtocol:
    """
    Implements gossip-based dissemination for distributed state synchronization.

    Features:
    - Anti-entropy: Periodic full state synchronization
    - Rumor mongering: Efficient propagation of updates
    - Failure detection integration
    - Configurable gossip patterns
    """

    def __init__(self, node_id: str):
        self.node_id = node_id

        # Rumor tracking
        self.rumors: Dict[str, RumorState] = {}
        self.seen_messages: Set[str] = set()

        # Configuration
        self.gossip_fanout = 3  # Number of peers to gossip with per round
        self.rumor_timeout = 30.0  # Seconds to keep rumors alive
        self.anti_entropy_interval = 10.0  # Seconds between full sync
        self.max_message_age = 60.0  # Maximum age for messages

        # State tracking
        self.last_anti_entropy = 0.0
        self.gossip_rounds = 0

    def create_gossip_message(self, node) -> GossipMessage:
        """Create a gossip message containing current node state."""
        # Determine message type based on time since last anti-entropy
        current_time = time.time()

        if current_time - self.last_anti_entropy > self.anti_entropy_interval:
            message_type = 'anti_entropy'
            self.last_anti_entropy = current_time
            payload = self._create_full_state_payload(node)
        else:
            message_type = 'rumor'
            payload = self._create_rumor_payload(node)

        return GossipMessage(
            sender_id=self.node_id,
            timestamp=current_time,
            message_type=message_type,
            vector_clock=node.vector_clock.to_dict(),
            payload=payload
        )

    def _create_full_state_payload(self, node) -> Dict[str, Any]:
        """Create payload containing full node state for anti-entropy."""
        return {
            'crdt_store': node.crdt_store.to_dict(),
            'recent_events': [
                event.to_dict()
                for event in list(node.event_log.values())[-50:]  # Last 50 events
            ],
            'peer_info': {
                peer_id: {
                    'host': peer.host,
                    'port': peer.port,
                    'last_seen': peer.last_seen,
                    'is_alive': peer.is_alive
                }
                for peer_id, peer in node.peers.items()
            }
        }

    def _create_rumor_payload(self, node) -> Dict[str, Any]:
        """Create payload containing recent updates for rumor spreading."""
        current_time = time.time()
        recent_events = []

        # Include events from the last few gossip intervals
        cutoff_time = current_time - (self.anti_entropy_interval / 2)

        for event in node.event_log.values():
            # Extract timestamp from vector clock (approximate)
            if hasattr(event, 'timestamp') and event.node_id == self.node_id:
                recent_events.append(event.to_dict())

        # Limit to most recent events
        recent_events = recent_events[-10:]

        return {
            'recent_events': recent_events,
            'vector_clock_diff': node.vector_clock.to_dict()
        }

    async def handle_gossip(self, message: GossipMessage, node) -> None:
        """Handle incoming gossip message."""
        # Check if we've seen this message before
        message_id = f"{message.sender_id}:{message.timestamp}:{message.hop_count}"

        if message_id in self.seen_messages:
            return

        self.seen_messages.add(message_id)

        # Clean up old messages
        self._cleanup_old_messages()

        # Process based on message type
        if message.message_type == 'anti_entropy':
            await self._handle_anti_entropy(message, node)
        elif message.message_type == 'rumor':
            await self._handle_rumor(message, node)

        # Decide whether to propagate further
        if self._should_propagate(message):
            await self._propagate_message(message, node)

    async def _handle_anti_entropy(self, message: GossipMessage, node) -> None:
        """Handle anti-entropy message (full state sync)."""
        payload = message.payload

        # Merge CRDT store
        if 'crdt_store' in payload:
            try:
                other_store = node.crdt_store.__class__.from_dict(payload['crdt_store'])
                node.crdt_store.merge_store(other_store)
            except Exception as e:
                node.logger.warning(f"Failed to merge CRDT store: {e}")

        # Process recent events
        if 'recent_events' in payload:
            for event_data in payload['recent_events']:
                try:
                    from .causal_clock import CausalEvent
                    event = CausalEvent.from_dict(event_data)
                    if event.event_id not in node.event_log:
                        await node._process_causal_event(event)
                except Exception as e:
                    node.logger.warning(f"Failed to process event: {e}")

        # Update peer information
        if 'peer_info' in payload:
            for peer_id, peer_data in payload['peer_info'].items():
                if peer_id not in node.peers and peer_id != node.node_id:
                    # Add new peer
                    from .mesh_node import PeerInfo
                    peer_info = PeerInfo(
                        node_id=peer_id,
                        host=peer_data['host'],
                        port=peer_data['port'],
                        last_seen=peer_data['last_seen'],
                        is_alive=peer_data['is_alive']
                    )
                    node.peers[peer_id] = peer_info

    async def _handle_rumor(self, message: GossipMessage, node) -> None:
        """Handle rumor message (incremental updates)."""
        payload = message.payload

        # Merge vector clock
        from .causal_clock import VectorClock
        other_clock = VectorClock.from_dict(message.vector_clock)
        node.vector_clock = node.vector_clock.merge(other_clock)

        # Process recent events
        if 'recent_events' in payload:
            for event_data in payload['recent_events']:
                try:
                    from .causal_clock import CausalEvent
                    event = CausalEvent.from_dict(event_data)
                    if event.event_id not in node.event_log:
                        await node._process_causal_event(event)
                except Exception as e:
                    node.logger.warning(f"Failed to process rumor event: {e}")

    def _should_propagate(self, message: GossipMessage) -> bool:
        """Decide whether to propagate a message further."""
        # Don't propagate if TTL expired
        if message.ttl <= 0:
            return False

        # Don't propagate if too many hops
        if message.hop_count >= 5:
            return False

        # Probabilistic propagation for rumors
        if message.message_type == 'rumor':
            # Higher probability for newer messages
            age = time.time() - message.timestamp
            propagation_probability = max(0.1, 0.8 - (age / 30.0))
            return random.random() < propagation_probability

        # Always propagate anti-entropy messages (but with limited hops)
        return message.message_type == 'anti_entropy' and message.hop_count < 2

    async def _propagate_message(self, message: GossipMessage, node) -> None:
        """Propagate a message to selected peers."""
        # Select peers for propagation
        alive_peers = [p for p in node.peers.values() if p.is_alive]

        if not alive_peers:
            return

        # Choose random subset of peers
        num_peers = min(self.gossip_fanout, len(alive_peers))
        selected_peers = random.sample(alive_peers, num_peers)

        # Create propagated message
        propagated_msg = message.increment_hop()

        # Track rumor state
        message_id = f"{message.sender_id}:{message.timestamp}"
        if message_id not in self.rumors:
            self.rumors[message_id] = RumorState(message_id=message_id)

        rumor_state = self.rumors[message_id]

        # Send to selected peers
        for peer in selected_peers:
            if peer.node_id not in rumor_state.sent_to:
                await self._send_gossip_to_peer(peer, propagated_msg, node)
                rumor_state.sent_to.add(peer.node_id)

        rumor_state.last_propagated = time.time()
        rumor_state.propagation_count += 1

    async def _send_gossip_to_peer(self, peer, message: GossipMessage, node) -> None:
        """Send gossip message to a specific peer."""
        try:
            import asyncio
            import json

            reader, writer = await asyncio.open_connection(peer.host, peer.port)

            gossip_envelope = {
                'type': 'gossip',
                'from_node': self.node_id,
                'gossip_data': message.to_dict()
            }

            writer.write(json.dumps(gossip_envelope).encode() + b'\n')
            await writer.drain()

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            node.logger.warning(f"Failed to send gossip to {peer.node_id}: {e}")
            peer.is_alive = False

    def _cleanup_old_messages(self) -> None:
        """Clean up old messages and rumors."""
        current_time = time.time()

        # Clean up seen messages
        if len(self.seen_messages) > 1000:  # Keep last 1000 messages
            # This is a simple cleanup - in production you'd want timestamp-based cleanup
            old_messages = list(self.seen_messages)[:500]
            for msg_id in old_messages:
                self.seen_messages.discard(msg_id)

        # Clean up old rumors
        expired_rumors = []
        for message_id, rumor_state in self.rumors.items():
            if current_time - rumor_state.first_seen > self.rumor_timeout:
                expired_rumors.append(message_id)

        for message_id in expired_rumors:
            del self.rumors[message_id]

    def get_gossip_stats(self) -> Dict[str, Any]:
        """Get gossip protocol statistics."""
        current_time = time.time()

        active_rumors = sum(
            1 for rumor in self.rumors.values()
            if current_time - rumor.first_seen < self.rumor_timeout
        )

        return {
            'gossip_rounds': self.gossip_rounds,
            'active_rumors': active_rumors,
            'total_rumors': len(self.rumors),
            'seen_messages': len(self.seen_messages),
            'last_anti_entropy': self.last_anti_entropy,
            'fanout': self.gossip_fanout
        }