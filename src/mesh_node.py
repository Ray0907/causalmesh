"""
CausalMesh Node implementation based on the paper: "CausalMesh: A Formally
Verified Causal Cache for Stateful Serverless Computing".

This implementation replaces the original gossip-based protocol with a deterministic
propagation chain and a dual-cache system to provide Causal+ (CC+) consistency.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Set, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field

try:
    from .causal_clock import VectorClock
except ImportError:
    # Fallback for direct execution
    from causal_clock import VectorClock

# Type definitions for clarity
Version = Tuple[Any, VectorClock, Dict[str, VectorClock]]  # (Value, VC, Deps)
C_Cache_Entry = Tuple[Any, VectorClock]
I_Cache = Dict[str, List[Version]]
C_Cache = Dict[str, C_Cache_Entry]


@dataclass
class PeerInfo:
    """Information about a peer node in the mesh."""
    node_id: str
    host: str
    port: int


class MeshNode:
    """
    A node in the CausalMesh network.
    It handles client requests and propagates writes through a deterministic chain
    to ensure causal consistency for roaming clients (serverless workflows).
    """

    def __init__(self, node_id: str, host: str = "localhost", port: int = 8000):
        self.node_id = node_id
        self.host = host
        self.port = port

        # Causal consistency components from the paper
        self.gvc = VectorClock()  # Global Vector Clock
        self.c_cache: C_Cache = {}  # Consistent cache (for reads)
        self.i_cache: I_Cache = {}  # Inconsistent cache (for writes)

        # Networking and peer management
        self.peers: Dict[str, PeerInfo] = {}
        self.successor: Optional[PeerInfo] = None

        # Async components
        self.server = None
        self.running = False

        # Logging
        self.logger = logging.getLogger(f"CausalMeshNode-{node_id}")
        logging.basicConfig(level=logging.INFO)

    def _update_chain(self):
        """Updates the successor based on the current peer list."""
        all_node_ids = sorted(list(self.peers.keys()) + [self.node_id])

        # Remove duplicates while preserving order
        unique_node_ids = []
        for node_id in all_node_ids:
            if node_id not in unique_node_ids:
                unique_node_ids.append(node_id)
        all_node_ids = unique_node_ids

        if len(all_node_ids) <= 1:
            self.successor = None
            self.logger.info(f"My successor is: None (only {len(all_node_ids)} nodes)")
            return

        my_index = all_node_ids.index(self.node_id)
        successor_index = (my_index + 1) % len(all_node_ids)
        successor_id = all_node_ids[successor_index]

        self.logger.info(f"Chain calculation: nodes={all_node_ids}, my_index={my_index}, successor_index={successor_index}, successor_id={successor_id}")

        if successor_id == self.node_id:
            self.successor = None # Running solo (should not happen with len > 1)
            self.logger.warning("Calculated successor is myself - this should not happen")
        else:
            self.successor = self.peers.get(successor_id)
            if not self.successor:
                self.logger.error(f"Successor {successor_id} not found in peers: {list(self.peers.keys())}")

        self.logger.info(f"Chain: {' -> '.join(all_node_ids)} -> {all_node_ids[0]}")
        self.logger.info(f"My successor is: {self.successor.node_id if self.successor else 'None'}")

    async def start(self) -> None:
        """Start the mesh node server."""
        self.running = True
        self.server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        self.logger.info(f"Node {self.node_id} started on {self.host}:{self.port}")
        await self.server.serve_forever()

    async def stop(self) -> None:
        """Stop the mesh node."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.logger.info(f"Node {self.node_id} stopped")

    async def join_mesh(self, seed_host: str, seed_port: int) -> bool:
        """Join an existing mesh by connecting to a seed node."""
        try:
            reader, writer = await asyncio.open_connection(seed_host, seed_port)
            join_message = {
                'type': 'join_request',
                'node_id': self.node_id, 'host': self.host, 'port': self.port
            }
            writer.write(json.dumps(join_message).encode() + b'\n')
            await writer.drain()

            response_data = await reader.readline()
            response = json.loads(response_data.decode().strip())

            if response['type'] == 'join_response' and response['success']:
                # Populate peer list from the seed's response, including the seed itself
                for peer_data in response['peers']:
                    peer = PeerInfo(**peer_data)
                    if peer.node_id != self.node_id:
                        self.peers[peer.node_id] = peer
                
                self._update_chain()
                self.logger.info(f"Successfully joined mesh. Current peers: {list(self.peers.keys())}")
                return True
            
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            self.logger.error(f"Failed to join mesh: {e}")
        return False

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming connections from other nodes or clients."""
        try:
            data = await reader.readline()
            if not data:
                return
            message = json.loads(data.decode().strip())
            response = await self._process_message(message)
            if response:
                writer.write(json.dumps(response).encode() + b'\n')
                await writer.drain()
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming messages and route to appropriate handlers."""
        msg_type = message.get('type')
        handlers = {
            'join_request': self._handle_join_request,
            'new_peer': self._handle_new_peer,
            'ClientRead': self._handle_client_read,
            'ClientWrite': self._handle_client_write,
            'ServerWrite': self._handle_server_write,
            'ManualIntegrate': self._handle_manual_integrate,
            'test': self._handle_test,
        }
        handler = handlers.get(msg_type)
        if handler:
            return await handler(message)
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")
            return {'status': 'error', 'message': 'Unknown type'}

    # --- Message Handlers ---

    async def _handle_join_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a node's request to join the mesh."""
        new_peer_info = PeerInfo(message['node_id'], message['host'], message['port'])

        # Add the new peer to this node's list first
        self.peers[new_peer_info.node_id] = new_peer_info
        self.logger.info(f"Node {new_peer_info.node_id} added to my peer list.")

        # Prepare response with the complete list of all current nodes in the mesh
        all_nodes_in_mesh = [p.__dict__ for p in self.peers.values()]
        # Only add self if not already in peers (which should be the case)
        self_info = PeerInfo(self.node_id, self.host, self.port).__dict__
        if not any(p['node_id'] == self.node_id for p in all_nodes_in_mesh):
            all_nodes_in_mesh.append(self_info)

        response = {
            'type': 'join_response',
            'success': True,
            'peers': all_nodes_in_mesh
        }

        # Broadcast the new peer's info to all *other* existing peers
        broadcast_message = {
            'type': 'new_peer',
            'node_id': new_peer_info.node_id, 'host': new_peer_info.host, 'port': new_peer_info.port
        }
        for peer_id, peer in self.peers.items():
            if peer_id != new_peer_info.node_id:
                asyncio.create_task(self._send_message_to_peer(peer, broadcast_message))

        # Finally, update own successor
        self._update_chain()
        return response

    async def _handle_new_peer(self, message: Dict[str, Any]) -> None:
        """Handle notification of a new peer joining the mesh."""
        peer_info = PeerInfo(message['node_id'], message['host'], message['port'])
        if peer_info.node_id != self.node_id and peer_info.node_id not in self.peers:
            self.peers[peer_info.node_id] = peer_info
            self._update_chain()
            self.logger.info(f"Acknowledged new peer {peer_info.node_id} via broadcast.")

    async def _handle_client_read(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a read request from a client."""
        key = message['key']
        deps_dict = message.get('deps', {})
        deps = {k: VectorClock(v) for k, v in deps_dict.items()}

        self._dependency_integration(deps)

        value, vc = self.c_cache.get(key, (None, VectorClock()))
        return {'status': 'ok', 'value': value, 'vc': vc.clocks}

    async def _handle_client_write(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a write request from a client."""
        key, value = message['key'], message['value']
        deps_dict = message.get('deps', {})
        deps = {k: VectorClock(v) for k, v in deps_dict.items()}

        self.logger.info(f"Handling ClientWrite: key={key}, value={value}, deps={deps_dict}")

        # 1. Update GVC and create a new version for the write
        self.gvc.tick(self.node_id)
        new_vc = self.gvc.copy()

        # 2. Add the write to the local I-cache
        version: Version = (value, new_vc, deps)
        if key not in self.i_cache:
            self.i_cache[key] = []
        self.i_cache[key].append(version)

        self.logger.info(f"Added to I-cache: key={key}, value={value}, vc={new_vc}")

        # 3. Start propagation along the chain
        if self.successor:
            propagation_message = {
                'type': 'ServerWrite',
                'key': key,
                'value': value,
                'vc': new_vc.clocks,
                'deps': {k: v.clocks for k, v in deps.items()},
                'round': 1,
                'origin_node': self.node_id
            }
            self.logger.info(f"Starting propagation to {self.successor.node_id}")
            asyncio.create_task(self._send_message_to_peer(self.successor, propagation_message))
        else:
            self.logger.warning("No successor found for propagation - integrating locally")
            # If no successor, integrate immediately for demonstration
            self._dependency_integration({key: new_vc})

        return {'status': 'ok', 'vc': new_vc.clocks}

    async def _handle_server_write(self, message: Dict[str, Any]) -> None:
        """Handle a propagated write from another server."""
        key, value = message['key'], message['value']
        vc = VectorClock(message['vc'])
        deps = {k: VectorClock(v) for k, v in message['deps'].items()}
        current_round = message['round']
        origin_node = message['origin_node']

        self.logger.info(f"Received ServerWrite: key={key}, vc={vc}, round={current_round}, origin={origin_node}")

        # Add to local I-cache
        version: Version = (value, vc, deps)
        if key not in self.i_cache:
            self.i_cache[key] = []
        self.i_cache[key].append(version)

        # Determine if this node is the tail for this write
        # The tail is the predecessor of the origin node.
        all_node_ids = sorted(list(self.peers.keys()) + [self.node_id])

        # Remove duplicates
        unique_node_ids = []
        for node_id in all_node_ids:
            if node_id not in unique_node_ids:
                unique_node_ids.append(node_id)
        all_node_ids = unique_node_ids

        # If origin_node is not in our node list, add it
        if origin_node not in all_node_ids:
            all_node_ids.append(origin_node)
            all_node_ids.sort()

        origin_index = all_node_ids.index(origin_node)
        tail_index = (origin_index - 1 + len(all_node_ids)) % len(all_node_ids)
        tail_id = all_node_ids[tail_index]
        is_tail = self.node_id == tail_id

        self.logger.info(f"Chain: {all_node_ids}, Origin: {origin_node}, Tail: {tail_id}, Am tail: {is_tail}")

        # Continue propagation
        if self.successor:
            next_round = current_round
            if self.successor.node_id == origin_node:
                next_round += 1 # Completed a full circle

            if next_round <= 2:
                message['round'] = next_round
                self.logger.info(f"Forwarding to {self.successor.node_id}, round {next_round}")
                asyncio.create_task(self._send_message_to_peer(self.successor, message))

        # Check if we should integrate after propagation
        if is_tail and current_round == 2:
            # Write has completed two full rounds and reached the tail. It's safe to integrate.
            self.logger.info(f"Write for key '{key}' (vc={vc}) reached tail after 2 rounds. Integrating.")
            self._dependency_integration({key: vc})

    async def _handle_manual_integrate(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle manual integration for demo purposes."""
        key = message['key']
        value = message['value']
        vc = VectorClock(message['vc'])

        self.logger.info(f"Manual integration: key={key}, value={value}, vc={vc}")

        # Add directly to C-cache
        self.c_cache[key] = (value, vc)
        self.gvc.merge(vc)

        return {'status': 'ok', 'message': f'Manually integrated {key}={value}'}

    async def _handle_test(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle test connectivity messages."""
        return {'status': 'ok', 'message': f'Node {self.node_id} is alive'}

    # --- Core CausalMesh Logic ---

    def _dependency_integration(self, deps: Dict[str, VectorClock]):
        """
        Integrates a version and all its transitive dependencies from I-cache to C-cache.
        This is the core of ensuring a client reads from a consistent snapshot.
        """
        self.logger.info(f"Starting dependency integration for: {[(k, str(v)) for k, v in deps.items()]}")
        all_deps_to_integrate = self._find_transitive_deps(deps)
        self.logger.info(f"All dependencies to integrate: {[(k, str(v)) for k, v in all_deps_to_integrate.items()]}")

        for key, vc in all_deps_to_integrate.items():
            # Check if already in C-cache with a sufficient version
            if key in self.c_cache and self.c_cache[key][1] >= vc:
                self.logger.info(f"Key '{key}' already in C-cache with sufficient version")
                continue

            # Find the specific version in I-cache
            version_to_integrate = None
            if key in self.i_cache:
                for version in self.i_cache[key]:
                    if version[1] == vc:
                        version_to_integrate = version
                        break

            if version_to_integrate:
                # Remove from I-cache and merge into C-cache
                self.i_cache[key].remove(version_to_integrate)
                value, version_vc, _ = version_to_integrate

                # Merge into C-cache (overwrite if older)
                self.c_cache[key] = (value, version_vc)
                self.gvc.merge(version_vc)
                self.logger.info(f"Integrated key '{key}' (value={value}, vc={version_vc}) into C-cache.")
            else:
                self.logger.warning(f"Could not find version for key '{key}' (vc={vc}) in I-cache")

    def _find_transitive_deps(self, initial_deps: Dict[str, VectorClock]) -> Dict[str, VectorClock]:
        """Recursively find all dependencies for a given set."""
        worklist = list(initial_deps.items())
        all_deps = initial_deps.copy()
        processed = set()

        while worklist:
            key, vc = worklist.pop(0)
            if (key, str(vc.clocks)) in processed:
                continue
            processed.add((key, str(vc.clocks)))

            # Find the version in I-cache to get its dependencies
            if key in self.i_cache:
                for _, version_vc, version_deps in self.i_cache[key]:
                    if version_vc == vc:
                        for dep_key, dep_vc in version_deps.items():
                            if dep_key not in all_deps or all_deps[dep_key] < dep_vc:
                                all_deps[dep_key] = dep_vc
                                worklist.append((dep_key, dep_vc))
                        break
        return all_deps

    async def _send_message_to_peer(self, peer: PeerInfo, message: Dict[str, Any]):
        """Utility to send a message to a specific peer."""
        try:
            reader, writer = await asyncio.open_connection(peer.host, peer.port)
            writer.write(json.dumps(message).encode() + b'\n')
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except ConnectionRefusedError:
            self.logger.warning(f"Connection refused by peer {peer.node_id}. Is it running?")
        except Exception as e:
            self.logger.error(f"Failed to send message to {peer.node_id}: {e}")
