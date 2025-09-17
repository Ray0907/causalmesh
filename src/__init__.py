"""
CausalMesh - A distributed causal consistency system POC.

This package implements key concepts from distributed systems research:
- Causal consistency with vector clocks
- Conflict-free replicated data types (CRDTs)
- Mesh-based peer-to-peer networking
- Gossip protocols for state propagation
"""

from causal_clock import VectorClock, CausalEvent, CausalConsistencyChecker
from crdt_store import CRDTStore, LWWRegister, GSet, ORSet
from mesh_node import MeshNode, PeerInfo
from gossip_protocol import GossipProtocol, GossipMessage

__version__ = "0.1.0"
__all__ = [
    "VectorClock",
    "CausalEvent",
    "CausalConsistencyChecker",
    "CRDTStore",
    "LWWRegister",
    "GSet",
    "ORSet",
    "MeshNode",
    "PeerInfo",
    "GossipProtocol",
    "GossipMessage"
]