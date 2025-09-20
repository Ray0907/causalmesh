# CausalMesh

A distributed system that provides **causal consistency** through deterministic propagation, dual caching, and dependency integration.

## Architecture

CausalMesh ensures causal consistency by combining:

- **Propagation Chain**: Deterministic, two-round write propagation along a ring topology.
- **Dual Cache**:
  - `C-cache` (Consistent Cache) – serves read requests.
  - `I-cache` (Inconsistent Cache) – buffers local and in-flight writes.
- **Dependency Integration**: Ensures all causal dependencies of a read are resolved by recursively moving required entries from `I-cache` to `C-cache`.
- **Vector Clocks**: Lightweight mechanism for tracking causal relationships.

## Key Features

- **Causal Consistency**: Guarantees that reads respect causality.
- **Deterministic Write Propagation**: Eliminates uncertainty by replacing gossip with ordered two-phase forwarding.
- **Cache Separation**: Optimized read/write handling with dual caches.
- **Dependency Resolution**: Reads automatically integrate dependencies to return consistent results.
- **Simplicity**: Focused architecture aligned with the CausalMesh paper.

## References

This implementation follows the architecture described in:

1. **CausalMesh: A Formally Verified Causal Cache for Stateful Serverless Computing** ([Paper](https://arxiv.org/abs/2508.15647))

## Core Components

### Mesh Node (`src/mesh_node.py`)

- Manages node participation in the propagation chain.
- Forwards writes deterministically to successors.
- Applies dependency integration before serving reads.

### Vector Clocks (`src/causal_clock.py`)

- Provides causal ordering of events.
- Simplified to core vector clock logic for integration with caches.

### Dual Cache

- **C-cache**: Stable, causally consistent state for serving reads.
- **I-cache**: Holds pending or in-flight writes until dependencies are integrated.

## Use Cases

CausalMesh is suitable for:

- **Collaborative Applications**: Real-time document editing, shared state.
- **Distributed Databases**: Replication with causal guarantees.
- **Edge & Serverless Computing**: Systems requiring partition tolerance and causality.
- **Research**: Exploring causal consistency and cache-based designs.
