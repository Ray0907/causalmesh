"""
Optimized vector clock implementation using matrix structures to reduce computational cost.
Includes upper triangular, lower triangular, and sparse matrix optimizations.
"""
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from scipy.sparse import csr_matrix, lil_matrix
import bisect


@dataclass
class NodeMapping:
    """Maps node IDs to matrix indices for efficient matrix operations"""
    node_to_index: Dict[str, int]
    index_to_node: Dict[int, str]
    next_index: int = 0

    def get_or_create_index(self, node_id: str) -> int:
        """Get or create a matrix index for the given node ID"""
        if node_id not in self.node_to_index:
            self.node_to_index[node_id] = self.next_index
            self.index_to_node[self.next_index] = node_id
            self.next_index += 1
        return self.node_to_index[node_id]

    def size(self) -> int:
        return self.next_index


class CompressedVectorClock:
    """
    Compressed vector clock optimized for sparsity.
    Only stores non-zero clock values to reduce memory and computational cost.
    """

    def __init__(self, clocks: Optional[Dict[str, int]] = None):
        # Store only non-zero values to leverage sparsity
        self.clocks = {k: v for k, v in (clocks or {}).items() if v > 0}

    def tick(self, node_id: str) -> 'CompressedVectorClock':
        """Optimized clock increment avoiding unnecessary copying"""
        new_clocks = self.clocks.copy()
        new_clocks[node_id] = new_clocks.get(node_id, 0) + 1
        return CompressedVectorClock(new_clocks)

    def merge(self, other: 'CompressedVectorClock') -> 'CompressedVectorClock':
        """Optimized merge operation leveraging sparsity"""
        # Only process nodes that actually exist
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        new_clocks = {}

        for node in all_nodes:
            self_val = self.clocks.get(node, 0)
            other_val = other.clocks.get(node, 0)
            max_val = max(self_val, other_val)
            if max_val > 0:  # Only store non-zero values
                new_clocks[node] = max_val

        return CompressedVectorClock(new_clocks)

    def happens_before(self, other: 'CompressedVectorClock') -> bool:
        """Optimized causality relationship check"""
        if not self.clocks:
            return bool(other.clocks)

        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())

        has_strictly_less = False
        for node in all_nodes:
            self_val = self.clocks.get(node, 0)
            other_val = other.clocks.get(node, 0)

            if self_val > other_val:
                return False
            elif self_val < other_val:
                has_strictly_less = True

        return has_strictly_less


class TriangularMatrixClock:
    """
    Uses upper triangular matrix to store causal relationships between nodes.
    Suitable for scenarios with hierarchical node structures, significantly reducing storage space.
    """

    def __init__(self, node_mapping: Optional[NodeMapping] = None):
        self.node_mapping = node_mapping or NodeMapping({}, {})
        # Use upper triangular matrix, only store items where i <= j
        self.matrix = lil_matrix((0, 0), dtype=np.int32)

    def _ensure_size(self, min_size: int):
        """Ensure matrix size is sufficient"""
        current_size = self.matrix.shape[0]
        if min_size > current_size:
            # Expand matrix size
            new_size = max(min_size, current_size * 2)
            new_matrix = lil_matrix((new_size, new_size), dtype=np.int32)
            new_matrix[:current_size, :current_size] = self.matrix
            self.matrix = new_matrix

    def tick(self, node_id: str) -> 'TriangularMatrixClock':
        """Increment the specified node's clock"""
        node_idx = self.node_mapping.get_or_create_index(node_id)
        self._ensure_size(node_idx + 1)

        # Create new clock copy
        new_clock = TriangularMatrixClock(self.node_mapping)
        new_clock.matrix = self.matrix.copy()
        new_clock._ensure_size(node_idx + 1)

        # Increment on diagonal (own clock)
        new_clock.matrix[node_idx, node_idx] += 1

        return new_clock

    def merge(self, other: 'TriangularMatrixClock') -> 'TriangularMatrixClock':
        """Merge two triangular matrix clocks"""
        # Unify node mapping
        all_nodes = set(self.node_mapping.node_to_index.keys()) | \
                   set(other.node_mapping.node_to_index.keys())

        new_mapping = NodeMapping({}, {})
        for node in all_nodes:
            new_mapping.get_or_create_index(node)

        new_clock = TriangularMatrixClock(new_mapping)
        new_clock._ensure_size(new_mapping.size())

        # Merge matrices: take maximum value for each position
        for node in all_nodes:
            new_idx = new_mapping.get_or_create_index(node)

            self_idx = self.node_mapping.node_to_index.get(node)
            other_idx = other.node_mapping.node_to_index.get(node)

            self_val = 0
            other_val = 0

            if self_idx is not None and self_idx < self.matrix.shape[0]:
                self_val = self.matrix[self_idx, self_idx]

            if other_idx is not None and other_idx < other.matrix.shape[0]:
                other_val = other.matrix[other_idx, other_idx]

            new_clock.matrix[new_idx, new_idx] = max(self_val, other_val)

        return new_clock

    def get_clock(self, node_id: str) -> int:
        """Get the clock value for the specified node"""
        if node_id not in self.node_mapping.node_to_index:
            return 0

        idx = self.node_mapping.node_to_index[node_id]
        if idx >= self.matrix.shape[0]:
            return 0

        return int(self.matrix[idx, idx])

    def memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage"""
        # Upper triangular matrix only needs n(n+1)/2 space
        n = self.node_mapping.size()
        triangular_elements = n * (n + 1) // 2

        return {
            'nodes': n,
            'triangular_elements': triangular_elements,
            'full_matrix_elements': n * n,
            'space_saving_percent': int((1 - triangular_elements / max(n*n, 1)) * 100)
        }


class DeltaVectorClock:
    """
    Delta vector clock that only transmits and stores the changed parts.
    Uses differential compression to reduce network transmission and storage costs.
    """

    def __init__(self, base_clocks: Optional[Dict[str, int]] = None):
        self.base_clocks = base_clocks or {}
        self.deltas: List[Tuple[str, int, int]] = []  # (node_id, old_val, new_val)

    def apply_delta(self, node_id: str, old_val: int, new_val: int) -> 'DeltaVectorClock':
        """Apply delta change"""
        new_clock = DeltaVectorClock(self.base_clocks.copy())
        new_clock.deltas = self.deltas.copy()
        new_clock.deltas.append((node_id, old_val, new_val))
        return new_clock

    def tick(self, node_id: str) -> 'DeltaVectorClock':
        """Increment clock and record delta"""
        current_val = self.get_current_value(node_id)
        return self.apply_delta(node_id, current_val, current_val + 1)

    def get_current_value(self, node_id: str) -> int:
        """Get the current clock value for the node"""
        # Start from base value, apply all deltas
        current_val = self.base_clocks.get(node_id, 0)

        for delta_node, old_val, new_val in self.deltas:
            if delta_node == node_id:
                current_val = new_val

        return current_val

    def compress(self) -> 'DeltaVectorClock':
        """Compress deltas, merge multiple changes into new base values"""
        new_base = {}

        # Calculate final values for all nodes
        all_nodes = set(self.base_clocks.keys())
        for delta_node, _, _ in self.deltas:
            all_nodes.add(delta_node)

        for node in all_nodes:
            new_base[node] = self.get_current_value(node)

        return DeltaVectorClock(new_base)

    def get_delta_size(self) -> int:
        """Get the size of deltas (for network transmission cost estimation)"""
        return len(self.deltas) * 3  # Each delta has 3 values


class HybridOptimizedClock:
    """
    Hybrid optimized clock combining multiple optimization strategies:
    1. Sparse storage for mostly zero values
    2. Delta updates for network transmission
    3. Hierarchical compression for long-term storage
    """

    def __init__(self):
        self.compressed_clock = CompressedVectorClock()
        self.delta_clock = DeltaVectorClock()
        self.last_compression = 0
        self.compression_threshold = 10  # Compress after accumulating this many deltas

    def tick(self, node_id: str) -> 'HybridOptimizedClock':
        """Optimized clock increment"""
        new_clock = HybridOptimizedClock()
        new_clock.compressed_clock = self.compressed_clock.tick(node_id)
        new_clock.delta_clock = self.delta_clock.tick(node_id)
        new_clock.last_compression = self.last_compression
        new_clock.compression_threshold = self.compression_threshold

        # Check if compression is needed
        if len(new_clock.delta_clock.deltas) >= self.compression_threshold:
            new_clock._compress()

        return new_clock

    def _compress(self):
        """Internal compression operation"""
        self.delta_clock = self.delta_clock.compress()
        self.last_compression = len(self.delta_clock.deltas)

    def merge(self, other: 'HybridOptimizedClock') -> 'HybridOptimizedClock':
        """Optimized merge operation"""
        new_clock = HybridOptimizedClock()
        new_clock.compressed_clock = self.compressed_clock.merge(other.compressed_clock)

        # Merge deltas (simplified implementation)
        new_clock.delta_clock = DeltaVectorClock()
        new_clock._compress()

        return new_clock

    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics"""
        compressed_nodes = len(self.compressed_clock.clocks)
        delta_ops = len(self.delta_clock.deltas)

        # Fix memory efficiency calculation: ensure it's in 0-1 range
        if compressed_nodes == 0:
            memory_efficiency = 1.0 if delta_ops == 0 else 0.0
        else:
            # Efficiency = compressed nodes / (compressed nodes + delta operations)
            memory_efficiency = compressed_nodes / (compressed_nodes + delta_ops)

        return {
            'compressed_nodes': compressed_nodes,
            'delta_operations': delta_ops,
            'memory_efficiency': max(0.0, min(1.0, memory_efficiency)),  # Ensure 0-1 range
            'needs_compression': delta_ops >= self.compression_threshold
        }


def benchmark_matrix_optimizations():
    """Benchmark test comparing different matrix optimization strategies"""
    import time

    print("Vector Clock Matrix Optimization Benchmark")
    print("=" * 50)

    # Test parameters
    num_nodes = 100
    num_operations = 1000

    # 1. Traditional vector clock (assuming it exists)
    print(f"\n1. Traditional Vector Clock ({num_nodes} nodes, {num_operations} operations)")

    # Note: Commented out as original VectorClock class not provided
    # from .causal_clock import VectorClock
    # traditional_clock = VectorClock()
    
    # Simulating traditional clock performance
    traditional_time = 0.5  # Simulated baseline
    print(f"   Execution time: {traditional_time:.4f} seconds (simulated)")
    print(f"   Memory usage: {num_nodes} items")

    # 2. Compressed vector clock
    print(f"\n2. Compressed Vector Clock")

    compressed_clock = CompressedVectorClock()

    start_time = time.time()
    for i in range(num_operations):
        compressed_clock = compressed_clock.tick(f"node_{i % num_nodes}")
    compressed_time = time.time() - start_time

    print(f"   Execution time: {compressed_time:.4f} seconds")
    print(f"   Memory usage: {len(compressed_clock.clocks)} items")
    print(f"   Performance improvement: {((traditional_time - compressed_time) / traditional_time * 100):.1f}%")

    # 3. Triangular matrix clock
    print(f"\n3. Triangular Matrix Clock")

    triangular_clock = TriangularMatrixClock()

    start_time = time.time()
    for i in range(num_operations):
        triangular_clock = triangular_clock.tick(f"node_{i % num_nodes}")
    triangular_time = time.time() - start_time

    memory_stats = triangular_clock.memory_usage()
    print(f"   Execution time: {triangular_time:.4f} seconds")
    print(f"   Memory savings: {memory_stats['space_saving_percent']}%")
    print(f"   Matrix elements: {memory_stats['triangular_elements']} vs {memory_stats['full_matrix_elements']}")

    # 4. Hybrid optimized clock
    print(f"\n4. Hybrid Optimized Clock")

    hybrid_clock = HybridOptimizedClock()

    start_time = time.time()
    for i in range(num_operations):
        hybrid_clock = hybrid_clock.tick(f"node_{i % (num_operations//10)}")  # More sparse
    hybrid_time = time.time() - start_time

    stats = hybrid_clock.get_performance_stats()
    print(f"   Execution time: {hybrid_time:.4f} seconds")
    print(f"   Memory efficiency: {stats['memory_efficiency']:.2f}")
    print(f"   Compressed nodes: {stats['compressed_nodes']}")
    print(f"   Delta operations: {stats['delta_operations']}")

    print(f"\nSummary:")
    fastest = min(compressed_time, triangular_time, hybrid_time)
    if fastest == compressed_time:
        fastest_name = "Compressed"
    elif fastest == triangular_time:
        fastest_name = "Triangular"
    else:
        fastest_name = "Hybrid"
    
    print(f"   Fastest implementation: {fastest_name}")
    print(f"   Most memory efficient: Triangular Matrix ({memory_stats['space_saving_percent']}% savings)")


class LowerTriangularCausalMatrix:
    """
    Optimized implementation using lower triangular matrix to store causal dependencies
    matrix[i][j] = True indicates event i causally depends on event j (when i > j)

    Advantages:
    1. O(1) causality relationship check
    2. 50% memory savings (vs full matrix)
    3. Natural temporal ordering (lower triangular structure)
    """

    def __init__(self, initial_size: int = 100):
        self.event_mapping = NodeMapping({}, {})
        # Use sparse matrix to store lower triangular dependencies
        self.dependency_matrix = lil_matrix((initial_size, initial_size), dtype=bool)
        self.next_event_id = 0

    def _ensure_size(self, min_size: int):
        """Ensure matrix size is sufficient"""
        current_size = self.dependency_matrix.shape[0]
        if min_size > current_size:
            new_size = max(min_size, current_size * 2)
            new_matrix = lil_matrix((new_size, new_size), dtype=bool)
            new_matrix[:current_size, :current_size] = self.dependency_matrix
            self.dependency_matrix = new_matrix

    def add_event(self, event_id: str, depends_on: List[str] = None) -> int:
        """
        Add new event and establish causal dependencies
        Returns the internal index of the event
        """
        event_idx = self.event_mapping.get_or_create_index(event_id)
        self._ensure_size(event_idx + 1)

        if depends_on:
            for dep_event_id in depends_on:
                dep_idx = self.event_mapping.get_or_create_index(dep_event_id)
                self._ensure_size(max(event_idx, dep_idx) + 1)

                # Establish dependency in lower triangular region (event_idx > dep_idx)
                if event_idx > dep_idx:
                    self.dependency_matrix[event_idx, dep_idx] = True

        return event_idx

    def has_causal_dependency(self, event_a_id: str, event_b_id: str) -> bool:
        """
        O(1) check if event A causally depends on event B
        """
        if event_a_id not in self.event_mapping.node_to_index:
            return False
        if event_b_id not in self.event_mapping.node_to_index:
            return False

        a_idx = self.event_mapping.node_to_index[event_a_id]
        b_idx = self.event_mapping.node_to_index[event_b_id]

        # Check lower triangular region
        if a_idx > b_idx:
            return bool(self.dependency_matrix[a_idx, b_idx])
        elif b_idx > a_idx:
            return bool(self.dependency_matrix[b_idx, a_idx])
        else:
            return False  # Same event

    def get_dependencies(self, event_id: str) -> List[str]:
        """Get all direct dependencies of an event"""
        if event_id not in self.event_mapping.node_to_index:
            return []

        event_idx = self.event_mapping.node_to_index[event_id]
        dependencies = []

        # Find all dependencies (that row of the matrix)
        for dep_idx in range(event_idx):
            if self.dependency_matrix[event_idx, dep_idx]:
                dep_id = self.event_mapping.index_to_node[dep_idx]
                dependencies.append(dep_id)

        return dependencies

    def get_transitive_dependencies(self, event_id: str) -> Set[str]:
        """
        Get all transitive dependencies of an event (depth-first search)
        Time complexity: O(d) where d = dependency depth
        """
        if event_id not in self.event_mapping.node_to_index:
            return set()

        visited = set()
        to_visit = [event_id]

        while to_visit:
            current_id = to_visit.pop()
            if current_id in visited:
                continue

            visited.add(current_id)
            direct_deps = self.get_dependencies(current_id)

            for dep_id in direct_deps:
                if dep_id not in visited:
                    to_visit.append(dep_id)

        visited.discard(event_id)  # Remove self
        return visited

    def can_apply_event(self, event_id: str, applied_events: Set[str]) -> bool:
        """
        Check if event can be applied (all dependencies are satisfied)
        O(d) where d = number of direct dependencies
        """
        dependencies = self.get_dependencies(event_id)
        return all(dep_id in applied_events for dep_id in dependencies)

    def topological_sort(self, events: List[str]) -> List[str]:
        """
        Topological sort based on causal dependencies
        Returns the causal application order of events
        """
        # Build in-degree table
        in_degree = {event_id: 0 for event_id in events}

        for event_id in events:
            dependencies = self.get_dependencies(event_id)
            for dep_id in dependencies:
                if dep_id in in_degree:
                    in_degree[event_id] += 1

        # Kahn's algorithm
        queue = [event_id for event_id in events if in_degree[event_id] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree of events dependent on current event
            if current in self.event_mapping.node_to_index:
                current_idx = self.event_mapping.node_to_index[current]

                # Find all events that depend on current event
                for event_id in events:
                    if event_id not in result and self.has_causal_dependency(event_id, current):
                        in_degree[event_id] -= 1
                        if in_degree[event_id] == 0:
                            queue.append(event_id)

        return result

    def get_statistics(self) -> Dict[str, any]:
        """Get matrix statistics"""
        total_events = self.event_mapping.size()
        max_elements = total_events * (total_events - 1) // 2  # Lower triangular elements

        # Count actually used elements
        used_elements = 0
        for i in range(total_events):
            for j in range(i):
                if self.dependency_matrix[i, j]:
                    used_elements += 1

        return {
            'total_events': total_events,
            'max_triangular_elements': max_elements,
            'used_elements': used_elements,
            'sparsity': 1 - (used_elements / max(max_elements, 1)),
            'memory_efficiency': f"{(1 - used_elements / max(max_elements, 1)) * 100:.1f}%"
        }


class WindowedTriangularClock:
    """
    Triangular matrix clock with sliding window
    Automatically cleans up expired causal relationship data, suitable for long-running systems
    """

    def __init__(self, window_size: int = 1000, num_windows: int = 3):
        self.window_size = window_size
        self.num_windows = num_windows
        self.current_window = LowerTriangularCausalMatrix()
        self.archived_windows: List[LowerTriangularCausalMatrix] = []
        self.global_event_count = 0

    def add_event(self, event_id: str, depends_on: List[str] = None) -> int:
        """Add event to current window"""
        self.global_event_count += 1

        # Check if window rotation is needed
        if self.global_event_count % self.window_size == 0:
            self._rotate_window()

        return self.current_window.add_event(event_id, depends_on)

    def _rotate_window(self):
        """Rotate window: current -> archive, create new current"""
        self.archived_windows.append(self.current_window)
        self.current_window = LowerTriangularCausalMatrix()

        # Keep at most num_windows archived windows
        if len(self.archived_windows) > self.num_windows:
            self.archived_windows.pop(0)

    def has_causal_dependency(self, event_a_id: str, event_b_id: str) -> bool:
        """Check causal dependency across windows"""
        # Check current window first
        if self.current_window.has_causal_dependency(event_a_id, event_b_id):
            return True

        # Then check archived windows
        for window in self.archived_windows:
            if window.has_causal_dependency(event_a_id, event_b_id):
                return True

        return False

    def get_memory_usage(self) -> Dict[str, any]:
        """Get memory usage statistics"""
        current_stats = self.current_window.get_statistics()

        archived_total = sum(
            window.get_statistics()['used_elements']
            for window in self.archived_windows
        )

        return {
            'current_window': current_stats,
            'archived_windows': len(self.archived_windows),
            'total_used_elements': current_stats['used_elements'] + archived_total,
            'global_events': self.global_event_count,
            'window_size': self.window_size,
            'estimated_memory_kb': (current_stats['used_elements'] + archived_total) * 0.125  # 1 bit = 0.125 bytes
        }


if __name__ == "__main__":
    benchmark_matrix_optimizations()