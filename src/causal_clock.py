"""
Vector Clock implementation for tracking causality in the CausalMesh system.
"""
from typing import Dict, Any

class VectorClock:
    """
    Represents a vector clock for tracking causal dependencies.
    """

    def __init__(self, clocks: Dict[str, int] = None):
        self.clocks: Dict[str, int] = clocks if clocks is not None else {}

    def tick(self, node_id: str):
        """Increment the clock for the given node."""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1
        return self

    def merge(self, other: 'VectorClock'):
        """Merge another vector clock into this one."""
        for node_id, timestamp in other.clocks.items():
            self.clocks[node_id] = max(self.clocks.get(node_id, 0), timestamp)
        return self

    def __le__(self, other: 'VectorClock') -> bool:
        """Less than or equal to comparison (causally precedes or is concurrent)."""
        for node_id, timestamp in self.clocks.items():
            if timestamp > other.clocks.get(node_id, 0):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self.clocks == other.clocks

    def __ne__(self, other: object) -> bool:
        """Inequality comparison."""
        return not self.__eq__(other)
    
    def __ge__(self, other: 'VectorClock') -> bool:
        """Greater than or equal to comparison."""
        return other.__le__(self)

    def copy(self) -> 'VectorClock':
        """Return a copy of this vector clock."""
        return VectorClock(self.clocks.copy())

    def to_dict(self) -> Dict[str, int]:
        """Convert to a dictionary for serialization."""
        return self.clocks

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'VectorClock':
        """Create a VectorClock from a dictionary."""
        return cls(data)

    def __str__(self) -> str:
        return str(self.clocks)
