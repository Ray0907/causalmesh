"""
Conflict-free Replicated Data Type (CRDT) implementation for the causal mesh.
Provides eventually consistent data structures that can be merged without conflicts.
"""
from typing import Dict, Set, Any, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import uuid


class CRDT(ABC):
    """Base class for Conflict-free Replicated Data Types."""

    @abstractmethod
    def merge(self, other: 'CRDT') -> 'CRDT':
        """Merge this CRDT with another, resolving conflicts automatically."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRDT':
        """Deserialize from dictionary."""
        pass


@dataclass
class LWWRegister(CRDT):
    """Last-Writer-Wins Register - resolves conflicts using timestamps."""

    value: Any = None
    timestamp: float = 0.0
    node_id: str = ""

    def set(self, value: Any, node_id: str) -> 'LWWRegister':
        """Set a new value with current timestamp."""
        return LWWRegister(
            value=value,
            timestamp=time.time(),
            node_id=node_id
        )

    def merge(self, other: 'LWWRegister') -> 'LWWRegister':
        """Merge with another register, keeping the later timestamp."""
        if self.timestamp > other.timestamp:
            return self
        elif self.timestamp < other.timestamp:
            return other
        else:
            # Tie-break using node_id for deterministic results
            if self.node_id >= other.node_id:
                return self
            else:
                return other

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'LWWRegister',
            'value': self.value,
            'timestamp': self.timestamp,
            'node_id': self.node_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LWWRegister':
        return cls(
            value=data['value'],
            timestamp=data['timestamp'],
            node_id=data['node_id']
        )


@dataclass
class GSet(CRDT):
    """Grow-only Set - elements can only be added, never removed."""

    elements: Set[Any] = field(default_factory=set)

    def add(self, element: Any) -> 'GSet':
        """Add an element to the set."""
        new_elements = self.elements.copy()
        new_elements.add(element)
        return GSet(new_elements)

    def contains(self, element: Any) -> bool:
        """Check if element is in the set."""
        return element in self.elements

    def merge(self, other: 'GSet') -> 'GSet':
        """Merge with another G-Set (union of elements)."""
        return GSet(self.elements | other.elements)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'GSet',
            'elements': list(self.elements)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GSet':
        return cls(set(data['elements']))


@dataclass
class ORSet(CRDT):
    """Observed-Remove Set - supports both additions and removals."""

    added: Dict[Any, Set[str]] = field(default_factory=dict)  # element -> set of unique tags
    removed: Dict[Any, Set[str]] = field(default_factory=dict)  # element -> set of unique tags

    def add(self, element: Any, node_id: str) -> 'ORSet':
        """Add an element with a unique tag."""
        unique_tag = f"{node_id}:{uuid.uuid4().hex[:8]}"
        new_added = self.added.copy()
        new_added.setdefault(element, set()).add(unique_tag)

        return ORSet(new_added, self.removed.copy())

    def remove(self, element: Any) -> 'ORSet':
        """Remove an element by marking all its current tags as removed."""
        if element not in self.added:
            return self

        new_removed = self.removed.copy()
        tags_to_remove = self.added[element].copy()
        new_removed[element] = new_removed.get(element, set()) | tags_to_remove

        return ORSet(self.added.copy(), new_removed)

    def contains(self, element: Any) -> bool:
        """Check if element is in the set (not removed)."""
        if element not in self.added:
            return False

        added_tags = self.added[element]
        removed_tags = self.removed.get(element, set())

        # Element exists if it has tags that haven't been removed
        return len(added_tags - removed_tags) > 0

    def elements(self) -> Set[Any]:
        """Get all elements currently in the set."""
        result = set()
        for element in self.added:
            if self.contains(element):
                result.add(element)
        return result

    def merge(self, other: 'ORSet') -> 'ORSet':
        """Merge with another OR-Set."""
        # Merge added tags
        new_added = {}
        all_elements = set(self.added.keys()) | set(other.added.keys())

        for element in all_elements:
            self_tags = self.added.get(element, set())
            other_tags = other.added.get(element, set())
            new_added[element] = self_tags | other_tags

        # Merge removed tags
        new_removed = {}
        all_removed_elements = set(self.removed.keys()) | set(other.removed.keys())

        for element in all_removed_elements:
            self_removed = self.removed.get(element, set())
            other_removed = other.removed.get(element, set())
            new_removed[element] = self_removed | other_removed

        return ORSet(new_added, new_removed)

    def to_dict(self) -> Dict[str, Any]:
        # Convert sets to lists for JSON serialization
        added_serializable = {
            str(k): list(v) for k, v in self.added.items()
        }
        removed_serializable = {
            str(k): list(v) for k, v in self.removed.items()
        }

        return {
            'type': 'ORSet',
            'added': added_serializable,
            'removed': removed_serializable
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ORSet':
        # Convert lists back to sets
        added = {
            k: set(v) for k, v in data['added'].items()
        }
        removed = {
            k: set(v) for k, v in data['removed'].items()
        }

        return cls(added, removed)


class CRDTStore:
    """
    A key-value store using CRDTs for conflict-free replication.
    Supports different CRDT types for different use cases.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.data: Dict[str, CRDT] = {}

    def set_lww(self, key: str, value: Any) -> None:
        """Set a key using Last-Writer-Wins semantics."""
        if key in self.data and not isinstance(self.data[key], LWWRegister):
            raise ValueError(f"Key {key} exists with different CRDT type")

        current = self.data.get(key, LWWRegister())
        self.data[key] = current.set(value, self.node_id)

    def add_to_set(self, key: str, element: Any, allow_remove: bool = True) -> None:
        """Add element to a set (G-Set or OR-Set based on allow_remove)."""
        if key in self.data:
            if allow_remove and not isinstance(self.data[key], ORSet):
                raise ValueError(f"Key {key} exists as different set type")
            elif not allow_remove and not isinstance(self.data[key], GSet):
                raise ValueError(f"Key {key} exists as different set type")

        if allow_remove:
            current = self.data.get(key, ORSet())
            self.data[key] = current.add(element, self.node_id)
        else:
            current = self.data.get(key, GSet())
            self.data[key] = current.add(element)

    def remove_from_set(self, key: str, element: Any) -> None:
        """Remove element from an OR-Set."""
        if key not in self.data:
            return

        if not isinstance(self.data[key], ORSet):
            raise ValueError(f"Key {key} is not an OR-Set")

        self.data[key] = self.data[key].remove(element)

    def get(self, key: str) -> Any:
        """Get value for a key."""
        if key not in self.data:
            return None

        crdt = self.data[key]
        if isinstance(crdt, LWWRegister):
            return crdt.value
        elif isinstance(crdt, (GSet, ORSet)):
            return crdt.elements() if isinstance(crdt, ORSet) else crdt.elements
        else:
            return crdt

    def merge_store(self, other_store: 'CRDTStore') -> None:
        """Merge another CRDT store into this one."""
        for key, other_crdt in other_store.data.items():
            if key in self.data:
                # Merge CRDTs of the same type
                if type(self.data[key]) != type(other_crdt):
                    raise ValueError(f"CRDT type mismatch for key {key}")
                self.data[key] = self.data[key].merge(other_crdt)
            else:
                # Copy the CRDT
                self.data[key] = other_crdt

    def to_dict(self) -> Dict[str, Any]:
        """Serialize store to dictionary."""
        return {
            'node_id': self.node_id,
            'data': {k: v.to_dict() for k, v in self.data.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRDTStore':
        """Deserialize store from dictionary."""
        store = cls(data['node_id'])

        for key, crdt_data in data['data'].items():
            crdt_type = crdt_data['type']
            if crdt_type == 'LWWRegister':
                store.data[key] = LWWRegister.from_dict(crdt_data)
            elif crdt_type == 'GSet':
                store.data[key] = GSet.from_dict(crdt_data)
            elif crdt_type == 'ORSet':
                store.data[key] = ORSet.from_dict(crdt_data)
            else:
                raise ValueError(f"Unknown CRDT type: {crdt_type}")

        return store

    def keys(self) -> List[str]:
        """Get all keys in the store."""
        return list(self.data.keys())

    def __str__(self) -> str:
        items = []
        for key, crdt in self.data.items():
            if isinstance(crdt, LWWRegister):
                items.append(f"{key}: {crdt.value}")
            elif isinstance(crdt, (GSet, ORSet)):
                elements = crdt.elements() if isinstance(crdt, ORSet) else crdt.elements
                items.append(f"{key}: {elements}")

        return f"CRDTStore({self.node_id}): {{{', '.join(items)}}}"