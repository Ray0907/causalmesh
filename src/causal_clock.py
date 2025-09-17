"""
Causal Vector Clock implementation for tracking causality relationships.
Inspired by distributed systems research on causal consistency.
"""
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass, field


@dataclass
class VectorClock:
    """Vector clock for tracking causal relationships between events."""

    clocks: Dict[str, int] = field(default_factory=dict)

    def tick(self, node_id: str) -> 'VectorClock':
        """Increment the clock for a specific node."""
        new_clocks = self.clocks.copy()
        new_clocks[node_id] = new_clocks.get(node_id, 0) + 1
        return VectorClock(new_clocks)

    def merge(self, other: 'VectorClock') -> 'VectorClock':
        """Merge two vector clocks, taking the maximum for each node."""
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        new_clocks = {}

        for node in all_nodes:
            new_clocks[node] = max(
                self.clocks.get(node, 0),
                other.clocks.get(node, 0)
            )

        return VectorClock(new_clocks)

    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this clock happens before another (partial ordering)."""
        if not self.clocks or not other.clocks:
            return False

        # Self <= other and self != other
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())

        less_than_or_equal = True
        strictly_less = False

        for node in all_nodes:
            self_val = self.clocks.get(node, 0)
            other_val = other.clocks.get(node, 0)

            if self_val > other_val:
                less_than_or_equal = False
                break
            elif self_val < other_val:
                strictly_less = True

        return less_than_or_equal and strictly_less

    def concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if two clocks are concurrent (neither happens before the other)."""
        return not self.happens_before(other) and not other.happens_before(self)

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for serialization."""
        return self.clocks.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'VectorClock':
        """Create from dictionary."""
        return cls(data.copy())

    def __str__(self) -> str:
        return f"VC{self.clocks}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, VectorClock):
            return False
        return self.clocks == other.clocks


@dataclass
class CausalEvent:
    """An event with causal metadata."""

    event_id: str
    node_id: str
    timestamp: VectorClock
    operation: str
    key: str
    value: any
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'node_id': self.node_id,
            'timestamp': self.timestamp.to_dict(),
            'operation': self.operation,
            'key': self.key,
            'value': self.value,
            'dependencies': self.dependencies
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CausalEvent':
        """Create from dictionary."""
        return cls(
            event_id=data['event_id'],
            node_id=data['node_id'],
            timestamp=VectorClock.from_dict(data['timestamp']),
            operation=data['operation'],
            key=data['key'],
            value=data['value'],
            dependencies=data.get('dependencies', [])
        )


class CausalConsistencyChecker:
    """Utility for checking causal consistency properties."""

    @staticmethod
    def can_apply_event(event: CausalEvent, node_clock: VectorClock,
                       applied_events: Dict[str, CausalEvent]) -> bool:
        """
        Check if an event can be applied based on causal dependencies.
        An event can be applied if all its causal dependencies have been satisfied.
        """
        # Check if all dependency events have been applied
        for dep_id in event.dependencies:
            if dep_id not in applied_events:
                return False

        # Check if the event's timestamp is compatible with current node state
        # For causal consistency, we need to ensure we don't violate causality
        event_node_clock = event.timestamp.clocks.get(event.node_id, 0)
        current_node_clock = node_clock.clocks.get(event.node_id, 0)

        # The event should be the next expected event from its originating node
        return event_node_clock == current_node_clock + 1

    @staticmethod
    def order_events_causally(events: List[CausalEvent]) -> List[CausalEvent]:
        """
        Order events according to causal dependencies.
        This implements a topological sort based on causal relationships.
        """
        if not events:
            return []

        # Build dependency graph
        event_map = {e.event_id: e for e in events}
        ordered = []
        applied = set()

        def can_apply(event: CausalEvent) -> bool:
            return all(dep in applied for dep in event.dependencies)

        # Repeatedly find events that can be applied
        remaining = events.copy()
        while remaining:
            # Find events that can be applied now
            applicable = [e for e in remaining if can_apply(e)]

            if not applicable:
                # If no events can be applied, there might be a cycle or missing dependency
                # For this POC, we'll break ties using timestamp comparison
                applicable = [min(remaining, key=lambda e: str(e.timestamp.clocks))]

            # Apply the first applicable event
            next_event = applicable[0]
            ordered.append(next_event)
            applied.add(next_event.event_id)
            remaining.remove(next_event)

        return ordered