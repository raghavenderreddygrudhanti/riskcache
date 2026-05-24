"""
RiskCache — risk-aware memory system.

A simple in-memory implementation that demonstrates the risk-aware retention
concept. In production this would wrap Bitcache; for the benchmark we use a
pure-Python implementation to avoid Rust compilation dependencies.

Each memory has:
  - content (text)
  - embedding (vector for similarity search)
  - importance (0.0-1.0, decays over time)
  - risk_class (CRITICAL/IMPORTANT/NORMAL/LOW_VALUE)
  - access_count
  - created_at / last_accessed (simulated timestamps)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from classifier import RiskClass


@dataclass
class Memory:
    id: str
    content: str
    embedding: np.ndarray
    importance: float
    risk_class: RiskClass
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0


class RiskCacheMemory:
    """Risk-aware agent memory with per-class decay and eviction policies."""

    # Decay multipliers per risk class (relative to base_decay_rate)
    DECAY_MULTIPLIERS = {
        RiskClass.CRITICAL: 0.0,    # no decay
        RiskClass.IMPORTANT: 0.3,   # 30% of base rate
        RiskClass.NORMAL: 1.0,      # full base rate
        RiskClass.LOW_VALUE: 3.0,   # 3x base rate
    }

    # Eviction order: higher number = evict first
    EVICTION_PRIORITY = {
        RiskClass.CRITICAL: 0,      # never evict
        RiskClass.IMPORTANT: 1,
        RiskClass.NORMAL: 2,
        RiskClass.LOW_VALUE: 3,     # evict first
    }

    def __init__(self, capacity: int, base_decay_rate: float = 0.05,
                 reinforce_amount: float = 0.1):
        self.capacity = capacity
        self.base_decay_rate = base_decay_rate
        self.reinforce_amount = reinforce_amount
        self.memories: Dict[str, Memory] = {}
        self._next_id = 0
        self._current_time = 0.0  # simulated time (days)

    def advance_time(self, days: float):
        """Simulate time passing (for benchmarking)."""
        self._current_time += days

    def save(self, content: str, embedding: np.ndarray, importance: float,
             risk_class: RiskClass, memory_id: Optional[str] = None) -> str:
        """Store a memory with risk classification."""
        if memory_id is None:
            memory_id = f"mem_{self._next_id}"
            self._next_id += 1

        self.memories[memory_id] = Memory(
            id=memory_id,
            content=content,
            embedding=embedding / (np.linalg.norm(embedding) + 1e-10),
            importance=min(1.0, max(0.0, importance)),
            risk_class=risk_class,
            access_count=0,
            created_at=self._current_time,
            last_accessed=self._current_time,
        )

        # Evict if over capacity
        while len(self.memories) > self.capacity:
            if not self._evict_one():
                break  # all remaining are CRITICAL, allow over-capacity

        return memory_id

    def retrieve(self, query_embedding: np.ndarray, k: int = 5,
                 min_importance: float = 0.0) -> List[Memory]:
        """Retrieve top-k memories by similarity, applying decay first."""
        self._apply_decay()

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        scored = []
        for mem in self.memories.values():
            if mem.importance < min_importance:
                continue
            sim = float(np.dot(query_norm, mem.embedding))
            scored.append((sim, mem))

        scored.sort(key=lambda x: -x[0])
        results = []
        for sim, mem in scored[:k]:
            # Reinforce on access
            mem.access_count += 1
            mem.last_accessed = self._current_time
            mem.importance = min(1.0, mem.importance + self.reinforce_amount)
            results.append(mem)

        return results

    def get_critical_memories(self) -> List[Memory]:
        """Return all CRITICAL memories (for prompt injection)."""
        return [m for m in self.memories.values() if m.risk_class == RiskClass.CRITICAL]

    def get_all_memories(self) -> List[Memory]:
        """Return all memories (for inspection)."""
        return list(self.memories.values())

    def stats(self) -> Dict:
        by_class = {rc: 0 for rc in RiskClass}
        for m in self.memories.values():
            by_class[m.risk_class] += 1
        return {
            "total": len(self.memories),
            "capacity": self.capacity,
            "by_class": {rc.value: count for rc, count in by_class.items()},
        }

    def _apply_decay(self):
        """Apply time-based decay to all memories based on risk class."""
        for mem in self.memories.values():
            days_since_access = self._current_time - mem.last_accessed
            if days_since_access <= 0:
                continue
            multiplier = self.DECAY_MULTIPLIERS[mem.risk_class]
            decay = self.base_decay_rate * multiplier * days_since_access
            mem.importance = max(0.0, mem.importance - decay)

    def _evict_one(self):
        """Evict the lowest-priority memory. Never evicts CRITICAL."""
        candidates = [
            (self.EVICTION_PRIORITY[m.risk_class], m.importance, mid)
            for mid, m in self.memories.items()
            if m.risk_class != RiskClass.CRITICAL  # never evict critical
        ]

        if not candidates:
            # All memories are CRITICAL — allow over-capacity rather than
            # infinite loop. In production, surface a warning.
            return False

        # Sort by: eviction priority DESC (high = evict first), then importance ASC
        candidates.sort(key=lambda x: (-x[0], x[1]))
        worst_id = candidates[0][2]
        del self.memories[worst_id]
        return True


class UniformMemory:
    """Baseline: all memories treated equally (no risk awareness).
    Same interface as RiskCacheMemory but ignores risk_class."""

    def __init__(self, capacity: int, base_decay_rate: float = 0.05,
                 reinforce_amount: float = 0.1):
        self.capacity = capacity
        self.base_decay_rate = base_decay_rate
        self.reinforce_amount = reinforce_amount
        self.memories: Dict[str, Memory] = {}
        self._next_id = 0
        self._current_time = 0.0

    def advance_time(self, days: float):
        self._current_time += days

    def save(self, content: str, embedding: np.ndarray, importance: float,
             risk_class: RiskClass = RiskClass.NORMAL, memory_id: Optional[str] = None) -> str:
        """Store memory — ignores risk_class, treats all equally."""
        if memory_id is None:
            memory_id = f"mem_{self._next_id}"
            self._next_id += 1

        self.memories[memory_id] = Memory(
            id=memory_id,
            content=content,
            embedding=embedding / (np.linalg.norm(embedding) + 1e-10),
            importance=min(1.0, max(0.0, importance)),
            risk_class=risk_class,  # stored but not used
            access_count=0,
            created_at=self._current_time,
            last_accessed=self._current_time,
        )

        while len(self.memories) > self.capacity:
            self._evict_one()

        return memory_id

    def retrieve(self, query_embedding: np.ndarray, k: int = 5,
                 min_importance: float = 0.0) -> List[Memory]:
        self._apply_decay()
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        scored = []
        for mem in self.memories.values():
            if mem.importance < min_importance:
                continue
            sim = float(np.dot(query_norm, mem.embedding))
            scored.append((sim, mem))
        scored.sort(key=lambda x: -x[0])
        results = []
        for sim, mem in scored[:k]:
            mem.access_count += 1
            mem.last_accessed = self._current_time
            mem.importance = min(1.0, mem.importance + self.reinforce_amount)
            results.append(mem)
        return results

    def get_critical_memories(self) -> List[Memory]:
        return [m for m in self.memories.values() if m.risk_class == RiskClass.CRITICAL]

    def get_all_memories(self) -> List[Memory]:
        return list(self.memories.values())

    def _apply_decay(self):
        for mem in self.memories.values():
            days_since = self._current_time - mem.last_accessed
            if days_since <= 0:
                continue
            decay = self.base_decay_rate * days_since
            mem.importance = max(0.0, mem.importance - decay)

    def _evict_one(self):
        """Evict lowest importance — no risk awareness."""
        if not self.memories:
            return
        worst_id = min(self.memories.items(), key=lambda x: x[1].importance)[0]
        del self.memories[worst_id]
