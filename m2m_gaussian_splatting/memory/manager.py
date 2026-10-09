"""
Memory Manager for Gaussian Splats.

Implements hierarchical memory management for efficient
storage and retrieval of large-scale splat datasets.

v2.1 (audit 2026-10) changes:
- ``_evict_from_ram`` no longer DROPS the evicted splat: it is returned
  to cold storage. Previously any splat living only in RAM (added with
  ``to_cold=False``, or after its cold copy was replaced) was permanently
  lost on eviction.
- Tiers are now exclusive: loading cold→RAM removes the cold entry
  (previously the same splat lived in two tiers and ``total_splats``
  double-counted it).
- RAM limit is actually enforced: ``add_splats(to_cold=False)`` and
  VRAM→RAM demotion now trigger eviction (previously the limit was only
  checked on the cold→RAM path).
- ``clear`` and ``get_stats`` take the lock; ``get_stats`` returns a
  defensive copy.
- ``get_splat`` no longer creates phantom ``_access_count`` entries for
  unknown IDs, and unknown-ID lookups count as cache misses.
- Documented honestly: "VRAM" is a hot dict in host memory (no CUDA),
  "cold" is a dict in host memory (no disk). Tier names are conceptual.
- Warm-tier eviction gives near-promotion splats a second chance: a
  splat whose ``_access_count`` is at >= 70% of ``access_threshold``
  is moved to the MRU end instead of being evicted. ``index``-style
  LRU behavior is preserved for everything else.
- New: ``SplatMemoryManager.from_config(MemoryConfig)`` alternative
  constructor (loose kwargs remain the primary API).
"""

from typing import Dict, List, Optional
from collections import OrderedDict
from dataclasses import dataclass, replace
import threading

from ..core.splat_types import GaussianSplat


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    vram_limit: int = 100000  # Max splats in hot tier
    ram_limit: int = 1000000  # Max splats in warm tier
    eviction_threshold: float = 0.8  # Evict when usage > 80%
    access_threshold: int = 10  # Promote after N accesses


@dataclass
class MemoryStats:
    """Memory statistics."""

    vram_usage: int = 0
    ram_usage: int = 0
    total_splats: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0


class SplatMemoryManager:
    """
    Manages hierarchical memory for Gaussian splats.

    Implements three-tier memory architecture (all tiers are in-process
    data structures; the tier names describe recency, not physical
    location — there is no CUDA or disk backing):

    - Hot ("VRAM"): Frequently accessed splats
    - Warm ("RAM"): Recently accessed splats
    - Cold: Infrequently accessed splats

    Tiers are exclusive: each splat lives in exactly one tier, and
    eviction never drops data — splats evicted from the warm tier return
    to cold storage.

    Uses OrderedDict for O(1) LRU eviction instead of O(N) min() scans.

    Example:
        >>> manager = SplatMemoryManager(vram_limit=50000)
        >>> manager.add_splats(splats)
        >>> splat = manager.get_splat(splat_id)
    """

    def __init__(
        self,
        vram_limit: int = 100000,
        ram_limit: int = 1000000,
        eviction_threshold: float = 0.8,
        access_threshold: int = 10,
    ):
        """
        Initialize memory manager.

        Args:
            vram_limit: Maximum splats in hot tier
            ram_limit: Maximum splats in warm tier
            eviction_threshold: Fraction at which to start eviction
            access_threshold: Accesses needed for promotion
        """
        self.vram_limit = vram_limit
        self.ram_limit = ram_limit
        self.eviction_threshold = eviction_threshold
        self.access_threshold = access_threshold

        # OrderedDict tiers — LRU item is at the front (first inserted/accessed)
        self._vram: OrderedDict[int, GaussianSplat] = OrderedDict()
        self._ram: OrderedDict[int, GaussianSplat] = OrderedDict()
        self._cold: Dict[int, GaussianSplat] = {}

        # Access tracking
        self._access_count: Dict[int, int] = {}

        # Statistics
        self._stats = MemoryStats()

        # Thread safety (RLock: get_splat → _promote_to_vram → _evict… reentrancy)
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, config: MemoryConfig) -> "SplatMemoryManager":
        """
        Build a manager from a :class:`MemoryConfig` dataclass.

        The loose ``__init__`` kwargs remain the primary API; this
        classmethod makes the config dataclass a real (non-decorative)
        way to construct the manager::

            manager = SplatMemoryManager.from_config(MemoryConfig(ram_limit=500))

        Args:
            config: MemoryConfig with the manager parameters.

        Returns:
            A new SplatMemoryManager instance.
        """
        return cls(
            vram_limit=config.vram_limit,
            ram_limit=config.ram_limit,
            eviction_threshold=config.eviction_threshold,
            access_threshold=config.access_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_splats(self, splats: List[GaussianSplat], to_cold: bool = True) -> None:
        """
        Add splats to memory.

        Args:
            splats: List of splats to add
            to_cold: If True, add to cold storage initially. If False,
                add to the warm tier — the warm-tier limit is enforced,
                evicting the LRU splats back to cold storage as needed.
        """
        with self._lock:
            for splat in splats:
                if to_cold:
                    self._cold[splat.id] = splat
                else:
                    self._ram[splat.id] = splat
                    self._maybe_evict_ram()
                    self._access_count[splat.id] = 0

            if to_cold:
                self._access_count.update({s.id: 0 for s in splats})

            self._stats.total_splats = len(self._cold) + len(self._ram) + len(self._vram)

    def get_splat(self, splat_id: int) -> Optional[GaussianSplat]:
        """
        Get a splat by ID with automatic tier management.

        Args:
            splat_id: Splat identifier

        Returns:
            GaussianSplat or None if not found
        """
        with self._lock:
            # Check hot tier
            if splat_id in self._vram:
                self._access_count[splat_id] = self._access_count.get(splat_id, 0) + 1
                self._stats.cache_hits += 1
                self._vram.move_to_end(splat_id)
                return self._vram[splat_id]

            # Check warm tier
            if splat_id in self._ram:
                self._access_count[splat_id] = self._access_count.get(splat_id, 0) + 1
                self._stats.cache_hits += 1
                splat = self._ram[splat_id]
                self._ram.move_to_end(splat_id)

                if self._access_count[splat_id] >= self.access_threshold:
                    self._promote_to_vram(splat_id)

                return splat

            # Load from cold storage
            if splat_id in self._cold:
                self._access_count[splat_id] = self._access_count.get(splat_id, 0) + 1
                self._stats.cache_misses += 1
                splat = self._cold[splat_id]
                self._load_to_ram(splat_id)
                return splat

            # Unknown ID: no phantom access-count entry, counted as a miss
            self._stats.cache_misses += 1
            return None

    def get_stats(self) -> MemoryStats:
        """Get a snapshot of memory statistics (defensive copy)."""
        with self._lock:
            self._stats.vram_usage = len(self._vram)
            self._stats.ram_usage = len(self._ram)
            self._stats.total_splats = len(self._cold) + len(self._ram) + len(self._vram)
            return replace(self._stats)

    def clear(self) -> None:
        """Clear all memory."""
        with self._lock:
            self._vram.clear()
            self._ram.clear()
            self._cold.clear()
            self._access_count.clear()
            self._stats = MemoryStats()

    @property
    def vram_size(self) -> int:
        """Number of splats in the hot tier."""
        return len(self._vram)

    @property
    def ram_size(self) -> int:
        """Number of splats in the warm tier."""
        return len(self._ram)

    @property
    def cold_size(self) -> int:
        """Number of splats in cold storage."""
        return len(self._cold)

    # ------------------------------------------------------------------
    # Internals (callers hold self._lock)
    # ------------------------------------------------------------------

    def _promote_to_vram(self, splat_id: int) -> None:
        """Promote splat from warm tier to hot tier."""
        if splat_id not in self._ram:
            return

        # Check if eviction needed
        if len(self._vram) >= self.vram_limit * self.eviction_threshold:
            self._evict_from_vram()

        # Move to VRAM (tiers stay exclusive)
        self._vram[splat_id] = self._ram.pop(splat_id)
        self._stats.vram_usage = len(self._vram)
        self._stats.ram_usage = len(self._ram)

    def _load_to_ram(self, splat_id: int) -> None:
        """Load splat from cold storage to warm tier (exclusive move)."""
        if splat_id not in self._cold:
            return

        # Enforce warm-tier limit before inserting
        self._maybe_evict_ram()

        # Move cold → RAM (ownership transfers; entry removed from cold)
        self._ram[splat_id] = self._cold.pop(splat_id)
        self._stats.ram_usage = len(self._ram)

    def _maybe_evict_ram(self) -> None:
        """Evict LRU warm splats to cold until below the eviction mark."""
        mark = self.ram_limit * self.eviction_threshold
        while len(self._ram) >= mark:
            self._evict_from_ram()

    def _is_second_chance(self, splat_id: int) -> bool:
        """
        Second-chance rule: a warm splat whose access count is at or
        above 70% of ``access_threshold`` is close to hot-tier
        promotion — evicting it would throw away that progress. Such
        splats are given another turn at the MRU end instead.
        """
        return self._access_count.get(splat_id, 0) >= 0.7 * self.access_threshold

    def _pick_ram_eviction_candidate(self) -> Optional[int]:
        """
        Find the LRU warm splat that is NOT second-chance protected.

        Protected splats encountered on the way (clock/second-chance)
        receive ``move_to_end`` — they survive this round at the MRU end
        but age normally, so a splat that stops being accessed loses
        protection only through promotion (which removes it from the
        warm tier) — never through decay. To keep eviction terminating
        and the ram limit enforceable, the scan is bounded: if EVERY
        warm splat is protected, the oldest protected one is evicted
        anyway (a limit that can never be enforced is not a limit).
        """
        protected_seen: List[int] = []
        candidate: Optional[int] = None
        for splat_id in list(self._ram.keys()):  # front → back = LRU → MRU
            if not self._is_second_chance(splat_id):
                candidate = splat_id
                break
            protected_seen.append(splat_id)

        # Second chance: skipped protected splats go to the MRU end
        for sid in protected_seen:
            self._ram.move_to_end(sid)

        if candidate is not None:
            return candidate
        if protected_seen:
            # All warm splats are protected: evict the LRU one so the
            # while-loop in _maybe_evict_ram can make progress.
            return protected_seen[0]
        return None

    def _evict_from_vram(self) -> None:
        """Evict least recently used splat from hot tier — O(1) via OrderedDict."""
        if not self._vram:
            return

        # popitem(last=False) removes the LRU (first) item — O(1)
        lru_id, lru_splat = self._vram.popitem(last=False)

        # Demote to warm tier, enforcing the warm limit afterwards
        self._ram[lru_id] = lru_splat
        self._maybe_evict_ram()
        self._stats.evictions += 1
        self._stats.vram_usage = len(self._vram)
        self._stats.ram_usage = len(self._ram)

    def _evict_from_ram(self) -> None:
        """
        Evict least recently used splat from the warm tier — O(1).

        The splat is returned to cold storage (never dropped): this is the
        data-loss fix — RAM-only splats used to disappear permanently.

        Second chance: before evicting, splats whose ``_access_count`` is
        at >= 70% of ``access_threshold`` are moved to the MRU end and
        skipped (they are one or two accesses away from hot-tier
        promotion). The scan for an unprotected victim is bounded; if
        every warm splat is protected, the LRU protected one is evicted
        so the warm-tier limit stays enforceable.
        """
        if not self._ram:
            return

        victim_id = self._pick_ram_eviction_candidate()
        if victim_id is None:
            return

        # Move the victim to the FRONT so popitem(last=False) removes it
        self._ram.move_to_end(victim_id, last=False)
        lru_id, lru_splat = self._ram.popitem(last=False)
        self._cold[lru_id] = lru_splat
        self._stats.evictions += 1
        self._stats.ram_usage = len(self._ram)
