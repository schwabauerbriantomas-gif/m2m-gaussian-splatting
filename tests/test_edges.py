"""
Edge-case tests for HRM2Engine and SplatMemoryManager.

Each test pins the CURRENT behavior as a contract (docstrings state what
"current" means). If one of these starts failing after an engine change,
the change altered a documented contract — decide deliberately.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m2m_gaussian_splatting.core.hrm2_engine import HRM2Engine, generate_test_splats
from m2m_gaussian_splatting.core.splat_types import GaussianSplat
from m2m_gaussian_splatting.memory.manager import SplatMemoryManager


def _build_engine(n_splats=500, seed=11, n_probe=5):
    engine = HRM2Engine(n_coarse=5, n_fine=10, n_probe=n_probe, use_gpu=False)
    engine.add_splats(generate_test_splats(n_splats, seed=seed))
    engine.index()
    return engine


class TestQueryEdgeCases:
    """query() edge cases."""

    def test_query_huge_k_returns_all_splats(self):
        """query(k=10**6) over 500 splats returns exactly 500 results.

        CURRENT behavior: ``k`` is clamped to the number of IVF candidates
        (``k_actual = min(k, len(global_indices))``). With n_probe=5 probing
        all 5 coarse clusters of a 500-splat index, every splat is a
        candidate, so len(results) == len(engine.splats) == 500.

        NOTE: with a smaller n_probe the IVF path returns only the probed
        clusters' members (e.g. 498 with n_probe=3 here) — that truncation
        is inherent to IVF and equally documented behavior.
        """
        engine = _build_engine(n_splats=500, seed=11, n_probe=5)
        assert len(engine.splats) == 500

        results = engine.query(engine.embeddings[0], k=10**6)

        assert len(results) == 500
        # Every splat appears exactly once.
        ids = [s.id for s, _ in results]
        assert len(set(ids)) == 500
        # Distances sorted ascending.
        dists = [d for _, d in results]
        assert dists == sorted(dists)

    def test_query_k_zero_current_behavior(self):
        """query(k=0) returns an empty list — no argpartition crash.

        CURRENT behavior (pinned): ``k=0`` is silently accepted and returns
        ``[]``. The feared ``np.argpartition(x, kth=-1)`` crash does NOT
        happen: with k=0 the candidate set (500 splats, all clusters
        probed) is larger than k, so argpartition is called with kth=-1…
        which NumPy accepts as "partition by largest element" and the
        subsequent [:0] slice discards everything, yielding [].

        IMPROVEMENT NOTE (for the engine owner, not fixed here): k=0 is
        almost certainly a caller bug; a ``ValueError("k must be >= 1")``
        guard would surface it instead of silently returning [].
        """
        engine = _build_engine(n_splats=500, seed=11, n_probe=5)

        results = engine.query(engine.embeddings[0], k=0)

        assert results == []
        assert isinstance(results, list)

    def test_batch_query_accepts_1d_and_2d_shapes(self):
        """batch_query accepts both (1, 640) and (640,) inputs, same output.

        CURRENT behavior: a 1-D input is reshaped to (1, D) internally, so
        both forms return a 1-element list of result lists with identical
        splat ids and distances.
        """
        engine = _build_engine(n_splats=300, seed=17, n_probe=5)
        D = engine.embedding_dim
        q = engine.embeddings[3]
        assert q.shape == (D,)  # 640

        results_2d = engine.batch_query(q.reshape(1, D), k=5)
        results_1d = engine.batch_query(q, k=5)

        assert len(results_2d) == 1
        assert len(results_1d) == 1
        assert all(len(r) == 5 for r in results_1d + results_2d)
        assert [s.id for s, _ in results_2d[0]] == [s.id for s, _ in results_1d[0]]
        for (_, d1), (_, d2) in zip(results_2d[0], results_1d[0]):
            np.testing.assert_allclose(d1, d2, rtol=1e-5)


class TestBuildEdgeCases:
    """add_splats()/index() edge cases."""

    def test_add_empty_then_index_is_noop(self):
        """add_splats([]) followed by index() is a clean no-op returning 0.0.

        CURRENT behavior (pinned): ``index()`` short-circuits on an empty
        splat list — it returns 0.0 immediately WITHOUT building anything
        and leaves ``_is_indexed`` False. A subsequent ``query`` raises the
        standard ``RuntimeError("Index not built. Call index() first.")``.
        (This is the return-0.0 branch, not an exception.)
        """
        engine = HRM2Engine(n_coarse=5, n_fine=10, use_gpu=False)
        engine.add_splats([])

        assert engine.splats == []

        build_time = engine.index()

        assert build_time == 0.0
        assert not engine._is_indexed
        assert engine.embeddings is None

        with pytest.raises(RuntimeError, match="Index not built"):
            engine.query(np.zeros(640, dtype=np.float32), k=3)

    def test_duplicate_ids_last_one_wins_for_lookup(self):
        """Duplicate splat ids: container semantics documented per component.

        CURRENT behavior (pinned as the contract):

        - SplatMemoryManager: dict-backed per id — the LAST splat added
          with a given id wins; ``get_splat(id)`` returns that object and
          ``total_splats`` counts the id once.

        - HRM2Engine: the splat LIST keeps BOTH entries (no dedup); both
          are indexed and both can appear in query results as distinct
          result rows. Id-based consumers must treat engine ids as
          positional, not unique.
        """
        first = generate_test_splats(1, seed=100)[0]
        second = generate_test_splats(1, seed=101)[0]
        first.id = 7
        second.id = 7
        assert first.position[0] != second.position[0]  # genuinely distinct

        # --- Manager: last added wins, single count ---
        manager = SplatMemoryManager()
        manager.add_splats([first, second])

        stats = manager.get_stats()
        assert stats.total_splats == 1
        got = manager.get_splat(7)
        assert got is second  # last one wins
        assert got is not first

        # --- Engine: both kept, both searchable ---
        engine = HRM2Engine(n_coarse=2, n_fine=4, n_probe=2, use_gpu=False)
        engine.add_splats([first, second])
        engine.index()

        assert len(engine.splats) == 2

        # n_coarse clamps to max(1, min(2, 2//10)) = 1 → single cluster
        # holding both splats; k=10 must surface both objects.
        results = engine.query(engine.embeddings[0], k=10)
        returned_objects = {id(s) for s, _ in results}
        assert returned_objects == {id(first), id(second)}


class TestSplatLookupEdgeCases:
    """get_splat() edge cases."""

    def test_get_splat_minus_one_is_none(self):
        """get_splat(-1) on a populated manager returns None (a miss).

        CURRENT behavior: -1 (Python's sentinel "last element" index, a
        classic bug source) is treated as an ordinary unknown id — None is
        returned, the lookup counts as a cache miss, and no phantom state
        is created. Contrast with get_splat(0), which is a real hit here.
        """
        manager = SplatMemoryManager()
        manager.add_splats(generate_test_splats(5, seed=3))

        assert manager.get_splat(0) is not None  # sanity: real id works

        result = manager.get_splat(-1)

        assert result is None
        stats = manager.get_stats()
        assert stats.cache_misses >= 1
        # No phantom entry was created for -1.
        assert -1 not in manager._access_count

    def test_get_splat_unknown_id_is_none(self):
        """get_splat on a never-added id returns None without side effects."""
        manager = SplatMemoryManager()
        manager.add_splats(generate_test_splats(3, seed=4))

        assert manager.get_splat(99999) is None
        assert manager.get_stats().total_splats == 3


class TestEngineStateAfterAdd:
    """add_splats invalidates a previously built index."""

    def test_add_splats_after_index_invalidates(self):
        """Adding splats after index() resets _is_indexed (must re-index).

        CURRENT behavior: ``add_splats`` sets ``_is_indexed = False``;
        querying before re-indexing raises RuntimeError. Queries after a
        rebuild include the new splats. Pinned so a future "auto-reindex"
        change is a deliberate contract break.
        """
        engine = HRM2Engine(n_coarse=2, n_fine=4, n_probe=2, use_gpu=False)
        engine.add_splats(generate_test_splats(100, seed=21))
        engine.index()
        assert engine._is_indexed

        engine.add_splats(generate_test_splats(50, seed=22))

        assert not engine._is_indexed
        with pytest.raises(RuntimeError, match="Index not built"):
            engine.query(np.zeros(640, dtype=np.float32), k=3)

        engine.index()
        results = engine.query(engine.embeddings[0], k=200)
        assert len(engine.splats) == 150
        assert len(results) == 150  # n_probe=2 probes both coarse clusters
