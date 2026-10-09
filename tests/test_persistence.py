"""
Persistence tests for HRM2Engine.save_index / load_index.

These tests pin the CURRENT behavior as a contract:
- load_index on a missing file fails with FileNotFoundError/OSError (clean).
- save/load round-trip restores search results and the saved n_probe.
- load_index does NOT persist splats: self.splats remains the caller's
  responsibility. Querying a loaded engine that has no splats re-added
  crashes with an opaque IndexError (documented below as a known gap).
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m2m_gaussian_splatting.core.hrm2_engine import HRM2Engine, generate_test_splats


def _build_engine(n_splats=300, seed=5, n_probe=3):
    engine = HRM2Engine(n_coarse=5, n_fine=10, n_probe=n_probe, use_gpu=False)
    engine.add_splats(generate_test_splats(n_splats, seed=seed))
    engine.index()
    return engine


class TestPersistence:
    """save_index / load_index contracts."""

    def test_load_index_missing_file_fails_clean(self):
        """load_index on a deleted file raises FileNotFoundError, not an ugly crash.

        CURRENT behavior: load_index does not pre-check os.path.exists;
        np.load itself raises ``FileNotFoundError`` (subclass of OSError)
        with "[Errno 2] No such file or directory: '<path>'". That is a
        clean, catchable OS error — no AssertionError, no corrupt state
        (the engine is left untouched and still un-indexed).

        IMPROVEMENT NOTE (for the engine owner, not fixed here): an
        explicit existence check raising e.g.
        ``FileNotFoundError(f"index file not found: {path}")`` would give a
        domain-specific message instead of the raw errno one.
        """
        engine = _build_engine()

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "idx.npz")
            engine.save_index(path)
            assert os.path.exists(path)
            os.remove(path)

            fresh = HRM2Engine(n_coarse=5, n_fine=10, use_gpu=False)
            with pytest.raises((FileNotFoundError, OSError)) as excinfo:
                fresh.load_index(path)

            # Message mentions the file (errno text carries the path).
            assert str(path) in str(excinfo.value) or "No such file" in str(excinfo.value)
            # Engine state untouched by the failed load.
            assert not fresh._is_indexed

    def test_roundtrip_custom_n_probe_then_query(self):
        """Round-trip with a custom n_probe: config is restored and queries work.

        CURRENT behavior: ``n_probe`` is serialized in the npz and OVERRIDES
        whatever the loading engine was constructed with (engine2 is built
        with n_probe=5 but ends up with the saved n_probe=2). Queries after
        load return the same splat ids and distances as the saving engine.
        """
        engine1 = _build_engine(n_splats=300, seed=5, n_probe=2)
        assert engine1.n_probe == 2
        query = engine1.embeddings[11]
        expected = engine1.query(query, k=5)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "idx.npz")
            engine1.save_index(path)

            engine2 = HRM2Engine(n_coarse=5, n_fine=10, n_probe=5, use_gpu=False)
            engine2.add_splats(list(engine1.splats))
            engine2.load_index(path)

            assert engine2._is_indexed
            # Saved n_probe wins over the constructor argument.
            assert engine2.n_probe == 2

            got = engine2.query(query, k=5)

        assert len(got) == len(expected) == 5
        for (s1, d1), (s2, d2) in zip(expected, got):
            assert s1.id == s2.id
            np.testing.assert_allclose(d1, d2, rtol=1e-5)

    def test_load_index_without_splats_query_current_behavior(self):
        """Querying a loaded engine with NO splats re-added: clear ValueError.

        CONTRACT (v2.1 close-out): ``load_index`` validates that the engine
        holds exactly as many splats as the index was built for, and raises
        ``ValueError`` with an actionable message otherwise. This replaced
        the previous behavior (load succeeded, later query raised a bare
        ``IndexError: list index out of range`` from ``self.splats[idx]``).
        """
        engine1 = _build_engine(n_splats=200, seed=9)
        query = engine1.embeddings[0]

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "idx.npz")
            engine1.save_index(path)

            engine2 = HRM2Engine(n_coarse=5, n_fine=10, use_gpu=False)
            # NOTE: no add_splats() call — splats were never re-supplied.
            with pytest.raises(ValueError, match="add_splats"):
                engine2.load_index(path)

    def test_save_does_not_persist_splats_caller_owns_them(self):
        """Splats are NOT persisted: the caller re-adds them on load.

        CURRENT behavior: the npz contains only index structures
        (embeddings, centroids, assignments, CSR layout, config) — no splat
        payloads. After load, ``engine.splats`` is exactly the list the
        caller added (same objects, same order), and a save→load round-trip
        across two engines with the same splats yields identical queries.
        """
        splats = generate_test_splats(250, seed=13)
        engine1 = HRM2Engine(n_coarse=5, n_fine=10, n_probe=4, use_gpu=False)
        engine1.add_splats(splats)
        engine1.index()
        query = engine1.embeddings[42]
        expected = engine1.query(query, k=7)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "idx.npz")
            engine1.save_index(path)

            # No splat payload keys in the archive.
            with np.load(path, allow_pickle=False) as z:
                assert "splats" not in z.files
                assert not any("splat" in k for k in z.files)

            engine2 = HRM2Engine(n_coarse=50, n_fine=10, n_probe=1, use_gpu=False)
            engine2.add_splats(splats)  # caller's responsibility
            engine2.load_index(path)

            # The caller's list is used as-is: same objects, same order.
            assert engine2.splats is not None
            assert len(engine2.splats) == len(splats)
            assert all(a is b for a, b in zip(engine2.splats, splats))

            got = engine2.query(query, k=7)

        assert [s.id for s, _ in expected] == [s.id for s, _ in got]
        for (_, d1), (_, d2) in zip(expected, got):
            np.testing.assert_allclose(d1, d2, rtol=1e-5)
