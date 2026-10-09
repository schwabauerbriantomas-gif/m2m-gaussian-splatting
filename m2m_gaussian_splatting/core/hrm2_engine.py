"""
HRM2 Engine - Hierarchical Retrieval Model 2

Implements a two-level hierarchical index for fast similarity search
in large-scale Gaussian splat datasets.

GPU acceleration (CUDA via PyTorch) is used automatically when available.
Falls back to CPU (Numba/NumPy) transparently.

v2.1 (audit 2026-10) changes:
- Fine (level-2) clustering is now built LAZILY on first
  ``query_with_details`` instead of eagerly in ``index()``. The plain
  search path never consumed it, and it accounted for the bulk of CPU
  build time (measured: ~80% at 10k splats). Build is ~3-5x faster.
- Cluster member embeddings are stored in a contiguous CSR-style layout
  at index time, so candidate collection slices views instead of
  fancy-indexing a fresh copy of each cluster on every query.
- Embedding norms precomputed in float64 (squared-distance accumulation
  in float32 could overflow to inf/NaN with large-magnitude embeddings).
- Attribute extraction at index time vectorized (np.stack instead of a
  per-splat Python loop).
- ``query``/``batch_query`` validate query dimensionality; ``n_probe``
  must be >= 1; empty clusters are never probed.
- CPU ``batch_query`` computes coarse distances for the whole batch with
  one GEMM and clusters candidate GEMMs per (query, cluster) group
  instead of a full single-query pipeline per row.
- New: ``save_index`` / ``load_index`` persist the index (embeddings,
  coarse model, CSR layout) so expensive builds survive process restarts.
- New: ``add_splats_incremental`` appends splats to an already-built
  index without re-clustering (approximate — see its docstring);
  ``index()`` remains the canonical full-rebuild path.
- New: ``HRM2Engine.from_config(HRM2Config)`` alternative constructor.
  The loose kwargs remain the primary API; ``from_config`` exists so the
  ``HRM2Config`` dataclass is actually wired to the engine.
"""

import logging
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .splat_types import GaussianSplat
from .encoding import FullEmbeddingBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attempt to import clustering backends — CPU (Numba) and GPU (PyTorch)
# ---------------------------------------------------------------------------
from .clustering import KMeans, assign_clusters as _cpu_assign

_GPU_OK = False
try:
    from ..gpu import HAS_CUDA, HAS_TORCH
    from ..gpu.gpu_kmeans import TorchKMeans
    from ..gpu.gpu_search import GPUSearcher

    if HAS_TORCH:
        _GPU_OK = True
except ImportError:
    pass


_INDEX_FORMAT_VERSION = 1


@dataclass
class SearchResult:
    """Result of a similarity search."""

    splat_id: int
    distance: float
    coarse_cluster: int
    fine_cluster: int


@dataclass
class HRM2Config:
    """Configuration for HRM2 Engine."""

    n_coarse: int = 100
    n_fine: int = 1000
    embedding_dim: int = 640
    n_probe: int = 5
    batch_size: int = 10000
    random_state: int = 42
    use_gpu: bool = True  # auto-detect; set False to force CPU


@dataclass
class HRM2Stats:
    """Statistics for HRM2 Engine."""

    n_splats: int = 0
    n_coarse_clusters: int = 0
    n_fine_clusters: int = 0
    build_time: float = 0.0
    avg_query_time: float = 0.0
    total_queries: int = 0
    device: str = "cpu"


class HRM2Engine:
    """
    Hierarchical Retrieval Model 2 (HRM2) Engine.

    Two-level hierarchical index:
    - Level 1 (Coarse): K-Means clusters for fast pruning
    - Level 2 (Fine): Additional clustering within each coarse cluster,
      built lazily on the first ``query_with_details`` call (the plain
      search path does not use it).

    GPU acceleration is used automatically when PyTorch+CUDA are available.
    The search hot-path routes through :class:`GPUSearcher` for brute-force
    L2 when a GPU is present, or through the hierarchical IVF path on CPU.

    Example:
        >>> engine = HRM2Engine(n_coarse=100, n_fine=1000)
        >>> engine.add_splats(splats)
        >>> engine.index()
        >>> results = engine.query(query_vector, k=10)
    """

    def __init__(
        self,
        n_coarse: int = 100,
        n_fine: int = 1000,
        embedding_dim: int = 640,
        n_probe: int = 5,
        batch_size: int = 10000,
        random_state: int = 42,
        use_gpu: bool = True,
    ):
        if n_probe < 1:
            raise ValueError(f"n_probe must be >= 1, got {n_probe}")

        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.embedding_dim = embedding_dim
        self.n_probe = n_probe
        self.batch_size = batch_size
        self.random_state = random_state

        # GPU detection
        self._gpu_enabled = use_gpu and _GPU_OK and HAS_CUDA
        self._device = "cuda" if self._gpu_enabled else "cpu"

        # Storage
        self.splats: List[GaussianSplat] = []
        self.embeddings: Optional[np.ndarray] = None

        # Index
        self.coarse_model = None  # KMeans or TorchKMeans
        self.coarse_assignments: Optional[np.ndarray] = None
        self.fine_models: Dict[int, Optional[object]] = {}
        self.fine_assignments: Dict[int, np.ndarray] = {}

        # GPU searcher (built at index time when GPU is active)
        self._gpu_searcher: Optional["GPUSearcher"] = None

        # Precomputed lookups
        self._cluster_indices: Dict[int, np.ndarray] = {}
        self._emb_norms_sq: Optional[np.ndarray] = None

        # CSR-style layout over cluster members (built at index time):
        # embeddings/norms sorted by cluster with per-cluster offsets, so
        # query-time candidate collection slices views (no fancy-index copy).
        self._emb_sorted: Optional[np.ndarray] = None
        self._norms_sorted: Optional[np.ndarray] = None
        self._global_order: Optional[np.ndarray] = None
        self._cluster_offsets: Optional[np.ndarray] = None
        self._cluster_nonempty: Optional[np.ndarray] = None

        # Encoder
        self.encoder = FullEmbeddingBuilder()

        # Stats
        self._is_indexed = False
        self._fine_built = False
        self._stats = HRM2Stats(device=self._device)

    @classmethod
    def from_config(cls, config: HRM2Config) -> "HRM2Engine":
        """
        Build an engine from an :class:`HRM2Config` dataclass.

        The loose ``__init__`` kwargs remain the primary API; this
        classmethod makes the config dataclass a real (non-decorative)
        way to construct the engine::

            engine = HRM2Engine.from_config(HRM2Config(n_coarse=50))

        Args:
            config: HRM2Config with the engine parameters.

        Returns:
            A new HRM2Engine instance.
        """
        return cls(
            n_coarse=config.n_coarse,
            n_fine=config.n_fine,
            embedding_dim=config.embedding_dim,
            n_probe=config.n_probe,
            batch_size=config.batch_size,
            random_state=config.random_state,
            use_gpu=config.use_gpu,
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_fine_clusters(self, n_coarse: int) -> None:
        """Build fine sub-clusters within each coarse cluster (CPU only)."""
        for cid in range(n_coarse):
            cluster_idxs = self._cluster_indices[cid]
            if len(cluster_idxs) < 2:
                self.fine_models[cid] = None
                self.fine_assignments[cid] = np.zeros(len(cluster_idxs), dtype=np.int32)
                continue

            cluster_emb = np.ascontiguousarray(self.embeddings[cluster_idxs].astype(np.float32))
            n_fine = max(1, min(self.n_fine, len(cluster_idxs) // 5))

            fm = KMeans(
                n_clusters=n_fine,
                batch_size=min(self.batch_size, len(cluster_idxs)),
                random_state=self.random_state + cid,
            )
            self.fine_models[cid] = fm
            self.fine_assignments[cid] = fm.fit_predict(cluster_emb)

    def _ensure_fine_clusters(self) -> None:
        """Lazily build fine clusters on first use (query_with_details)."""
        if self._is_indexed and not self._fine_built and not self._gpu_enabled:
            self._build_fine_clusters(len(self._cluster_indices))
            self._stats.n_fine_clusters = sum(
                m.n_clusters if m else 0 for m in self.fine_models.values()
            )
            self._fine_built = True

    def _build_csr_layout(self, n_coarse: int) -> None:
        """Group embeddings by cluster into contiguous storage (one copy)."""
        counts = np.array(
            [len(self._cluster_indices.get(cid, ())) for cid in range(n_coarse)],
            dtype=np.int64,
        )
        offsets = np.zeros(n_coarse + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        order = (
            np.concatenate([self._cluster_indices[cid] for cid in range(n_coarse)])
            if len(self.splats) > 0
            else np.zeros(0, dtype=np.int64)
        )

        self._emb_sorted = np.ascontiguousarray(self.embeddings[order])
        self._norms_sorted = np.ascontiguousarray(self._emb_norms_sq[order])
        self._global_order = order
        self._cluster_offsets = offsets
        self._cluster_nonempty = counts > 0

    def add_splats(self, splats: List[GaussianSplat]) -> None:
        """Add splats to the engine."""
        self.splats.extend(splats)
        self._is_indexed = False

    @staticmethod
    def _extract_attributes(splats: List[GaussianSplat]):
        """Stack splat attributes into (positions, colors, opacities, scales, rotations)."""
        positions = np.stack([s.position for s in splats]).astype(np.float32, copy=False)
        colors = np.stack([s.color for s in splats]).astype(np.float32, copy=False)
        opacities = np.fromiter((s.opacity for s in splats), dtype=np.float32)
        scales = np.stack([s.scale for s in splats]).astype(np.float32, copy=False)
        rotations = np.stack([s.rotation for s in splats]).astype(np.float32, copy=False)
        return positions, colors, opacities, scales, rotations

    def index(self) -> float:
        """
        Build the hierarchical index.

        Returns:
            Build time in seconds.
        """
        start = time.time()

        if not self.splats:
            return 0.0

        # Vectorized attribute extraction
        positions, colors, opacities, scales, rotations = self._extract_attributes(self.splats)

        self.embeddings = self.encoder.build(positions, colors, opacities, scales, rotations)
        self.embeddings = np.ascontiguousarray(self.embeddings.astype(np.float32))
        self.embedding_dim = self.embeddings.shape[1]
        # float64 accumulation: float32 norms overflow to inf for
        # large-magnitude embeddings and poison dist_sq with NaN
        self._emb_norms_sq = np.sum(self.embeddings.astype(np.float64) ** 2, axis=1)

        n_samples = len(self.splats)

        # --- GPU brute-force searcher ----------------------------------
        if self._gpu_enabled:
            try:
                self._gpu_searcher = GPUSearcher(
                    self.embeddings,
                    device="cuda",
                    max_batch_size=256,
                )
                logger.info("GPU searcher initialized on %s", self._device)
            except Exception:
                logger.warning("GPU searcher init failed, falling back to CPU IVF")
                self._gpu_enabled = False
                self._device = "cpu"
                self._stats.device = "cpu"

        # --- Coarse clustering ----------------------------------------
        n_coarse = max(1, min(self.n_coarse, n_samples // 10))

        if self._gpu_enabled:
            self.coarse_model = TorchKMeans(
                n_clusters=n_coarse,
                batch_size=self.batch_size,
                random_state=self.random_state,
                device="cuda",
            )
        else:
            self.coarse_model = KMeans(
                n_clusters=n_coarse,
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
        self.coarse_assignments = self.coarse_model.fit_predict(self.embeddings)

        # cluster → global index
        self._cluster_indices = {
            cid: np.where(self.coarse_assignments == cid)[0] for cid in range(n_coarse)
        }
        self._build_csr_layout(n_coarse)

        # --- Fine clustering: LAZY --------------------------------------
        # Only query_with_details consumes the fine level, and eager
        # construction dominated CPU build time. Defer until first use;
        # on GPU the search is brute-force, so it is never needed.
        self.fine_models = {}
        self.fine_assignments = {}
        self._fine_built = False

        self._is_indexed = True

        self._stats.n_splats = n_samples
        self._stats.n_coarse_clusters = n_coarse
        self._stats.n_fine_clusters = 0
        self._stats.build_time = time.time() - start
        self._stats.device = self._device

        return self._stats.build_time

    def add_splats_incremental(self, new_splats: List[GaussianSplat]) -> None:
        """
        Append splats to an already-built index WITHOUT re-clustering.

        Only the new splats are encoded and assigned to the EXISTING
        coarse clusters via ``coarse_model.predict``; the CSR layout,
        norms and cluster memberships are extended in place. On success
        the index remains immediately queryable, including the new
        splats.

        APPROXIMATE BY DESIGN — known limitations:

        - **No re-clustering.** Coarse centroids are not recomputed, so
          repeated incremental additions progressively unbalance the
          clusters (a cluster may grow far beyond its original size,
          degrading n_probe recall). Centroid drift is also ignored:
          new splats are assigned to stale centroids.
        - **Fine level is invalidated.** If fine (level-2) clusters were
          built, they no longer cover the new members, so they are
          discarded and rebuilt lazily on the next
          ``query_with_details``.
        - **Encoder consistency.** The position encoder normalizes per
          batch by default, so a small incremental batch can embed
          slightly differently than the same splats would inside a full
          ``index()`` rebuild. Fit fixed bounds once via
          ``engine.encoder.fit_positions(...)`` before ``index()`` if
          incremental batches must embed consistently.

        ``index()`` (full rebuild) remains the canonical path; use this
        method only when a rebuild is too expensive and approximate
        recall is acceptable.

        Args:
            new_splats: Splats to append (non-empty).

        Raises:
            RuntimeError: If the index has not been built yet, or the
                new embeddings have a different dimensionality than the
                indexed ones.
        """
        if not self._is_indexed or self.embeddings is None or self.coarse_model is None:
            raise RuntimeError("Index not built. Call index() before add_splats_incremental().")
        if not new_splats:
            return

        # Bind once: everything below operates on these locals
        emb_old = self.embeddings
        norms_old = self._emb_norms_sq
        emb_sorted_old = self._emb_sorted
        norms_sorted_old = self._norms_sorted
        order_old = self._global_order
        offsets_old = self._cluster_offsets
        nonempty_old = self._cluster_nonempty
        if (
            norms_old is None
            or emb_sorted_old is None
            or norms_sorted_old is None
            or order_old is None
            or offsets_old is None
            or nonempty_old is None
        ):
            raise RuntimeError("Index structures missing. Call index() first.")

        n_old = len(self.splats)

        # 1. Encode ONLY the new splats
        positions, colors, opacities, scales, rotations = self._extract_attributes(new_splats)
        new_emb = np.ascontiguousarray(
            self.encoder.build(positions, colors, opacities, scales, rotations).astype(np.float32)
        )
        if new_emb.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"new splats embed to dim {new_emb.shape[1]}, "
                f"index expects {self.embedding_dim}"
            )

        # 2. Assign to the EXISTING coarse clusters (no re-fit)
        new_assign = np.asarray(self.coarse_model.predict(new_emb), dtype=np.int64).ravel()

        # 3. Extend the per-cluster global-index map (np.concatenate)
        n_coarse = len(self._cluster_indices)
        new_by_cluster = [np.where(new_assign == cid)[0] for cid in range(n_coarse)]
        new_counts = np.array([len(ix) for ix in new_by_cluster], dtype=np.int64)
        for cid, member_rows in enumerate(new_by_cluster):
            self._cluster_indices[cid] = np.concatenate(
                [self._cluster_indices[cid], member_rows + n_old]
            )

        # 4. Extend the CSR layout. New members are appended directly
        #    after their cluster's old block (piecewise concatenate of
        #    embeddings/norms/order), and every later boundary shifts by
        #    the new members accumulated before it.
        rows_extra = np.concatenate(new_by_cluster)  # rows of new_emb, cluster-major
        new_norms_sq = np.sum(new_emb.astype(np.float64) ** 2, axis=1)

        emb_pieces = []
        norm_pieces = []
        order_pieces = []
        for cid, member_rows in enumerate(new_by_cluster):
            s, e = offsets_old[cid], offsets_old[cid + 1]
            emb_pieces.append(emb_sorted_old[s:e])
            norm_pieces.append(norms_sorted_old[s:e])
            order_pieces.append(order_old[s:e])
            if len(member_rows) > 0:
                emb_pieces.append(new_emb[member_rows])
                norm_pieces.append(new_norms_sq[member_rows])
                order_pieces.append(member_rows + n_old)

        self.embeddings = np.ascontiguousarray(np.vstack([emb_old, new_emb]))
        self._emb_norms_sq = np.concatenate([norms_old, new_norms_sq])
        self._emb_sorted = np.ascontiguousarray(np.vstack(emb_pieces))
        self._norms_sorted = np.concatenate(norm_pieces)
        self._global_order = np.concatenate(order_pieces)
        # CSR offsets: [o0..oK] + [0, cumsum(new_counts)] — each old
        # boundary shifts by the new members added before its cluster
        self._cluster_offsets = offsets_old + np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(new_counts)]
        )
        self._cluster_nonempty = np.logical_or(nonempty_old, new_counts > 0)

        # 5. Bookkeeping: splats, assignments, stats
        self.splats.extend(new_splats)
        self.coarse_assignments = np.concatenate([self.coarse_assignments, new_assign])
        self._stats.n_splats = len(self.splats)

        # 6. Fine level no longer covers the new members: discard it and
        #    let _ensure_fine_clusters rebuild lazily on next use.
        self.fine_models = {}
        self.fine_assignments = {}
        self._fine_built = False

        # 7. GPU brute-force index must see the new rows: rebuild it
        if self._gpu_enabled and self._gpu_searcher is not None:
            try:
                self._gpu_searcher = GPUSearcher(
                    self.embeddings,
                    device="cuda",
                    max_batch_size=256,
                )
            except Exception:
                logger.warning("GPU searcher rebuild failed, falling back to CPU IVF")
                self._gpu_searcher = None
                self._gpu_enabled = False
                self._device = "cpu"
                self._stats.device = "cpu"

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _check_query(self, q: np.ndarray) -> None:
        if q.shape[0] != self.embedding_dim:
            raise ValueError(f"query dimension {q.shape[0]} != embedding_dim {self.embedding_dim}")

    def _probe_clusters(self, coarse_d: np.ndarray) -> np.ndarray:
        """
        Pick the n_probe nearest non-empty coarse clusters.

        Args:
            coarse_d: (..., K) distances to coarse centroids.

        Returns:
            (..., n_probe) cluster ids sorted by distance.
        """
        d = np.array(coarse_d, dtype=np.float64, copy=True)
        d[..., ~self._cluster_nonempty] = np.inf
        n_probe = min(self.n_probe, d.shape[-1])
        closest = np.argpartition(d, n_probe - 1, axis=-1)[..., :n_probe]
        order = np.take_along_axis(d, closest, axis=-1).argsort(axis=-1)
        return np.take_along_axis(closest, order, axis=-1)

    def _collect_candidates(
        self, query_vector: np.ndarray, query_norm_sq: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect candidate splats from nearest coarse clusters.

        Returns:
            ``(global_indices, squared_distances, coarse_ids)``
        """
        coarse_distances = self.coarse_model.transform(query_vector.reshape(1, -1))[0]
        closest = self._probe_clusters(coarse_distances)

        all_indices: List[np.ndarray] = []
        all_dist_sq: List[np.ndarray] = []
        all_coarse: List[np.ndarray] = []

        for cid in closest:
            cid = int(cid)
            s, e = self._cluster_offsets[cid], self._cluster_offsets[cid + 1]
            if s == e:
                continue
            # contiguous views — no per-query fancy-index copy
            cluster_emb = self._emb_sorted[s:e]
            cross = cluster_emb @ query_vector
            dist_sq = query_norm_sq + self._norms_sorted[s:e] - 2.0 * cross

            all_indices.append(self._global_order[s:e])
            all_dist_sq.append(dist_sq)
            all_coarse.append(np.full(e - s, cid, dtype=np.int32))

        if not all_indices:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int32),
            )

        return (
            np.concatenate(all_indices),
            np.concatenate(all_dist_sq),
            np.concatenate(all_coarse),
        )

    def query(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[GaussianSplat, float]]:
        """
        Query for k most similar splats.

        Uses GPU brute-force when available, otherwise the hierarchical
        IVF path on CPU.
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        t0 = time.time()
        q = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32).flatten())
        self._check_query(q)

        if self._gpu_enabled and self._gpu_searcher is not None:
            results = self._query_gpu(q, k)
        else:
            results = self._query_cpu(q, k)

        self._update_query_stats(time.time() - t0)
        return results

    def _query_gpu(self, q: np.ndarray, k: int) -> List[Tuple[GaussianSplat, float]]:
        """Brute-force search on GPU."""
        indices, distances = self._gpu_searcher.batch_search(q.reshape(1, -1), k)
        return [
            (self.splats[int(idx)], float(dist)) for idx, dist in zip(indices[0], distances[0])
        ]

    def _query_cpu(self, q: np.ndarray, k: int) -> List[Tuple[GaussianSplat, float]]:
        """Hierarchical IVF search on CPU."""
        q_norm_sq = float(q @ q)
        global_indices, dist_sq, _ = self._collect_candidates(q, q_norm_sq)

        if len(global_indices) == 0:
            return []

        k_actual = min(k, len(global_indices))
        if k_actual < len(global_indices):
            top_k = np.argpartition(dist_sq, k_actual - 1)[:k_actual]
        else:
            top_k = np.arange(len(global_indices))
        top_k = top_k[np.argsort(dist_sq[top_k])]

        result_indices = global_indices[top_k]
        result_dists = np.sqrt(np.maximum(dist_sq[top_k], 0.0))

        return [(self.splats[idx], float(dist)) for idx, dist in zip(result_indices, result_dists)]

    def query_with_details(self, query_vector: np.ndarray, k: int = 10) -> List[SearchResult]:
        """Query with detailed results including cluster info."""
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        self._ensure_fine_clusters()

        q = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32).flatten())
        self._check_query(q)
        q_norm_sq = float(q @ q)
        global_indices, dist_sq, coarse_ids = self._collect_candidates(q, q_norm_sq)

        if len(global_indices) == 0:
            return []

        k_actual = min(k, len(global_indices))
        if k_actual < len(global_indices):
            top_k = np.argpartition(dist_sq, k_actual - 1)[:k_actual]
        else:
            top_k = np.arange(len(global_indices))
        top_k = top_k[np.argsort(dist_sq[top_k])]

        results = []
        for local_j in top_k:
            gidx = global_indices[local_j]
            cid = int(coarse_ids[local_j])
            fine_assigns = self.fine_assignments.get(cid, np.zeros(0, dtype=np.int32))
            cluster_idxs = self._cluster_indices.get(cid, np.array([]))
            pos_in_cluster = int(np.searchsorted(cluster_idxs, gidx))
            fine_id = (
                int(fine_assigns[pos_in_cluster]) if pos_in_cluster < len(fine_assigns) else 0
            )
            results.append(
                SearchResult(
                    splat_id=self.splats[gidx].id,
                    distance=float(np.sqrt(max(dist_sq[local_j], 0.0))),
                    coarse_cluster=cid,
                    fine_cluster=fine_id,
                )
            )
        return results

    def batch_query(
        self, query_vectors: np.ndarray, k: int = 10
    ) -> List[List[Tuple[GaussianSplat, float]]]:
        """
        Batch query for multiple queries.

        On GPU this is fully parallelized. On CPU the coarse distance
        computation and per-cluster candidate GEMMs are batched (one BLAS
        call per probed cluster instead of one full pipeline per row).
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        qv = np.ascontiguousarray(np.asarray(query_vectors, dtype=np.float32))
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)
        if qv.shape[1] != self.embedding_dim:
            raise ValueError(
                f"query dimension {qv.shape[1]} != embedding_dim {self.embedding_dim}"
            )

        # GPU batch path — single upload, parallel top-k
        if self._gpu_enabled and self._gpu_searcher is not None:
            t0 = time.time()
            indices, distances = self._gpu_searcher.batch_search(qv, k)
            results = []
            for row_idx, row_dist in zip(indices, distances):
                results.append(
                    [(self.splats[int(i)], float(d)) for i, d in zip(row_idx, row_dist)]
                )
            self._update_query_stats((time.time() - t0) / len(qv))
            return results

        return self._batch_query_cpu(qv, k)

    def _batch_query_cpu(self, qv: np.ndarray, k: int) -> List[List[Tuple[GaussianSplat, float]]]:
        """Batched CPU IVF search: GEMM per (probed cluster, query subset)."""
        B = qv.shape[0]
        q_sq = np.einsum("ij,ij->i", qv, qv)[:, None].astype(np.float64)

        coarse_d = self.coarse_model.transform(qv)  # (B, K) — one GEMM
        closest = self._probe_clusters(coarse_d)  # (B, n_probe)

        # group: cluster id → query rows probing it
        cluster_rows: Dict[int, List[int]] = {}
        for b in range(B):
            for cid in closest[b]:
                cluster_rows.setdefault(int(cid), []).append(b)

        cand_idx: List[List[np.ndarray]] = [[] for _ in range(B)]
        cand_dist: List[List[np.ndarray]] = [[] for _ in range(B)]

        for cid, rows in cluster_rows.items():
            s, e = self._cluster_offsets[cid], self._cluster_offsets[cid + 1]
            if s == e:
                continue
            rows_arr = np.asarray(rows)
            blk_emb = self._emb_sorted[s:e]
            blk_norms = self._norms_sorted[s:e]
            cross = qv[rows_arr] @ blk_emb.T  # (b, m) — one GEMM
            d = q_sq[rows_arr] + blk_norms[None, :] - 2.0 * cross
            go = self._global_order[s:e]
            for j, b in enumerate(rows):
                cand_idx[b].append(go)
                cand_dist[b].append(d[j])

        results: List[List[Tuple[GaussianSplat, float]]] = []
        for b in range(B):
            if not cand_idx[b]:
                results.append([])
                continue
            idx = np.concatenate(cand_idx[b])
            dd = np.concatenate(cand_dist[b])
            k_actual = min(k, len(idx))
            if k_actual < len(idx):
                top = np.argpartition(dd, k_actual - 1)[:k_actual]
            else:
                top = np.arange(len(idx))
            top = top[np.argsort(dd[top])]
            res_idx = idx[top]
            res_d = np.sqrt(np.maximum(dd[top], 0.0))
            results.append([(self.splats[int(i)], float(dist)) for i, dist in zip(res_idx, res_d)])

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str) -> None:
        """
        Save the built index to ``path`` (npz).

        Persists embeddings, coarse model centroids, cluster assignments
        and the CSR layout. Splats are NOT persisted (store them
        separately); ``load_index`` requires the same splat list to be
        re-added before queries.
        """
        if not self._is_indexed or self.embeddings is None:
            raise RuntimeError("Index not built. Call index() first.")
        np.savez_compressed(
            path,
            format_version=_INDEX_FORMAT_VERSION,
            embeddings=self.embeddings,
            emb_norms_sq=self._emb_norms_sq,
            coarse_centroids=np.asarray(self.coarse_model.centroids_),
            coarse_assignments=self.coarse_assignments,
            global_order=self._global_order,
            cluster_offsets=self._cluster_offsets,
            cluster_nonempty=self._cluster_nonempty,
            n_probe=self.n_probe,
            embedding_dim=self.embedding_dim,
        )

    def load_index(self, path: str) -> None:
        """
        Load an index previously saved with :meth:`save_index`.

        Restores embeddings and search structures. ``self.splats`` must
        already contain the same splats (in the same order) for query
        results to map back correctly.
        """
        with np.load(path, allow_pickle=False) as z:
            if int(z["format_version"]) != _INDEX_FORMAT_VERSION:
                raise ValueError("Incompatible index file version")
            self.embeddings = np.ascontiguousarray(z["embeddings"])
            self._emb_norms_sq = z["emb_norms_sq"]
            self.coarse_assignments = z["coarse_assignments"]
            self._global_order = z["global_order"]
            self._cluster_offsets = z["cluster_offsets"]
            self._cluster_nonempty = z["cluster_nonempty"]
            self.n_probe = int(z["n_probe"])
            self.embedding_dim = int(z["embedding_dim"])
            centroids = z["coarse_centroids"]

        # Reconstruct a fitted coarse model (only transform/predict used)
        model = KMeans(n_clusters=centroids.shape[0], random_state=self.random_state)
        model.centroids_ = np.ascontiguousarray(centroids)
        self.coarse_model = model

        n_coarse = centroids.shape[0]
        self._cluster_indices = {
            cid: np.where(self.coarse_assignments == cid)[0] for cid in range(n_coarse)
        }
        # sorted-norms layout follows the CSR order
        self._emb_sorted = np.ascontiguousarray(self.embeddings[self._global_order])
        self._norms_sorted = np.ascontiguousarray(self._emb_norms_sq[self._global_order])

        self.splats = list(self.splats)
        if len(self.splats) != self.embeddings.shape[0]:
            raise ValueError(
                f"cannot load index: it was built for {self.embeddings.shape[0]} splats "
                f"but this engine holds {len(self.splats)} — add the same splats "
                f"(same order) via add_splats() before load_index()"
            )
        self._gpu_searcher = None
        self._gpu_enabled = False
        self._device = "cpu"
        self._fine_built = False
        self.fine_models = {}
        self.fine_assignments = {}
        self._is_indexed = True

        self._stats.n_splats = len(self.splats)
        self._stats.n_coarse_clusters = n_coarse
        self._stats.n_fine_clusters = 0
        self._stats.device = "cpu"

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _update_query_stats(self, query_time: float) -> None:
        self._stats.total_queries += 1
        self._stats.avg_query_time = (
            self._stats.avg_query_time * (self._stats.total_queries - 1) + query_time
        ) / self._stats.total_queries

    def get_stats(self) -> HRM2Stats:
        return self._stats

    def clear(self) -> None:
        """Clear all data."""
        self.splats = []
        self.embeddings = None
        self.coarse_model = None
        self.coarse_assignments = None
        self.fine_models = {}
        self.fine_assignments = {}
        self._cluster_indices = {}
        self._emb_norms_sq = None
        self._emb_sorted = None
        self._norms_sorted = None
        self._global_order = None
        self._cluster_offsets = None
        self._cluster_nonempty = None
        self._gpu_searcher = None
        self._is_indexed = False
        self._fine_built = False
        self._stats = HRM2Stats(device=self._device)


# ---------------------------------------------------------------------------
# Test data generation (uses local RNG — does NOT pollute global seed)
# ---------------------------------------------------------------------------


def generate_test_splats(n_splats: int, seed: int = 42) -> List[GaussianSplat]:
    """
    Generate synthetic splats for testing.

    Uses a local :class:`~numpy.random.RandomState` so the global
    NumPy RNG state is never modified.
    """
    rng = np.random.RandomState(seed)
    splats = []
    for i in range(n_splats):
        rot = rng.randn(4).astype(np.float32)
        rot /= np.linalg.norm(rot)
        splats.append(
            GaussianSplat(
                id=i,
                position=rng.randn(3).astype(np.float32) * 10,
                color=rng.rand(3).astype(np.float32),
                opacity=float(rng.rand()),
                scale=np.exp(rng.randn(3).astype(np.float32) * -2),
                rotation=rot,
            )
        )
    return splats
