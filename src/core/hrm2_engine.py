"""
HRM2 Engine - Hierarchical Retrieval Model 2

Implements a two-level hierarchical index for fast similarity search
in large-scale Gaussian splat datasets.

GPU acceleration (CUDA via PyTorch) is used automatically when available.
Falls back to CPU (Numba/NumPy) transparently.
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
    - Level 2 (Fine): Additional clustering within each coarse cluster

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

        # Encoder
        self.encoder = FullEmbeddingBuilder()

        # Stats
        self._is_indexed = False
        self._stats = HRM2Stats(device=self._device)

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
        """Lazily build fine clusters if not yet built."""
        if not self.fine_models and self._is_indexed:
            self._build_fine_clusters(len(self._cluster_indices))

    def add_splats(self, splats: List[GaussianSplat]) -> None:
        """Add splats to the engine."""
        self.splats.extend(splats)
        self._is_indexed = False

    def index(self) -> float:
        """
        Build the hierarchical index.

        Returns:
            Build time in seconds.
        """
        start = time.time()

        if not self.splats:
            return 0.0

        # Vectorized attribute extraction (avoid list-comprehension overhead)
        N = len(self.splats)
        positions = np.empty((N, 3), dtype=np.float32)
        colors = np.empty((N, 3), dtype=np.float32)
        opacities = np.empty(N, dtype=np.float32)
        scales = np.empty((N, 3), dtype=np.float32)
        rotations = np.empty((N, 4), dtype=np.float32)

        for i, s in enumerate(self.splats):
            positions[i] = s.position
            colors[i] = s.color
            opacities[i] = s.opacity
            scales[i] = s.scale
            rotations[i] = s.rotation

        self.embeddings = self.encoder.build(positions, colors, opacities, scales, rotations)
        self.embeddings = np.ascontiguousarray(self.embeddings.astype(np.float32))
        self.embedding_dim = self.embeddings.shape[1]
        self._emb_norms_sq = np.sum(self.embeddings**2, axis=1)

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
        self._cluster_indices = {}
        for cid in range(n_coarse):
            self._cluster_indices[cid] = np.where(self.coarse_assignments == cid)[0]

        # --- Fine clustering (lazy — only built when query_with_details is used)
        # On GPU, skip fine clustering entirely since search is brute-force.
        # On CPU, defer to keep index() fast.
        self.fine_models = {}
        self.fine_assignments = {}
        if not self._gpu_enabled:
            self._build_fine_clusters(n_coarse)

        self._is_indexed = True

        self._stats.n_splats = n_samples
        self._stats.n_coarse_clusters = n_coarse
        self._stats.n_fine_clusters = (
            sum(m.n_clusters if m else 0 for m in self.fine_models.values())
            if self.fine_models
            else 0
        )
        self._stats.build_time = time.time() - start
        self._stats.device = self._device

        return self._stats.build_time

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _collect_candidates(
        self, query_vector: np.ndarray, query_norm_sq: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect candidate splats from nearest coarse clusters.

        Returns:
            ``(global_indices, squared_distances, coarse_ids)``
        """
        coarse_distances = self.coarse_model.transform(query_vector.reshape(1, -1))[0]
        n_probe = min(self.n_probe, len(coarse_distances))
        closest = np.argpartition(coarse_distances, n_probe - 1)[:n_probe]
        closest = closest[np.argsort(coarse_distances[closest])]

        all_indices: List[np.ndarray] = []
        all_dist_sq: List[np.ndarray] = []
        all_coarse: List[np.ndarray] = []

        for cid in closest:
            cluster_idxs = self._cluster_indices.get(cid)
            if cluster_idxs is None or len(cluster_idxs) == 0:
                continue
            cluster_emb = self.embeddings[cluster_idxs]
            cross = cluster_emb @ query_vector
            cluster_norm_sq = self._emb_norms_sq[cluster_idxs]
            dist_sq = query_norm_sq + cluster_norm_sq - 2.0 * cross

            all_indices.append(cluster_idxs)
            all_dist_sq.append(dist_sq)
            all_coarse.append(np.full(len(cluster_idxs), cid, dtype=np.int32))

        if not all_indices:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
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

        On GPU this is fully parallelized. On CPU it iterates.
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        qv = np.ascontiguousarray(np.asarray(query_vectors, dtype=np.float32))
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)

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

        # CPU path
        return [self.query(qv[i], k=k) for i in range(qv.shape[0])]

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
        self._gpu_searcher = None
        self._is_indexed = False
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
