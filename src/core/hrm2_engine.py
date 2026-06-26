"""
HRM2 Engine - Hierarchical Retrieval Model 2

Implements a two-level hierarchical index for fast similarity search
in large-scale Gaussian splat datasets.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import time

from .splat_types import GaussianSplat, SplatEmbedding, SplatCluster
from .encoding import FullEmbeddingBuilder
from .clustering import KMeans, assign_clusters


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
    n_coarse: int = 100          # Number of coarse clusters
    n_fine: int = 1000           # Number of fine clusters per coarse
    embedding_dim: int = 640     # Embedding dimension (informational; actual dim from FullEmbeddingBuilder)
    n_probe: int = 5             # Clusters to probe during search
    batch_size: int = 10000      # Batch size for K-Means
    random_state: int = 42


@dataclass
class HRM2Stats:
    """Statistics for HRM2 Engine."""
    n_splats: int = 0
    n_coarse_clusters: int = 0
    n_fine_clusters: int = 0
    build_time: float = 0.0
    avg_query_time: float = 0.0
    total_queries: int = 0


class HRM2Engine:
    """
    Hierarchical Retrieval Model 2 (HRM2) Engine.

    Implements a two-level hierarchical index:
    - Level 1 (Coarse): K-Means clusters for fast pruning
    - Level 2 (Fine): Additional clustering within each coarse cluster

    This provides significant speedup over brute-force search while
    maintaining high recall.

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
        random_state: int = 42
    ):
        """
        Initialize HRM2 Engine.

        Args:
            n_coarse: Number of coarse clusters
            n_fine: Fine clusters per coarse cluster
            embedding_dim: Dimension of embedding vectors (informational)
            n_probe: Coarse clusters to search
            batch_size: Batch size for K-Means
            random_state: Random seed
        """
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.embedding_dim = embedding_dim
        self.n_probe = n_probe
        self.batch_size = batch_size
        self.random_state = random_state

        # Storage
        self.splats: List[GaussianSplat] = []
        self.embeddings: Optional[np.ndarray] = None

        # Index
        self.coarse_model: Optional[KMeans] = None
        self.coarse_assignments: Optional[np.ndarray] = None
        self.fine_models: Dict[int, KMeans] = {}
        self.fine_assignments: Dict[int, np.ndarray] = {}

        # Precomputed cluster → global index mapping (built at index time)
        self._cluster_indices: Dict[int, np.ndarray] = {}
        # Precomputed embedding norms (for query distance reuse)
        self._emb_norms_sq: Optional[np.ndarray] = None

        # Encoder
        self.encoder = FullEmbeddingBuilder()

        # Statistics
        self._is_indexed = False
        self._stats = HRM2Stats()

    def add_splats(self, splats: List[GaussianSplat]) -> None:
        """
        Add splats to the engine.

        Args:
            splats: List of GaussianSplat objects
        """
        self.splats.extend(splats)
        self._is_indexed = False

    def index(self) -> float:
        """
        Build the hierarchical index.

        Returns:
            Time taken to build index
        """
        start_time = time.time()

        if len(self.splats) == 0:
            return 0.0

        # Build embeddings
        positions = np.array([s.position for s in self.splats])
        colors = np.array([s.color for s in self.splats])
        opacities = np.array([s.opacity for s in self.splats])
        scales = np.array([s.scale for s in self.splats])
        rotations = np.array([s.rotation for s in self.splats])

        self.embeddings = self.encoder.build(
            positions, colors, opacities, scales, rotations
        )

        # Ensure correct dtype
        self.embeddings = np.ascontiguousarray(self.embeddings.astype(np.float32))

        # Update embedding_dim to actual value
        self.embedding_dim = self.embeddings.shape[1]

        # Precompute embedding squared norms (reused in queries)
        self._emb_norms_sq = np.sum(self.embeddings ** 2, axis=1)

        n_samples = len(self.splats)

        # Level 1: Coarse clustering
        n_coarse_effective = min(self.n_coarse, n_samples // 10)
        n_coarse_effective = max(1, n_coarse_effective)

        self.coarse_model = KMeans(
            n_clusters=n_coarse_effective,
            batch_size=self.batch_size,
            random_state=self.random_state
        )
        self.coarse_assignments = self.coarse_model.fit_predict(self.embeddings)

        # Precompute cluster → global index mapping for fast lookup at query time
        self._cluster_indices = {}
        for coarse_id in range(n_coarse_effective):
            mask = self.coarse_assignments == coarse_id
            self._cluster_indices[coarse_id] = np.where(mask)[0]

        # Level 2: Fine clustering within each coarse cluster
        self.fine_models = {}
        self.fine_assignments = {}

        for coarse_id in range(n_coarse_effective):
            cluster_indices = self._cluster_indices[coarse_id]

            if len(cluster_indices) < 2:
                self.fine_models[coarse_id] = None
                self.fine_assignments[coarse_id] = np.zeros(len(cluster_indices), dtype=np.int32)
                continue

            cluster_embeddings = np.ascontiguousarray(self.embeddings[cluster_indices].astype(np.float32))

            # Dynamic n_fine based on cluster size
            n_fine_effective = min(self.n_fine, len(cluster_indices) // 5)
            n_fine_effective = max(1, n_fine_effective)

            fine_model = KMeans(
                n_clusters=n_fine_effective,
                batch_size=min(self.batch_size, len(cluster_indices)),
                random_state=self.random_state + coarse_id
            )

            self.fine_models[coarse_id] = fine_model
            self.fine_assignments[coarse_id] = fine_model.fit_predict(cluster_embeddings)

        self._is_indexed = True

        # Update stats
        self._stats.n_splats = n_samples
        self._stats.n_coarse_clusters = n_coarse_effective
        self._stats.n_fine_clusters = sum(
            m.n_clusters if m else 0 for m in self.fine_models.values()
        )
        self._stats.build_time = time.time() - start_time

        return self._stats.build_time

    def _collect_candidates(
        self,
        query_vector: np.ndarray,
        query_norm_sq: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect candidate splats from nearest coarse clusters.

        Uses squared L2 distance (||q-m||² = ||q||² + ||m||² - 2·q·m)
        to avoid sqrt in the ranking stage.

        Returns:
            Tuple of (global_indices, squared_distances, coarse_ids)
        """
        # Find nearest coarse clusters via argpartition (O(K) vs O(K log K))
        coarse_distances = self.coarse_model.transform(query_vector.reshape(1, -1))[0]
        n_probe = min(self.n_probe, len(coarse_distances))
        closest_coarse = np.argpartition(coarse_distances, n_probe - 1)[:n_probe]
        closest_coarse = closest_coarse[np.argsort(coarse_distances[closest_coarse])]

        all_indices = []
        all_dist_sq = []
        all_coarse = []

        for coarse_id in closest_coarse:
            cluster_indices = self._cluster_indices.get(coarse_id)
            if cluster_indices is None or len(cluster_indices) == 0:
                continue

            cluster_emb = self.embeddings[cluster_indices]
            # Squared L2 via Gram trick: ||q-m||² = ||q||² + ||m||² - 2·q·m
            cross = cluster_emb @ query_vector            # (M,)
            cluster_norm_sq = self._emb_norms_sq[cluster_indices]
            dist_sq = query_norm_sq + cluster_norm_sq - 2.0 * cross

            all_indices.append(cluster_indices)
            all_dist_sq.append(dist_sq)
            all_coarse.append(np.full(len(cluster_indices), coarse_id, dtype=np.int32))

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

    def query(
        self,
        query_vector: np.ndarray,
        k: int = 10
    ) -> List[Tuple[GaussianSplat, float]]:
        """
        Query for k most similar splats.

        Args:
            query_vector: Query embedding vector
            k: Number of results

        Returns:
            List of (Splat, distance) tuples
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_start = time.time()

        query_vector = np.ascontiguousarray(
            np.asarray(query_vector, dtype=np.float32).flatten()
        )
        query_norm_sq = float(query_vector @ query_vector)

        global_indices, dist_sq, _ = self._collect_candidates(query_vector, query_norm_sq)

        if len(global_indices) == 0:
            self._update_query_stats(time.time() - query_start)
            return []

        # Top-k via argpartition (O(N) vs O(N log N) for sort)
        k_actual = min(k, len(global_indices))
        if k_actual < len(global_indices):
            top_k = np.argpartition(dist_sq, k_actual - 1)[:k_actual]
        else:
            top_k = np.arange(len(global_indices))

        # Sort just the k results by distance
        top_k = top_k[np.argsort(dist_sq[top_k])]

        # Convert squared distance to actual distance (sqrt only k values)
        result_indices = global_indices[top_k]
        result_dists = np.sqrt(np.maximum(dist_sq[top_k], 0.0))

        self._update_query_stats(time.time() - query_start)

        return [
            (self.splats[idx], float(dist))
            for idx, dist in zip(result_indices, result_dists)
        ]

    def query_with_details(
        self,
        query_vector: np.ndarray,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Query with detailed results including cluster info.

        Args:
            query_vector: Query embedding vector
            k: Number of results

        Returns:
            List of SearchResult objects
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_vector = np.ascontiguousarray(
            np.asarray(query_vector, dtype=np.float32).flatten()
        )
        query_norm_sq = float(query_vector @ query_vector)

        global_indices, dist_sq, coarse_ids = self._collect_candidates(
            query_vector, query_norm_sq
        )

        if len(global_indices) == 0:
            return []

        # Top-k via argpartition
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
            # Map global index to position within cluster
            cluster_idxs = self._cluster_indices.get(cid, np.array([]))
            pos_in_cluster = np.searchsorted(cluster_idxs, gidx)
            fine_id = int(fine_assigns[pos_in_cluster]) if pos_in_cluster < len(fine_assigns) else 0

            results.append(SearchResult(
                splat_id=self.splats[gidx].id,
                distance=float(np.sqrt(max(dist_sq[local_j], 0.0))),
                coarse_cluster=cid,
                fine_cluster=fine_id
            ))

        return results

    def batch_query(
        self,
        query_vectors: np.ndarray,
        k: int = 10
    ) -> List[List[Tuple[GaussianSplat, float]]]:
        """
        Batch query for multiple queries.

        Args:
            query_vectors: (B, D) array of query vectors
            k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_vectors = np.ascontiguousarray(
            np.asarray(query_vectors, dtype=np.float32)
        )
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        results = []
        for i in range(query_vectors.shape[0]):
            results.append(self.query(query_vectors[i], k=k))

        return results

    def _update_query_stats(self, query_time: float) -> None:
        """Update running query statistics."""
        self._stats.total_queries += 1
        self._stats.avg_query_time = (
            (self._stats.avg_query_time * (self._stats.total_queries - 1) + query_time)
            / self._stats.total_queries
        )

    def get_stats(self) -> HRM2Stats:
        """Get engine statistics."""
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
        self._is_indexed = False
        self._stats = HRM2Stats()


def generate_test_splats(n_splats: int, seed: int = 42) -> List[GaussianSplat]:
    """
    Generate synthetic splats for testing.

    Args:
        n_splats: Number of splats
        seed: Random seed

    Returns:
        List of GaussianSplat objects
    """
    np.random.seed(seed)

    splats = []
    for i in range(n_splats):
        splat = GaussianSplat(
            id=i,
            position=np.random.randn(3).astype(np.float32) * 10,
            color=np.random.rand(3).astype(np.float32),
            opacity=np.random.rand(),
            scale=np.exp(np.random.randn(3).astype(np.float32) * -2),
            rotation=np.random.randn(4).astype(np.float32)
        )
        # Normalize quaternion
        splat.rotation /= np.linalg.norm(splat.rotation)
        splats.append(splat)

    return splats
