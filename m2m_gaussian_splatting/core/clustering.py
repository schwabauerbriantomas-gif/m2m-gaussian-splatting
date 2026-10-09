"""
KMeans clustering optimized with Numba JIT + BLAS (NumPy).

This module provides fast K-Means clustering implementation
optimized for Gaussian splat embeddings.

v2.1 (audit 2026-10) changes:
- Cluster assignment now uses the Gram/GEMM trick (||x||² - 2x·c + ||c||²)
  dispatched to BLAS instead of an O(N·K·D) scalar triple loop.
- Numba kernels that use ``prange`` declare ``parallel=True`` (previously
  ``prange`` silently degraded to a serial loop).
- ``kmeans_plusplus_init`` accepts an int seed or a ``numpy.random.Generator``
  and no longer mutates the global RNG state via ``np.random.seed``.
- Degenerate input guard in k-means++ (all distances zero → random pick).
- ``mini_batch_kmeans`` now: samples mini-batches with replacement (no full
  O(N) permutation per iteration), tracks centroid shift with ``tol``
  early stopping, returns ``(centroids, labels, inertia, n_iter)``, and
  reseeds never-updated (dead) clusters from the farthest points.
- ``fit()`` rejects non-finite input (NaN/Inf would silently collapse
  every point into cluster 0).
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


# ==================== LOW-LEVEL KERNELS ====================


@njit(fastmath=True, cache=True, parallel=True)
def _sq_dist_to_centroid(data: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Squared distance from each row of ``data`` to a single centroid."""
    N = data.shape[0]
    out = np.empty(N, dtype=np.float32)
    for i in prange(N):
        s = 0.0
        for d in range(data.shape[1]):
            diff = data[i, d] - centroid[d]
            s += diff * diff
        out[i] = s
    return out


@njit(fastmath=True, cache=True)
def _sum_by_label(
    data: np.ndarray, labels: np.ndarray, n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cluster coordinate sums and point counts."""
    K = n_clusters
    D = data.shape[1]
    sums = np.zeros((K, D), dtype=np.float32)
    counts = np.zeros(K, dtype=np.int64)
    for i in range(data.shape[0]):
        k = labels[i]
        counts[k] += 1
        for d in range(D):
            sums[k, d] += data[i, d]
    return sums, counts


def _assign_gemm(data: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each point to the nearest centroid via BLAS GEMM.

    Returns:
        (labels (N,) int32, dist_sq (N, K) float32)
    """
    # einsum avoids materializing (N, D) squares
    data_sq = np.einsum("ij,ij->i", data, data)[:, None]
    cent_sq = np.einsum("ij,ij->i", centroids, centroids)[None, :]
    cross = data @ centroids.T
    dist_sq = data_sq + cent_sq - 2.0 * cross
    np.maximum(dist_sq, 0.0, out=dist_sq)
    labels = np.argmin(dist_sq, axis=1).astype(np.int32)
    return labels, dist_sq


# ==================== K-MEANS++ INITIALIZATION ====================


def kmeans_plusplus_init(
    data: np.ndarray,
    n_clusters: int,
    random_state: Union[int, np.random.Generator] = 42,
) -> np.ndarray:
    """
    KMeans++ initialization for better centroid selection.

    Uses a running minimum-distance array so each new centroid only
    needs O(N·D) work instead of recomputing all distances from scratch.
    Total complexity: O(N·K·D), with the per-centroid distance pass
    JIT-compiled and parallelized.

    Args:
        data: (N, D) array of data points
        n_clusters: Number of clusters
        random_state: Int seed or ``numpy.random.Generator``. The global
            NumPy RNG state is never modified.

    Returns:
        (n_clusters, D) array of initial centroids
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    N, D = data.shape
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    centroids = np.zeros((n_clusters, D), dtype=np.float32)

    # Choose first centroid randomly
    idx = int(rng.integers(0, N))
    centroids[0] = data[idx]

    # Running minimum squared distance to nearest selected centroid
    min_dist_sq = _sq_dist_to_centroid(data, centroids[0])
    np.maximum(min_dist_sq, 0.0, out=min_dist_sq)

    # Choose remaining centroids
    for k in range(1, n_clusters):
        total = float(min_dist_sq.sum())
        if total <= 0.0:
            # Degenerate: all points identical (or duplicated data).
            # Fall back to a random pick instead of repeating the same point.
            idx = int(rng.integers(0, N))
        else:
            # Sample ∝ D² via inverse-CDF on the (unnormalized) distances
            r = rng.random() * total
            idx = int(np.searchsorted(np.cumsum(min_dist_sq), r))
            idx = min(idx, N - 1)
        centroids[k] = data[idx]

        new_dist = _sq_dist_to_centroid(data, centroids[k])
        np.minimum(min_dist_sq, new_dist, out=min_dist_sq)

    return centroids


# ==================== CLUSTER ASSIGNMENT ====================


def assign_clusters(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each point to nearest centroid (BLAS GEMM, Gram trick).

    Args:
        data: (N, D) array
        centroids: (K, D) array

    Returns:
        (N,) array of cluster labels
    """
    labels, _ = _assign_gemm(data, centroids)
    return labels


# ==================== CENTROID UPDATE ====================


def update_centroids(
    data: np.ndarray, labels: np.ndarray, n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update centroids based on cluster assignments.

    Args:
        data: (N, D) array
        labels: (N,) cluster labels
        n_clusters: Number of clusters

    Returns:
        Tuple of (new_centroids, cluster_sizes)
    """
    sums, counts = _sum_by_label(data, labels, n_clusters)
    centroids = np.zeros_like(sums)
    nz = counts > 0
    centroids[nz] = sums[nz] / counts[nz, None].astype(np.float32)
    return centroids, counts


# ==================== MINI-BATCH K-MEANS ====================


def mini_batch_kmeans(
    data: np.ndarray,
    initial_centroids: np.ndarray,
    batch_size: int = 1000,
    max_iter: int = 100,
    tol: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Mini-batch KMeans for large datasets.

    More efficient than standard KMeans for large N.

    Args:
        data: (N, D) array
        initial_centroids: (K, D) initial centroids
        batch_size: Mini-batch size
        max_iter: Maximum iterations
        tol: Early-stop when the total squared centroid shift < tol
        rng: Optional ``numpy.random.Generator`` (a local default is
            created otherwise; the global RNG state is never touched)

    Returns:
        Tuple of (centroids, labels, inertia, n_iter)
    """
    if rng is None:
        rng = np.random.default_rng()

    centroids = np.ascontiguousarray(initial_centroids, dtype=np.float32).copy()
    N, D = data.shape
    K = centroids.shape[0]
    batch_size = max(1, min(batch_size, N))

    counts = np.ones(K, dtype=np.float64)  # per-centroid update count (sklearn-style)
    ever_updated = np.zeros(K, dtype=bool)
    n_iter = 0

    for iteration in range(max_iter):
        n_iter = iteration + 1

        # Sample mini-batch with replacement (sklearn-style) — O(batch) instead
        # of a full O(N) permutation per iteration.
        idx = rng.integers(0, N, size=batch_size)
        batch = np.ascontiguousarray(data[idx])

        batch_labels, _ = _assign_gemm(batch, centroids)
        sums, bcounts = _sum_by_label(batch, batch_labels, K)

        updated = bcounts > 0
        ever_updated |= updated
        if not updated.any():
            continue

        eta = 1.0 / (counts[updated] + bcounts[updated])
        new_centroids = centroids.copy()
        new_centroids[updated] = (1.0 - eta[:, None]) * centroids[updated] + eta[:, None] * (
            sums[updated] / bcounts[updated, None].astype(np.float32)
        )
        counts[updated] += bcounts[updated]

        shift = float(((new_centroids - centroids) ** 2).sum())
        centroids = new_centroids

        if shift < tol:
            break

    # Final assignment + inertia
    labels, dist_sq = _assign_gemm(data, centroids)
    inertia = float(dist_sq[np.arange(N), labels].sum())

    # Reseed never-updated (dead) clusters from the points farthest from
    # their assigned centroid, then reassign once. Without this, the
    # 1/counts learning rule freezes such centroids forever and the IVF
    # index wastes probes on empty buckets.
    dead = ~ever_updated
    n_dead = int(dead.sum())
    if n_dead > 0 and N > K:
        point_dist_sq = dist_sq[np.arange(N), labels]
        far_idx = np.argpartition(point_dist_sq, N - n_dead)[N - n_dead :]
        centroids[dead] = data[far_idx]
        labels, dist_sq = _assign_gemm(data, centroids)
        inertia = float(dist_sq[np.arange(N), labels].sum())

    return centroids, labels, inertia, n_iter


# ==================== FULL K-MEANS ====================


def kmeans_full(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Union[int, np.random.Generator] = 42,
    initial_centroids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Full (Lloyd's) K-Means.

    Args:
        data: (N, D) array
        n_clusters: Number of clusters
        max_iter: Maximum iterations
        tol: Convergence tolerance on the total squared centroid shift
        random_state: Int seed or Generator (global RNG never touched)
        initial_centroids: Optional (K, D) starting centroids; computed with
            k-means++ when omitted (avoids double init work when the caller
            already has them).

    Returns:
        Tuple of (centroids, labels, inertia)
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    N, D = data.shape
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    if initial_centroids is not None:
        centroids = np.ascontiguousarray(initial_centroids, dtype=np.float32).copy()
    else:
        centroids = kmeans_plusplus_init(data, n_clusters, rng)

    n_iter = 0
    for iteration in range(max_iter):
        n_iter = iteration + 1
        labels, _ = _assign_gemm(data, centroids)
        sums, counts = _sum_by_label(data, labels, n_clusters)

        new_centroids = centroids.copy()
        nz = counts > 0
        new_centroids[nz] = sums[nz] / counts[nz, None].astype(np.float32)

        # Handle empty clusters: reseed with the points farthest from their
        # assigned centroid (standard practice — a random point usually lands
        # in an already-dense region).
        n_empty = int((~nz).sum())
        if n_empty > 0 and N > n_clusters:
            _, dist_sq = _assign_gemm(data, new_centroids)
            point_dist_sq = dist_sq[np.arange(N), labels]
            far_idx = np.argpartition(point_dist_sq, N - n_empty)[N - n_empty :]
            new_centroids[~nz] = data[far_idx]

        shift = float(((new_centroids - centroids) ** 2).sum())
        centroids = new_centroids

        if shift < tol:
            break

    labels, dist_sq = _assign_gemm(data, centroids)
    inertia = float(dist_sq[np.arange(N), labels].sum())
    return centroids, labels, inertia


# ==================== PYTHON WRAPPER ====================


@dataclass
class KMeansResult:
    """Result of K-Means clustering."""

    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int


class KMeans:
    """
    K-Means clustering with Mini-batch optimization.

    Uses Mini-batch K-Means for efficiency on large datasets.

    Example:
        >>> kmeans = KMeans(n_clusters=100, batch_size=1000)
        >>> result = kmeans.fit(data)
        >>> labels = kmeans.predict(new_data)
    """

    def __init__(
        self,
        n_clusters: int = 100,
        batch_size: int = 1000,
        max_iter: int = 100,
        random_state: int = 42,
        use_mini_batch: bool = True,
        tol: float = 1e-4,
    ):
        """
        Initialize K-Means.

        Args:
            n_clusters: Number of clusters
            batch_size: Mini-batch size (for mini-batch mode)
            max_iter: Maximum iterations
            random_state: Random seed
            use_mini_batch: Use mini-batch K-Means
            tol: Convergence tolerance on total squared centroid shift
        """
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_mini_batch = use_mini_batch
        self.tol = tol

        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: int = 0

    def fit(self, data: np.ndarray) -> "KMeans":
        """
        Fit K-Means to data.

        Args:
            data: (N, D) array of data points

        Returns:
            self

        Raises:
            ValueError: If ``data`` is empty or contains NaN/Inf.
                Non-finite input silently collapses all points into
                cluster 0, so it is rejected up front.
        """
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[0] == 0:
            raise ValueError("Cannot fit KMeans on empty data")

        if not np.isfinite(data).all():
            raise ValueError(
                "KMeans.fit(): input contains NaN/Inf — " "sanitize embeddings before clustering"
            )

        rng = np.random.default_rng(self.random_state)
        n_clusters = min(self.n_clusters, data.shape[0])

        if self.use_mini_batch:
            self.centroids_, self.labels_, self.inertia_, self.n_iter_ = mini_batch_kmeans(
                data,
                kmeans_plusplus_init(data, n_clusters, rng),
                self.batch_size,
                self.max_iter,
                self.tol,
                rng,
            )
        else:
            self.centroids_, self.labels_, self.inertia_ = kmeans_full(
                data, n_clusters, self.max_iter, self.tol, rng
            )
            self.n_iter_ = -1  # Lloyd's loop does not expose iteration count

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            data: (N, D) array

        Returns:
            (N,) array of cluster labels
        """
        if self.centroids_ is None:
            raise RuntimeError("Must call fit() before predict()")

        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        return assign_clusters(data, self.centroids_)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and return labels.

        Args:
            data: (N, D) array

        Returns:
            (N,) array of cluster labels
        """
        self.fit(data)
        return self.labels_

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to cluster-distance space.

        Args:
            data: (N, D) array

        Returns:
            (N, n_clusters) array of distances
        """
        if self.centroids_ is None:
            raise RuntimeError("Must call fit() before transform()")

        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # ||data - centroid||² = ||data||² + ||centroid||² - 2·data·centroidᵀ
        _, dist_sq = _assign_gemm(data, self.centroids_)
        return np.sqrt(dist_sq)


# Alias for backward compatibility
KMeansJIT = KMeans
