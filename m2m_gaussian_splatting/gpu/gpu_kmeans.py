"""
GPU-accelerated K-Means clustering using PyTorch CUDA tensors.

Drop-in replacement for the Numba ``KMeans`` when a GPU is available.
Falls back to CPU automatically.

v2.1 (audit 2026-10) changes:
- The mini-batch centroid update no longer loops over clusters in Python
  (the old loop launched ~6 CUDA kernels and forced a device sync per
  cluster per iteration ≈ tens of thousands of syncs per fit). Now one
  ``index_add_`` + ``bincount`` per iteration, ~4 kernels, 1 sync.
- ``tol`` is actually honored (was accepted, stored, never used): the
  loop early-stops when the squared centroid shift drops below it, and
  ``n_iter_`` reports the iterations run.
- Mini-batch sampling uses ``torch.randint`` (with replacement) instead
  of a full O(N) ``randperm`` per iteration.
- Dead clusters (never updated, which the 1/counts rule freezes forever)
  are reseeded from the points farthest from their centroid before the
  final assignment.
- Everything runs under ``torch.inference_mode()``; on the OOM fallback
  ``torch.cuda.empty_cache()`` runs before the CPU retry.
- The CPU fallback is vectorized (reuses the JIT ``_sum_by_label``
  kernel) and shares the same convergence/reseeding semantics.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .backend import HAS_TORCH, HAS_CUDA

if HAS_TORCH:
    import torch

from ..core.clustering import _sum_by_label  # JIT per-cluster sums (CPU fallback)


class TorchKMeans:
    """
    K-Means clustering with GPU acceleration via PyTorch.

    Uses mini-batch updates and runs entirely on GPU when available.
    API mirrors the CPU ``KMeans`` class.
    """

    def __init__(
        self,
        n_clusters: int = 100,
        max_iter: int = 100,
        batch_size: int = 10000,
        tol: float = 1e-4,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state
        self._device = device or ("cuda" if HAS_CUDA else "cpu")

        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: int = 0

    # ---- public API --------------------------------------------------

    def fit(self, data: np.ndarray) -> "TorchKMeans":
        data_np = np.ascontiguousarray(data.astype(np.float32))
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        n_clusters = min(self.n_clusters, data_np.shape[0])

        if not HAS_TORCH or self._device == "cpu":
            return self._fit_cpu(data_np, n_clusters)

        try:
            return self._fit_gpu(data_np, n_clusters)
        except (RuntimeError, MemoryError):  # noqa: BLE001 — torch OOM subclasses RuntimeError
            # OOM or driver issue — free cache and fall back to CPU
            if HAS_CUDA:
                torch.cuda.empty_cache()
            return self._fit_cpu(data_np, n_clusters)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("Must call fit() before predict()")
        data_np = np.ascontiguousarray(data.astype(np.float32))
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)
        return self._assign_batch(data_np, self.centroids_)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.labels_  # type: ignore[return-value]

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Return distances from each point to every centroid."""
        if self.centroids_ is None:
            raise RuntimeError("Must call fit() before transform()")
        data_np = np.ascontiguousarray(data.astype(np.float32))
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        data_norms = np.sum(data_np**2, axis=1, keepdims=True)  # (N,1)
        cent_norms = np.sum(self.centroids_**2, axis=1)  # (K,)
        cross = data_np @ self.centroids_.T  # (N,K)
        dist_sq = data_norms + cent_norms - 2.0 * cross
        return np.sqrt(np.maximum(dist_sq, 0.0))

    # ---- GPU internals ---------------------------------------------------

    def _fit_gpu(self, data_np: np.ndarray, n_clusters: int) -> "TorchKMeans":
        """Full GPU path (vectorized mini-batch updates, no per-cluster loop)."""
        with torch.inference_mode():
            torch.manual_seed(self.random_state)
            device = torch.device(self._device)

            data_t = torch.from_numpy(data_np).to(device)
            N, D = data_t.shape

            centroids_t = self._kpp_init_gpu(data_t, n_clusters, device)

            counts = torch.ones(n_clusters, dtype=torch.float32, device=device)
            ever_updated = torch.zeros(n_clusters, dtype=torch.bool, device=device)
            batch_size = min(self.batch_size, N)
            n_iter = 0

            for it in range(self.max_iter):
                n_iter = it + 1

                idx = torch.randint(0, N, (batch_size,), device=device)
                batch = data_t[idx]

                # Assign: (batch, K)
                dists = self._pairwise_sq(batch, centroids_t)
                labels = torch.argmin(dists, dim=1)

                # Vectorized mini-batch update: per-cluster sums/counts in
                # ~4 kernels instead of a Python loop over clusters.
                sums = torch.zeros_like(centroids_t).index_add_(0, labels, batch)
                bcounts = torch.bincount(labels, minlength=n_clusters).to(torch.float32)
                updated = bcounts > 0
                ever_updated |= updated
                if not bool(updated.any()):
                    continue

                eta = 1.0 / (counts[updated] + bcounts[updated])
                new_centroids = centroids_t.clone()
                new_centroids[updated] = (1.0 - eta.unsqueeze(1)) * centroids_t[
                    updated
                ] + eta.unsqueeze(1) * (sums[updated] / bcounts[updated].unsqueeze(1))
                counts[updated] += bcounts[updated]

                shift = float((new_centroids - centroids_t).pow(2).sum().item())
                centroids_t = new_centroids

                if shift < self.tol:
                    break

            # Final assignment
            all_dists = self._pairwise_sq(data_t, centroids_t)
            all_labels = torch.argmin(all_dists, dim=1)

            # Reseed never-updated (dead) clusters from the farthest points
            dead = ~ever_updated
            n_dead = int(dead.sum().item())
            if n_dead > 0 and N > n_clusters:
                point_d = all_dists.gather(1, all_labels.unsqueeze(1)).squeeze(1)
                far = torch.topk(point_d, n_dead, largest=True).indices
                centroids_t[dead] = data_t[far]
                all_dists = self._pairwise_sq(data_t, centroids_t)
                all_labels = torch.argmin(all_dists, dim=1)

            self.centroids_ = centroids_t.cpu().numpy()
            self.labels_ = all_labels.cpu().numpy().astype(np.int32)
            self.inertia_ = float(all_dists.gather(1, all_labels.unsqueeze(1)).sum().item())
            self.n_iter_ = n_iter

            del data_t, centroids_t, all_dists, all_labels
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return self

    def _kpp_init_gpu(
        self, data_t: "torch.Tensor", k: int, device: "torch.device"
    ) -> "torch.Tensor":
        """K-Means++ initialization on GPU."""
        N, D = data_t.shape
        centroids = torch.empty(k, D, device=device, dtype=data_t.dtype)

        first = torch.randint(0, N, (1,), device=device)
        centroids[0] = data_t[first]

        min_dist_sq = self._pairwise_sq(data_t, centroids[:1]).squeeze(1)
        min_dist_sq = torch.clamp(min_dist_sq, min=0.0)  # avoid float negatives

        for i in range(1, k):
            total = min_dist_sq.sum() + 1e-10
            probs = min_dist_sq / total
            # Guard against NaN/inf from degenerate distributions
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() <= 0:
                # All points are at distance 0 — pick random
                idx = torch.randint(0, N, (1,), device=device)
            else:
                probs = probs / probs.sum()
                idx = torch.multinomial(probs, 1)
            centroids[i] = data_t[idx]
            new_dist = self._pairwise_sq(data_t, centroids[i : i + 1]).squeeze(1)
            new_dist = torch.clamp(new_dist, min=0.0)
            min_dist_sq = torch.minimum(min_dist_sq, new_dist)

        return centroids

    # ---- CPU fallback ----------------------------------------------------

    def _fit_cpu(self, data_np: np.ndarray, n_clusters: int) -> "TorchKMeans":
        """CPU fallback using vectorized NumPy (+ JIT per-cluster sums)."""
        rng = np.random.RandomState(self.random_state)
        centroids = self._kpp_init_cpu(data_np, n_clusters, rng)

        counts = np.ones(n_clusters, dtype=np.float64)
        ever_updated = np.zeros(n_clusters, dtype=bool)
        N = data_np.shape[0]
        batch_size = min(self.batch_size, N)
        n_iter = 0

        for it in range(self.max_iter):
            n_iter = it + 1
            idx = rng.randint(0, N, size=batch_size)
            batch = np.ascontiguousarray(data_np[idx])

            labels = self._assign_batch(batch, centroids)
            sums, bcounts = _sum_by_label(batch, labels.astype(np.int64), n_clusters)
            bcounts = bcounts.astype(np.float64)

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

            if shift < self.tol:
                break

        labels = self._assign_batch(data_np, centroids)

        # Reseed dead clusters from the farthest points, reassign once
        dead = ~ever_updated
        n_dead = int(dead.sum())
        if n_dead > 0 and N > n_clusters:
            data_norms = np.sum(data_np**2, axis=1)
            cent_norms = np.sum(centroids[labels] ** 2, axis=1)
            cross = np.sum(data_np * centroids[labels], axis=1)
            point_d = data_norms + cent_norms - 2.0 * cross
            far = np.argpartition(point_d, N - n_dead)[N - n_dead :]
            centroids[dead] = data_np[far]
            labels = self._assign_batch(data_np, centroids)

        cent_norms = np.sum(centroids[labels] ** 2, axis=1)
        cross = np.sum(data_np * centroids[labels], axis=1)
        data_norms = np.sum(data_np**2, axis=1)
        inertia = float(np.sum(data_norms + cent_norms - 2.0 * cross))

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        return self

    # ---- helpers -----------------------------------------------------

    @staticmethod
    def _pairwise_sq(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        """Squared L2 distance matrix via Gram trick."""
        a_sq = (a**2).sum(dim=1, keepdim=True)  # (N,1)
        b_sq = (b**2).sum(dim=1, keepdim=True).t()  # (1,K)
        cross = a @ b.t()
        return a_sq + b_sq - 2.0 * cross

    @staticmethod
    def _kpp_init_cpu(data: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
        """K-Means++ init on CPU with running min-distance."""
        N, D = data.shape
        centroids = np.zeros((k, D), dtype=np.float32)

        idx = rng.randint(N)
        centroids[0] = data[idx]

        diff = data - centroids[0]
        min_dist_sq = np.sum(diff * diff, axis=1)

        for i in range(1, k):
            total = min_dist_sq.sum() + 1e-10
            probs = min_dist_sq / total
            if probs.sum() <= 0:
                # Degenerate: all remaining points identical → random pick
                idx = rng.randint(N)
            else:
                idx = rng.choice(N, p=probs / probs.sum())
            centroids[i] = data[idx]
            diff = data - centroids[i]
            new_dist = np.sum(diff * diff, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, new_dist)

        return centroids

    @staticmethod
    def _assign_batch(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign points to nearest centroid (vectorized NumPy)."""
        data_norms = np.sum(data**2, axis=1, keepdims=True)  # (N,1)
        cent_norms = np.sum(centroids**2, axis=1)  # (K,)
        cross = data @ centroids.T  # (N,K)
        dist_sq = data_norms + cent_norms - 2.0 * cross
        return np.argmin(dist_sq, axis=1).astype(np.int32)
