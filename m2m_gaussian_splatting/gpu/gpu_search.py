"""
GPU-accelerated similarity search using PyTorch CUDA.

Provides brute-force L2 search on GPU. When CUDA is unavailable,
falls back to NumPy automatically.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from .backend import HAS_TORCH, HAS_CUDA

if HAS_TORCH:
    import torch


class GPUSearcher:
    """
    Brute-force k-NN search on GPU.

    Uploads the index once at construction. Query batches are sent
    to the GPU, distances computed via matrix multiply, and top-k
    selected with ``torch.topk``.
    """

    def __init__(
        self,
        index_vectors: np.ndarray,
        device: Optional[str] = None,
        max_batch_size: int = 256,
    ):
        self._device = device or ("cuda" if HAS_CUDA else "cpu")
        self.max_batch_size = max_batch_size

        vectors = np.ascontiguousarray(index_vectors.astype(np.float32))
        self._N, self._D = vectors.shape

        if not HAS_TORCH or self._device == "cpu":
            self._index_np = vectors
            self._index_t = None
            self._norms_np = np.sum(vectors**2, axis=1)
            self._norms_t = None
            return

        # GPU path
        try:
            self._index_t = torch.from_numpy(vectors).to(self._device)
            self._norms_t = (self._index_t**2).sum(dim=1)  # (N,)
            self._index_np = None
            self._norms_np = None
        except (RuntimeError, torch.cuda.OutOfMemoryError):  # type: ignore
            self._index_np = vectors
            self._norms_np = np.sum(vectors**2, axis=1)
            self._index_t = None
            self._norms_t = None
            self._device = "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_gpu(self) -> bool:
        return self._device == "cuda"

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-query search.

        Returns:
            ``(indices, distances)`` — both shape ``(k,)``.
        """
        q = np.ascontiguousarray(query.astype(np.float32).flatten())
        indices, distances = self.batch_search(q.reshape(1, -1), k)
        return indices[0], distances[0]

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search.

        Args:
            queries: (B, D) array
            k: number of neighbors

        Returns:
            ``(indices, distances)`` — shapes (B, k).
        """
        queries = np.ascontiguousarray(queries.astype(np.float32))
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        k = min(k, self._N)

        if self.is_gpu and self._index_t is not None:
            return self._batch_gpu(queries, k)
        return self._batch_cpu(queries, k)

    # ---- GPU path ----------------------------------------------------

    def _batch_gpu(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        device = torch.device(self._device)
        q_t = torch.from_numpy(queries).to(device)
        B = q_t.shape[0]
        all_idx = []
        all_dist = []

        for start in range(0, B, self.max_batch_size):
            end = min(start + self.max_batch_size, B)
            chunk = q_t[start:end]  # (b, D)
            chunk_size = chunk.shape[0]

            # (b, N) = ||q||² + ||m||² - 2·q·m
            q_sq = (chunk**2).sum(dim=1, keepdim=True)  # (b,1)
            dist_sq = q_sq + self._norms_t.unsqueeze(0) - 2.0 * (chunk @ self._index_t.t())

            # top-k smallest distances
            neg_dist = -dist_sq  # topk on negative = bottom-k
            vals, idx = torch.topk(neg_dist, k=k, dim=1, largest=True, sorted=True)
            all_idx.append(idx.cpu().numpy())
            all_dist.append(torch.sqrt(torch.clamp(-vals, min=0.0)).cpu().numpy())

        indices = np.concatenate(all_idx, axis=0) if len(all_idx) > 1 else all_idx[0]
        distances = np.concatenate(all_dist, axis=0) if len(all_dist) > 1 else all_dist[0]
        return indices.astype(np.int32), distances.astype(np.float32)

    # ---- CPU path ----------------------------------------------------

    def _batch_cpu(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q_sq = np.sum(queries**2, axis=1, keepdims=True)  # (B,1)
        cross = queries @ self._index_np.T  # (B,N)
        dist_sq = q_sq + self._norms_np - 2.0 * cross
        np.maximum(dist_sq, 0.0, out=dist_sq)

        # argpartition for top-k per row
        k_actual = min(k, dist_sq.shape[1])
        indices = np.argpartition(dist_sq, k_actual - 1, axis=1)[:, :k_actual]
        # sort within each row
        row_dist = np.take_along_axis(dist_sq, indices, axis=1)
        order = np.argsort(row_dist, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        distances = np.sqrt(np.take_along_axis(dist_sq, indices, axis=1))
        return indices.astype(np.int32), distances.astype(np.float32)
