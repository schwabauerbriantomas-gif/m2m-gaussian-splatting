"""
GPU acceleration module for M2M Gaussian Splatting.

Auto-detects CUDA availability via PyTorch. Falls back to CPU (NumPy/Numba)
transparently when no GPU is present or PyTorch is not installed.
"""

from .backend import detect_device, HAS_TORCH, HAS_CUDA, get_gpu_info
from .gpu_kmeans import TorchKMeans
from .gpu_search import GPUSearcher

__all__ = [
    "detect_device",
    "HAS_TORCH",
    "HAS_CUDA",
    "get_gpu_info",
    "TorchKMeans",
    "GPUSearcher",
]
