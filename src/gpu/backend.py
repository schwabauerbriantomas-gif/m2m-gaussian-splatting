"""
Device detection backend.

Determines whether CUDA is available via PyTorch and exposes
helpers for querying GPU info.
"""

import logging

logger = logging.getLogger(__name__)

HAS_TORCH = False
HAS_CUDA = False

try:
    import torch

    HAS_TORCH = True
    if torch.cuda.is_available():
        HAS_CUDA = True
except ImportError:
    torch = None  # type: ignore[assignment]


def detect_device():
    """
    Return the best available compute device.

    Returns:
        ``"cuda"`` if CUDA is available, otherwise ``"cpu"``.
    """
    if HAS_CUDA:
        return "cuda"
    return "cpu"


def get_gpu_info() -> dict:
    """
    Return information about the active GPU (if any).

    Keys: ``device``, ``name``, ``vram_total_gb``, ``vram_free_gb``,
    ``cuda_version``.  When no GPU is available, only ``device`` is set.
    """
    info: dict = {"device": detect_device()}
    if HAS_CUDA:
        prop = torch.cuda.get_device_properties(0)
        info["name"] = prop.name
        info["vram_total_gb"] = prop.total_memory / (1024**3)
        try:
            free, _ = torch.cuda.mem_get_info()
            info["vram_free_gb"] = free / (1024**3)
        except RuntimeError:
            info["vram_free_gb"] = -1.0
        info["cuda_version"] = torch.version.cuda
    return info
