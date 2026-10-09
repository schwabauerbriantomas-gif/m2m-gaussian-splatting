"""
M2M Gaussian Splatting Package

A hierarchical memory management system for 3D Gaussian Splatting
with optimized encoding and clustering algorithms.
"""

__version__ = "2.1.1"
__author__ = "Brian Schwabauer"

from .core.splat_types import GaussianSplat, SplatEmbedding
from .core.encoding import SinusoidalPositionEncoder, ColorHistogramEncoder
from .core.clustering import KMeansResult
from .core.hrm2_engine import HRM2Engine, HRM2Config, SearchResult
from .memory.manager import SplatMemoryManager, MemoryConfig, MemoryStats
from .gpu import detect_device, HAS_CUDA

__all__ = [
    "GaussianSplat",
    "SplatEmbedding",
    "SinusoidalPositionEncoder",
    "ColorHistogramEncoder",
    "KMeansResult",
    "HRM2Engine",
    "HRM2Config",
    "SearchResult",
    "SplatMemoryManager",
    "MemoryConfig",
    "MemoryStats",
    "detect_device",
    "HAS_CUDA",
]
