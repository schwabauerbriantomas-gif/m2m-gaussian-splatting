"""
M2M Gaussian Splatting Package

A hierarchical memory management system for 3D Gaussian Splatting
with optimized encoding and clustering algorithms.
"""

__version__ = "2.0.0"
__author__ = "Brian Schwabauer"

from .core.splat_types import GaussianSplat, SplatEmbedding
from .core.encoding import SinusoidalPositionEncoder, ColorHistogramEncoder
from .core.hrm2_engine import HRM2Engine
from .memory.manager import SplatMemoryManager
from .gpu import detect_device, HAS_CUDA

__all__ = [
    "GaussianSplat",
    "SplatEmbedding",
    "SinusoidalPositionEncoder",
    "ColorHistogramEncoder",
    "HRM2Engine",
    "SplatMemoryManager",
    "detect_device",
    "HAS_CUDA",
]
version = "2.0.0"
