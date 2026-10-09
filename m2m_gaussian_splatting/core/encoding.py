"""
Encoding functions optimized with Numba JIT.

This module provides fast encoding functions for converting
Gaussian splat attributes into embedding vectors for indexing.

v2.1 (audit 2026-10) changes:
- ``SinusoidalPositionEncoder.encode`` always returns a 2D ``(N, dim)``
  array (previously a single point returned 1D, which crashed
  ``FullEmbeddingBuilder.build`` with ValueError for N=1).
- All encoders handle N=0 and NaN inputs explicitly instead of crashing
  (empty color array used to raise on ``colors.max()``) or silently
  emitting a zero embedding (NaN color).
- Numba kernels that use ``prange`` declare ``parallel=True`` (previously
  ``prange`` silently degraded to a serial loop); inputs are forced
  C-contiguous to avoid recompilation on F-order arrays.
- Color histogram: Gaussian weights and bin offsets are precomputed at
  module level (previously rebuilt per call, 125 ``exp`` evaluations per
  splat), rows are L2-normalized (corner colors previously got a smaller
  norm and biased L2 distances), and the ``[0,255]`` auto-detection can
  be overridden with an explicit ``color_space``.
- Position encoding: optional fixed scene bounds (``fit`` / pre-fit
  constructor arg). By default normalization remains per-batch for
  backward compatibility, but note that per-batch min/max makes the
  embedding of a point depend on the rest of the batch — pass fixed
  bounds (or call ``fit``) for query-time consistency. Frequencies now
  include the standard π factor (sin(π·2^d·x), NeRF-style); the old
  no-π scheme left the lowest bands nearly linear.
- Attribute encoder: opacities are clipped to [0,1], scales clamped to
  a small positive minimum and NaNs sanitized, preventing NaN/~1e8
  ratio features from contaminating the index.
"""

import numpy as np
from typing import Tuple, Optional

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


# ==================== POSITION ENCODING ====================


@njit(fastmath=True, cache=True, parallel=True)
def _sinusoidal_position_encoding_numba(
    positions: np.ndarray,
    dim: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    use_pi: int,
) -> np.ndarray:
    """
    Sinusoidal Position Encoding for 3D coordinates.

    Uses multi-frequency sinusoids similar to NeRF positional encoding:
    sin/cos of (π·2^d · coordinate) for d in [0, n_freq).
    Normalization bounds are passed in (computed by the wrapper).

    Args:
        positions: (N, 3) array of 3D positions
        dim: Output dimension (must be divisible by 6)
        min_x..max_z: normalization bounds
        use_pi: 1 to use the standard π·2^d frequencies, 0 for legacy 2^d

    Returns:
        (N, dim) array of position encodings
    """
    N = positions.shape[0]
    encodings = np.zeros((N, dim), dtype=np.float32)

    range_x = max_x - min_x + 1e-8
    range_y = max_y - min_y + 1e-8
    range_z = max_z - min_z + 1e-8

    n_freq = dim // 6
    pi = 3.141592653589793
    base = pi if use_pi == 1 else 1.0

    for i in prange(N):
        # Normalize to [0, 1]
        x = (positions[i, 0] - min_x) / range_x
        y = (positions[i, 1] - min_y) / range_y
        z = (positions[i, 2] - min_z) / range_z

        for d in range(n_freq):
            freq = base * (2.0**d)
            idx = d * 6

            encodings[i, idx] = np.sin(x * freq)
            encodings[i, idx + 1] = np.cos(x * freq)
            encodings[i, idx + 2] = np.sin(y * freq)
            encodings[i, idx + 3] = np.cos(y * freq)
            encodings[i, idx + 4] = np.sin(z * freq)
            encodings[i, idx + 5] = np.cos(z * freq)

    return encodings


def _batch_bounds(positions: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Per-batch min/max per axis (legacy normalization semantics)."""
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    return (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )


class SinusoidalPositionEncoder:
    """
    Encoder for 3D positions using sinusoidal functions.

    Similar to positional encoding in Transformers and NeRF.

    By default, coordinates are normalized by the min/max of each batch,
    which makes an embedding depend on the whole batch (a single point
    always encodes to a constant). Call :meth:`fit` (or pass
    ``bounds``) with the scene bounds to get batch-independent,
    query-consistent encodings.
    """

    def __init__(
        self,
        dim: int = 64,
        bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
        use_pi_frequencies: bool = True,
    ):
        """
        Initialize encoder.

        Args:
            dim: Output dimension. Uses dim//6 frequencies of (sin,cos) per
                 axis (x,y,z), filling dim//6*6 columns; remaining columns
                 are zero-padded so the output is exactly ``dim`` dims.
            bounds: Optional fixed scene bounds
                    (min_x, max_x, min_y, max_y, min_z, max_z).
            use_pi_frequencies: Use standard sin(π·2^d·x) frequencies
                 (NeRF). False restores the legacy sin(2^d·x) scheme.
        """
        self.dim = max(dim, 6)
        self.bounds = bounds
        self.use_pi_frequencies = use_pi_frequencies

    def fit(self, positions: np.ndarray) -> "SinusoidalPositionEncoder":
        """
        Fix the normalization bounds from a reference (scene) point set.

        After fitting, encodings are independent of the batch being
        encoded, so single-point queries match the indexed base.
        """
        positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        if positions.shape[0] == 0:
            raise ValueError("fit() requires at least one position")
        self.bounds = _batch_bounds(positions)
        return self

    def encode(self, positions: np.ndarray) -> np.ndarray:
        """
        Encode 3D positions.

        Args:
            positions: (N, 3) or (3,) array

        Returns:
            (N, dim) array — always 2D, even for a single point
        """
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        positions = positions.reshape(-1, 3)

        if positions.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        if np.isnan(positions).any():
            raise ValueError("positions contain NaN")

        if self.bounds is not None:
            bounds = self.bounds
        else:
            bounds = _batch_bounds(positions)

        return _sinusoidal_position_encoding_numba(
            positions,
            self.dim,
            *bounds,
            1 if self.use_pi_frequencies else 0,
        )


# ==================== COLOR ENCODING ====================

# Precomputed Gaussian kernel over the ±2 bin neighborhood:
# weights depend only on the offset, not on the splat, so they are
# computed once at module level (numba captures globals as constants).
_NEIGHBORHOOD_OFFSETS = np.array(
    [(dr, dg, db) for dr in range(-2, 3) for dg in range(-2, 3) for db in range(-2, 3)],
    dtype=np.int32,
)  # (125, 3)
_NEIGHBORHOOD_WEIGHTS = np.exp(
    -(_NEIGHBORHOOD_OFFSETS**2).sum(axis=1).astype(np.float32) / 4.0
).astype(
    np.float32
)  # (125,)


@njit(fastmath=True, cache=True, parallel=True)
def _color_histogram_encoding_numba(
    colors: np.ndarray, n_bins: int, offsets: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Histogram-based color encoding.

    Creates a sparse histogram with Gaussian-smoothed contributions.

    Args:
        colors: (N, 3) array of RGB colors in [0, 1]
        n_bins: Number of bins per channel (8 -> 512 dims)
        offsets: (125, 3) int32 neighborhood offsets (module constant)
        weights: (125,) float32 Gaussian weights (module constant)

    Returns:
        (N, n_bins³) array of color encodings
    """
    N = colors.shape[0]
    n_bins_cubed = n_bins * n_bins * n_bins  # 512 for n_bins=8
    encodings = np.zeros((N, n_bins_cubed), dtype=np.float32)

    # Only evaluate bins within radius 2 of the target — Gaussian kernel
    # decays as exp(-d²/4), so contributions beyond d=2 are negligible.
    for i in prange(N):
        r, g, b = colors[i, 0], colors[i, 1], colors[i, 2]

        # Quantize to bins
        bin_r = min(int(r * n_bins), n_bins - 1)
        bin_g = min(int(g * n_bins), n_bins - 1)
        bin_b = min(int(b * n_bins), n_bins - 1)

        for j in range(offsets.shape[0]):
            br = bin_r + offsets[j, 0]
            bg = bin_g + offsets[j, 1]
            bb = bin_b + offsets[j, 2]
            if br < 0 or br >= n_bins or bg < 0 or bg >= n_bins or bb < 0 or bb >= n_bins:
                continue
            idx = (br * n_bins + bg) * n_bins + bb
            encodings[i, idx] = weights[j]

    return encodings


class ColorHistogramEncoder:
    """
    Encoder for RGB colors using histogram representation.

    Output rows are L2-normalized: corner colors (black/white) touch
    fewer bins and would otherwise get a systematically smaller norm,
    biasing L2-based search against them.
    """

    def __init__(self, n_bins: int = 8, color_space: Optional[str] = None):
        """
        Initialize encoder.

        Args:
            n_bins: Bins per channel (output dim = n_bins^3)
            color_space: Explicit input scale: ``'01'`` for [0,1] floats or
                ``'255'`` for [0,255] bytes. None (default) keeps the
                legacy auto-detection (divide by 255 when the batch max
                exceeds 1.5) — note auto-detection misclassifies dark
                scenes in [0,255] whose max is below 1.5.
        """
        if color_space not in (None, "01", "255"):
            raise ValueError(f"color_space must be None, '01' or '255', got {color_space!r}")
        self.n_bins = n_bins
        self.color_space = color_space
        self.dim = n_bins**3  # 512 for n_bins=8

    def encode(self, colors: np.ndarray) -> np.ndarray:
        """
        Encode RGB colors.

        Args:
            colors: (N, 3) array in [0, 1] or [0, 255]

        Returns:
            (N, n_bins³) L2-normalized array (NaN input → ValueError)
        """
        colors = np.ascontiguousarray(colors, dtype=np.float32)
        if colors.ndim == 1:
            colors = colors.reshape(1, -1)
        colors = colors.reshape(-1, 3)

        if colors.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        if np.isnan(colors).any():
            raise ValueError("colors contain NaN")

        if self.color_space == "255":
            colors = colors / 255.0
        elif self.color_space == "01":
            pass
        else:
            # Legacy auto-detection: if any value > 1.0, treat as [0,255]
            cmax = float(colors.max())
            if cmax > 1.5:  # threshold accounts for float noise in [0,1]
                colors = colors / 255.0

        colors = np.clip(colors, 0.0, 1.0)

        enc = _color_histogram_encoding_numba(
            colors, self.n_bins, _NEIGHBORHOOD_OFFSETS, _NEIGHBORHOOD_WEIGHTS
        )
        # L2-normalize rows (avoids corner-color norm bias). Rows with no
        # active bin (impossible for valid inputs, but defensive) stay zero.
        norms = np.linalg.norm(enc, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        enc /= norms
        return enc


# ==================== ATTRIBUTE ENCODING ====================


@njit(fastmath=True, cache=True, parallel=True)
def _attribute_encoding_numba(
    opacities: np.ndarray, scales: np.ndarray, rotations: np.ndarray
) -> np.ndarray:
    """
    Attribute encoding for opacity, scale, and rotation.

    Creates hand-crafted features from splat attributes.
    Inputs are expected pre-sanitized by the wrapper (no NaN, positive
    scales, opacities in [0,1]).

    Args:
        opacities: (N,) array
        scales: (N, 3) array
        rotations: (N, 4) array (quaternions)

    Returns:
        (N, 64) array of attribute encodings
    """
    N = opacities.shape[0]
    encodings = np.zeros((N, 64), dtype=np.float32)

    for i in prange(N):
        o = opacities[i]

        # Opacity features (8 dims)
        encodings[i, 0] = o
        encodings[i, 1] = o * o
        encodings[i, 2] = o * o * o
        encodings[i, 3] = np.sqrt(o + 1e-8)
        encodings[i, 4] = np.log(o + 1e-8)
        encodings[i, 5] = 1.0 - o
        encodings[i, 6] = 1.0 if o > 0.5 else 0.0
        encodings[i, 7] = 1.0 if o < 0.5 else 0.0

        # Scale features (24 dims)
        sx, sy, sz = scales[i, 0], scales[i, 1], scales[i, 2]
        encodings[i, 8] = sx
        encodings[i, 9] = sy
        encodings[i, 10] = sz
        encodings[i, 11] = sx * sy * sz
        encodings[i, 12] = (sx + sy + sz) / 3.0
        encodings[i, 13] = np.sqrt(sx * sx + sy * sy + sz * sz)
        encodings[i, 14] = sx / (sy + 1e-6)
        encodings[i, 15] = sx / (sz + 1e-6)
        encodings[i, 16] = sy / (sz + 1e-6)
        encodings[i, 17] = sx * sx
        encodings[i, 18] = sy * sy
        encodings[i, 19] = sz * sz
        encodings[i, 20] = np.log(sx)
        encodings[i, 21] = np.log(sy)
        encodings[i, 22] = np.log(sz)
        # 23-31: zero padding

        # Rotation features (32 dims)
        qw, qx, qy, qz = rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]
        encodings[i, 32] = qw
        encodings[i, 33] = qx
        encodings[i, 34] = qy
        encodings[i, 35] = qz
        encodings[i, 36] = qw * qw
        encodings[i, 37] = qx * qx
        encodings[i, 38] = qy * qy
        encodings[i, 39] = qz * qz
        encodings[i, 40] = qw * qx
        encodings[i, 41] = qw * qy
        encodings[i, 42] = qw * qz
        encodings[i, 43] = qx * qy
        encodings[i, 44] = qx * qz
        encodings[i, 45] = qy * qz
        # 46-63: zero padding

    return encodings


class AttributeEncoder:
    """
    Encoder for splat attributes (opacity, scale, rotation).

    Inputs are sanitized before encoding: NaNs are rejected, opacities
    clipped to [0,1], scales clamped to ≥1e-6 (log/ratio features would
    otherwise produce NaN or ~1e8 magnitudes that poison the index).
    """

    def __init__(self, dim: int = 64):
        """Initialize encoder."""
        self.dim = dim

    def encode(
        self, opacities: np.ndarray, scales: np.ndarray, rotations: np.ndarray
    ) -> np.ndarray:
        """
        Encode splat attributes.

        Args:
            opacities: (N,) array
            scales: (N, 3) array
            rotations: (N, 4) array

        Returns:
            (N, 64) array
        """
        opacities = np.ascontiguousarray(opacities, dtype=np.float32).reshape(-1)
        scales = np.ascontiguousarray(scales, dtype=np.float32).reshape(-1, 3)
        rotations = np.ascontiguousarray(rotations, dtype=np.float32).reshape(-1, 4)

        N = opacities.shape[0]
        if scales.shape[0] != N or rotations.shape[0] != N:
            raise ValueError(
                f"shape mismatch: opacities N={N}, scales N={scales.shape[0]}, "
                f"rotations N={rotations.shape[0]}"
            )
        if N == 0:
            return np.zeros((0, 64), dtype=np.float32)
        if np.isnan(opacities).any() or np.isnan(scales).any() or np.isnan(rotations).any():
            raise ValueError("attributes contain NaN")

        opacities = np.clip(opacities, 0.0, 1.0)
        scales = np.maximum(scales, 1e-6)

        return _attribute_encoding_numba(opacities, scales, rotations)


# ==================== FULL EMBEDDING ====================


class FullEmbeddingBuilder:
    """
    Builder for complete 640D splat embeddings.
    """

    def __init__(
        self,
        position_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
        color_space: Optional[str] = None,
        use_pi_frequencies: bool = True,
    ):
        """Initialize all encoders.

        Args:
            position_bounds: optional fixed scene bounds for the position
                encoder (see SinusoidalPositionEncoder). When None the
                position encoding normalizes per batch (legacy behavior).
            color_space: optional explicit color space ('01' / '255').
            use_pi_frequencies: π-based positional frequencies (NeRF).
        """
        self.pos_encoder = SinusoidalPositionEncoder(
            dim=64, bounds=position_bounds, use_pi_frequencies=use_pi_frequencies
        )
        self.color_encoder = ColorHistogramEncoder(n_bins=8, color_space=color_space)
        self.attr_encoder = AttributeEncoder(dim=64)  # 64 dims

        # Actual dimensions
        self._pos_dim = self.pos_encoder.dim
        self._color_dim = self.color_encoder.dim
        self._attr_dim = 64
        self._total_dim = self._pos_dim + self._color_dim + self._attr_dim

    @property
    def embedding_dim(self) -> int:
        """Total embedding dimension."""
        return self._total_dim

    def fit_positions(self, positions: np.ndarray) -> "FullEmbeddingBuilder":
        """
        Fix position normalization bounds from a reference point set.

        Call this before ``build`` when the same points will later be
        encoded one at a time (e.g. queries), otherwise per-batch
        normalization makes single-point embeddings inconsistent with
        the indexed batch.
        """
        self.pos_encoder.fit(positions)
        return self

    def build(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
    ) -> np.ndarray:
        """
        Build full 640D embeddings.

        Args:
            positions: (N, 3) positions
            colors: (N, 3) colors
            opacities: (N,) opacities
            scales: (N, 3) scales
            rotations: (N, 4) quaternions

        Returns:
            (N, 640) float32 embeddings
        """
        pos_enc = np.atleast_2d(self.pos_encoder.encode(positions))
        color_enc = np.atleast_2d(self.color_encoder.encode(colors))
        attr_enc = np.atleast_2d(self.attr_encoder.encode(opacities, scales, rotations))

        if not (pos_enc.shape[0] == color_enc.shape[0] == attr_enc.shape[0]):
            raise ValueError(
                f"row mismatch: positions {pos_enc.shape[0]}, colors "
                f"{color_enc.shape[0]}, attributes {attr_enc.shape[0]}"
            )

        return np.concatenate([pos_enc, color_enc, attr_enc], axis=1).astype(
            np.float32, copy=False
        )


# Convenience function
_BUILDER = FullEmbeddingBuilder()


def build_full_embedding(
    positions: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    """
    Build full 640D embedding for splats.

    Convenience function using a shared FullEmbeddingBuilder.
    """
    return _BUILDER.build(positions, colors, opacities, scales, rotations)
