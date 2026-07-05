"""
Tests for M2M Gaussian Splatting
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSplatTypes:
    """Test splat data types."""

    def test_gaussian_splat_creation(self):
        """Test creating a Gaussian splat."""
        from src.core.splat_types import GaussianSplat

        splat = GaussianSplat(
            id=0,
            position=np.array([1.0, 2.0, 3.0]),
            color=np.array([0.5, 0.6, 0.7]),
            opacity=0.9,
        )

        assert splat.id == 0
        assert splat.position.shape == (3,)
        assert splat.color.shape == (3,)
        assert splat.opacity == 0.9

    def test_splat_embedding(self):
        """Test splat embedding."""
        from src.core.splat_types import SplatEmbedding

        embedding = SplatEmbedding(
            splat_id=0,
            position_encoding=np.zeros(64),
            color_encoding=np.zeros(512),
            attribute_encoding=np.zeros(64),
        )

        full = embedding.full_embedding
        assert full.shape == (640,)

    def test_splat_quaternion_normalization(self):
        """Test that quaternion is normalized on creation."""
        from src.core.splat_types import GaussianSplat

        splat = GaussianSplat(
            id=0,
            rotation=np.array([2.0, 0.0, 0.0, 0.0]),
        )
        assert abs(np.linalg.norm(splat.rotation) - 1.0) < 1e-6

    def test_splat_covariance(self):
        """Test covariance matrix computation."""
        from src.core.splat_types import GaussianSplat

        splat = GaussianSplat(
            id=0,
            scale=np.array([1.0, 2.0, 3.0]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        cov = splat.covariance_3d
        assert cov.shape == (3, 3)
        # Diagonal should match scale squared (identity rotation)
        assert abs(cov[0, 0] - 1.0) < 1e-5
        assert abs(cov[1, 1] - 4.0) < 1e-5
        assert abs(cov[2, 2] - 9.0) < 1e-5

    def test_splat_serialization(self):
        """Test to_dict / from_dict round-trip."""
        from src.core.splat_types import GaussianSplat

        original = GaussianSplat(
            id=42,
            position=np.array([1.0, 2.0, 3.0]),
            color=np.array([0.5, 0.6, 0.7]),
            opacity=0.8,
            scale=np.array([0.1, 0.2, 0.3]),
        )
        d = original.to_dict()
        restored = GaussianSplat.from_dict(d)
        assert restored.id == 42
        np.testing.assert_array_almost_equal(restored.position, original.position)
        np.testing.assert_array_almost_equal(restored.scale, original.scale)


class TestEncoding:
    """Test encoding functions."""

    def test_position_encoding(self):
        """Test sinusoidal position encoding."""
        from src.core.encoding import SinusoidalPositionEncoder

        encoder = SinusoidalPositionEncoder(dim=64)
        positions = np.random.randn(100, 3).astype(np.float32)

        encodings = encoder.encode(positions)

        assert encodings.shape == (100, 64)

    def test_position_encoding_dim_not_truncated(self):
        """Ensure position encoding outputs exactly the requested dim."""
        from src.core.encoding import SinusoidalPositionEncoder

        for dim in [6, 12, 24, 64, 66]:
            encoder = SinusoidalPositionEncoder(dim=dim)
            positions = np.random.randn(10, 3).astype(np.float32)
            encodings = encoder.encode(positions)
            assert encodings.shape == (10, dim), f"Failed for dim={dim}"

    def test_color_encoding(self):
        """Test color histogram encoding."""
        from src.core.encoding import ColorHistogramEncoder

        encoder = ColorHistogramEncoder(n_bins=8)
        colors = np.random.rand(100, 3).astype(np.float32)

        encodings = encoder.encode(colors)

        assert encodings.shape == (100, 512)

    def test_color_encoding_255_normalization(self):
        """Test that [0,255] colors are normalized to [0,1]."""
        from src.core.encoding import ColorHistogramEncoder

        encoder = ColorHistogramEncoder(n_bins=8)
        colors_255 = np.array([[255.0, 0.0, 0.0]], dtype=np.float32)
        colors_01 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        enc_255 = encoder.encode(colors_255)
        enc_01 = encoder.encode(colors_01)
        np.testing.assert_array_almost_equal(enc_255, enc_01, decimal=5)

    def test_full_embedding(self):
        """Test full embedding builder."""
        from src.core.encoding import FullEmbeddingBuilder

        builder = FullEmbeddingBuilder()

        N = 100
        positions = np.random.randn(N, 3).astype(np.float32)
        colors = np.random.rand(N, 3).astype(np.float32)
        opacities = np.random.rand(N).astype(np.float32)
        scales = np.exp(np.random.randn(N, 3).astype(np.float32) * -2)
        rotations = np.random.randn(N, 4).astype(np.float32)

        embeddings = builder.build(positions, colors, opacities, scales, rotations)

        assert embeddings.shape == (N, 640)

    def test_attribute_encoding(self):
        """Test attribute encoding for opacity/scale/rotation."""
        from src.core.encoding import AttributeEncoder

        encoder = AttributeEncoder()
        N = 50
        opacities = np.random.rand(N).astype(np.float32)
        scales = np.exp(np.random.randn(N, 3).astype(np.float32) * -2)
        rotations = np.random.randn(N, 4).astype(np.float32)

        encodings = encoder.encode(opacities, scales, rotations)
        assert encodings.shape == (N, 64)


class TestClustering:
    """Test clustering functions."""

    def test_kmeans_jit(self):
        """Test K-Means JIT implementation."""
        from src.core.clustering import KMeansJIT

        np.random.seed(42)
        data = np.random.randn(1000, 64).astype(np.float32)

        kmeans = KMeansJIT(n_clusters=10, random_state=42)
        kmeans.fit(data)

        assert kmeans.centroids_.shape == (10, 64)
        assert kmeans.labels_.shape == (1000,)

        labels = kmeans.predict(data)
        assert labels.shape == (1000,)

    def test_kmeans_transform(self):
        """Test transform produces distance matrix."""
        from src.core.clustering import KMeansJIT

        np.random.seed(42)
        data = np.random.randn(500, 32).astype(np.float32)
        kmeans = KMeansJIT(n_clusters=5, random_state=42)
        kmeans.fit(data)

        dists = kmeans.transform(data[:10])
        assert dists.shape == (10, 5)
        assert np.all(dists >= 0)

    def test_kmeans_convergence(self):
        """Test that K-Means produces stable clusters."""
        from src.core.clustering import KMeans

        # Three well-separated blobs
        blob1 = np.random.randn(100, 4).astype(np.float32) + np.array([0, 0, 0, 0])
        blob2 = np.random.randn(100, 4).astype(np.float32) + np.array([10, 10, 10, 10])
        blob3 = np.random.randn(100, 4).astype(np.float32) + np.array([-10, -10, -10, -10])
        data = np.vstack([blob1, blob2, blob3])

        kmeans = KMeans(n_clusters=3, random_state=42, use_mini_batch=False)
        kmeans.fit(data)

        # Each blob should be mostly in one cluster
        labels = kmeans.labels_
        assert len(np.unique(labels[:100])) == 1 or len(np.unique(labels[100:200])) == 1


class TestHRM2Engine:
    """Test HRM2 engine."""

    def test_engine_creation(self):
        """Test engine creation."""
        from src.core.hrm2_engine import HRM2Engine

        engine = HRM2Engine(n_coarse=10, n_fine=50)

        assert engine.n_coarse == 10
        assert engine.n_fine == 50

    def test_engine_indexing(self):
        """Test engine indexing."""
        from src.core.hrm2_engine import HRM2Engine, generate_test_splats

        splats = generate_test_splats(1000, seed=42)

        engine = HRM2Engine(n_coarse=10, n_fine=50)
        engine.add_splats(splats)
        build_time = engine.index()

        assert engine._is_indexed
        assert build_time > 0

    def test_engine_query(self):
        """Test engine query."""
        from src.core.hrm2_engine import HRM2Engine, generate_test_splats

        splats = generate_test_splats(1000, seed=42)

        engine = HRM2Engine(n_coarse=10, n_fine=50)
        engine.add_splats(splats)
        engine.index()

        query = engine.embeddings[0]
        results = engine.query(query, k=10)

        assert len(results) == 10
        # First result should be the query splat itself
        assert results[0][0].id == splats[0].id

    def test_engine_query_with_details(self):
        """Test query_with_details returns SearchResult objects."""
        from src.core.hrm2_engine import HRM2Engine, generate_test_splats, SearchResult

        splats = generate_test_splats(500, seed=42)
        engine = HRM2Engine(n_coarse=5, n_fine=10)
        engine.add_splats(splats)
        engine.index()

        results = engine.query_with_details(engine.embeddings[0], k=5)
        assert len(results) == 5
        assert isinstance(results[0], SearchResult)
        assert results[0].splat_id == splats[0].id

    def test_engine_batch_query(self):
        """Test batch query."""
        from src.core.hrm2_engine import HRM2Engine, generate_test_splats

        splats = generate_test_splats(1000, seed=42)
        engine = HRM2Engine(n_coarse=10, n_fine=50)
        engine.add_splats(splats)
        engine.index()

        queries = engine.embeddings[:5]
        results = engine.batch_query(queries, k=10)

        assert len(results) == 5
        assert all(len(r) == 10 for r in results)

    def test_engine_empty_query(self):
        """Test query on empty engine."""
        from src.core.hrm2_engine import HRM2Engine

        engine = HRM2Engine(n_coarse=5, n_fine=10)
        with pytest.raises(RuntimeError, match="Index not built"):
            engine.query(np.zeros(640, dtype=np.float32), k=10)

    def test_engine_stats(self):
        """Test that stats are properly tracked."""
        from src.core.hrm2_engine import HRM2Engine, generate_test_splats

        splats = generate_test_splats(500, seed=42)
        engine = HRM2Engine(n_coarse=5, n_fine=10)
        engine.add_splats(splats)
        engine.index()

        engine.query(engine.embeddings[0], k=5)

        stats = engine.get_stats()
        assert stats.total_queries == 1
        assert stats.avg_query_time > 0
        assert stats.n_splats == 500

    def test_engine_clear(self):
        """Test clear() resets state."""
        from src.core.hrm2_engine import HRM2Engine, generate_test_splats

        splats = generate_test_splats(100, seed=42)
        engine = HRM2Engine(n_coarse=5, n_fine=10)
        engine.add_splats(splats)
        engine.index()
        engine.clear()

        assert len(engine.splats) == 0
        assert not engine._is_indexed


class TestGPUAcceleration:
    """Test GPU acceleration module."""

    def test_device_detection(self):
        """Test device detection."""
        from src.gpu import detect_device, HAS_CUDA

        device = detect_device()
        assert device in ("cuda", "cpu")
        if HAS_CUDA:
            assert device == "cuda"

    def test_gpu_info(self):
        """Test GPU info retrieval."""
        from src.gpu import get_gpu_info

        info = get_gpu_info()
        assert "device" in info
        if info["device"] == "cuda":
            assert "name" in info
            assert "vram_total_gb" in info

    def test_gpu_searcher_recall(self):
        """Test that GPU search matches CPU brute-force results."""
        from src.gpu.gpu_search import GPUSearcher

        np.random.seed(42)
        N, D = 2000, 128
        data = np.random.randn(N, D).astype(np.float32)
        searcher = GPUSearcher(data)
        assert searcher.device in ("cuda", "cpu")

        query = data[100]
        gpu_indices, gpu_dists = searcher.search(query, k=10)

        # CPU brute-force ground truth
        cpu_dists = np.linalg.norm(data - query, axis=1)
        cpu_indices = np.argsort(cpu_dists)[:10]

        # Recall should be 100% for brute-force
        recall = len(set(int(i) for i in gpu_indices) & set(int(i) for i in cpu_indices)) / 10
        assert recall >= 0.9, f"Recall {recall} < 0.9"

    def test_gpu_searcher_batch(self):
        """Test batch search."""
        from src.gpu.gpu_search import GPUSearcher

        np.random.seed(42)
        N, D, B = 1000, 64, 10
        data = np.random.randn(N, D).astype(np.float32)
        searcher = GPUSearcher(data)

        queries = data[:B]
        indices, distances = searcher.batch_search(queries, k=5)

        assert indices.shape == (B, 5)
        assert distances.shape == (B, 5)
        assert np.all(distances >= 0)

    def test_gpu_kmeans(self):
        """Test GPU K-Means produces valid clusters."""
        from src.gpu.gpu_kmeans import TorchKMeans

        np.random.seed(42)
        blob1 = np.random.randn(100, 8).astype(np.float32)
        blob2 = np.random.randn(100, 8).astype(np.float32) + 10
        data = np.vstack([blob1, blob2])

        kmeans = TorchKMeans(n_clusters=2, random_state=42, max_iter=20)
        kmeans.fit(data)

        assert kmeans.centroids_.shape == (2, 8)
        assert kmeans.labels_.shape == (200,)

    def test_engine_gpu_mode(self):
        """Test that engine uses GPU when available."""
        from src.core.hrm2_engine import HRM2Engine, generate_test_splats
        from src.gpu import HAS_CUDA

        splats = generate_test_splats(500, seed=42)
        engine = HRM2Engine(n_coarse=5, n_fine=10, use_gpu=True)
        engine.add_splats(splats)
        engine.index()

        results = engine.query(engine.embeddings[0], k=5)
        assert len(results) == 5

        stats = engine.get_stats()
        if HAS_CUDA:
            assert stats.device == "cuda"

    def test_engine_cpu_fallback(self):
        """Test that engine works with use_gpu=False."""
        from src.core.hrm2_engine import HRM2Engine, generate_test_splats

        splats = generate_test_splats(500, seed=42)
        engine = HRM2Engine(n_coarse=5, n_fine=10, use_gpu=False)
        engine.add_splats(splats)
        engine.index()

        results = engine.query(engine.embeddings[0], k=5)
        assert len(results) == 5
        assert engine.get_stats().device == "cpu"


class TestGenerateTestSplats:
    """Test the test data generator."""

    def test_generate_does_not_pollute_global_seed(self):
        """generate_test_splats must not modify global np.random state."""
        from src.core.hrm2_engine import generate_test_splats

        np.random.seed(999)
        state_before = np.random.get_state()

        _ = generate_test_splats(100, seed=42)

        state_after = np.random.get_state()
        # The internal state arrays must be identical
        np.testing.assert_array_equal(state_before[1], state_after[1])

    def test_generate_reproducible(self):
        """Same seed produces same splats."""
        from src.core.hrm2_engine import generate_test_splats

        s1 = generate_test_splats(50, seed=42)
        s2 = generate_test_splats(50, seed=42)
        np.testing.assert_array_equal(s1[0].position, s2[0].position)


class TestMemoryManager:
    """Test memory manager."""

    def test_memory_manager_creation(self):
        """Test memory manager creation."""
        from src.memory.manager import SplatMemoryManager

        manager = SplatMemoryManager(vram_limit=1000, ram_limit=5000)

        assert manager.vram_limit == 1000
        assert manager.ram_limit == 5000

    def test_add_splats(self):
        """Test adding splats."""
        from src.memory.manager import SplatMemoryManager
        from src.core.splat_types import GaussianSplat

        splats = [GaussianSplat(id=i) for i in range(100)]

        manager = SplatMemoryManager()
        manager.add_splats(splats)

        stats = manager.get_stats()
        assert stats.total_splats == 100

    def test_get_splat(self):
        """Test getting splat."""
        from src.memory.manager import SplatMemoryManager
        from src.core.splat_types import GaussianSplat

        splats = [GaussianSplat(id=i) for i in range(100)]

        manager = SplatMemoryManager()
        manager.add_splats(splats)

        splat = manager.get_splat(0)
        assert splat is not None
        assert splat.id == 0

    def test_lru_eviction(self):
        """Test LRU eviction from VRAM."""
        from src.memory.manager import SplatMemoryManager
        from src.core.splat_types import GaussianSplat

        manager = SplatMemoryManager(vram_limit=5, ram_limit=100, eviction_threshold=1.0)
        manager.add_splats([GaussianSplat(id=i) for i in range(100)], to_cold=False)

        # Access ids 0-4 to promote them to VRAM
        for i in range(5):
            for _ in range(manager.access_threshold + 1):
                manager.get_splat(i)

        assert manager.vram_size == 5

        # Access id 5 to trigger eviction
        for _ in range(manager.access_threshold + 1):
            manager.get_splat(5)

        assert manager.vram_size <= 5
        assert manager.get_stats().evictions > 0

    def test_thread_safety(self):
        """Test concurrent access doesn't crash."""
        import threading
        from src.memory.manager import SplatMemoryManager
        from src.core.splat_types import GaussianSplat

        manager = SplatMemoryManager()
        manager.add_splats([GaussianSplat(id=i) for i in range(1000)])

        results = []
        errors = []

        def worker():
            try:
                for i in range(100):
                    s = manager.get_splat(i % 1000)
                    if s:
                        results.append(s.id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
