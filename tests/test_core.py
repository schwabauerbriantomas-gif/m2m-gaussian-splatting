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
            opacity=0.9
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
            attribute_encoding=np.zeros(64)
        )
        
        full = embedding.full_embedding
        assert full.shape == (640,)


class TestEncoding:
    """Test encoding functions."""
    
    def test_position_encoding(self):
        """Test sinusoidal position encoding."""
        from src.core.encoding import SinusoidalPositionEncoder
        
        encoder = SinusoidalPositionEncoder(dim=64)
        positions = np.random.randn(100, 3).astype(np.float32)
        
        encodings = encoder.encode(positions)
        
        assert encodings.shape == (100, 64)
    
    def test_color_encoding(self):
        """Test color histogram encoding."""
        from src.core.encoding import ColorHistogramEncoder
        
        encoder = ColorHistogramEncoder(n_bins=8)
        colors = np.random.rand(100, 3).astype(np.float32)
        
        encodings = encoder.encode(colors)
        
        assert encodings.shape == (100, 512)
    
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


class TestClustering:
    """Test clustering functions."""
    
    def test_kmeans_jit(self):
        """Test K-Means JIT implementation."""
        from src.core.clustering import KMeansJIT
        
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(1000, 64).astype(np.float32)
        
        # Fit
        kmeans = KMeansJIT(n_clusters=10, random_state=42)
        kmeans.fit(data)
        
        assert kmeans.centroids_.shape == (10, 64)
        assert kmeans.labels_.shape == (1000,)
        
        # Predict
        labels = kmeans.predict(data)
        assert labels.shape == (1000,)


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
        
        # Use first splat embedding as query
        query = engine.embeddings[0]
        results = engine.query(query, k=10)
        
        assert len(results) == 10
        # First result should be the query splat
        assert results[0][0].id == splats[0].id


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
