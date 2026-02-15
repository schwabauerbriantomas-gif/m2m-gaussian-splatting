#!/usr/bin/env python
"""
Quick Demo - M2M Gaussian Splatting

Demonstrates basic usage of the M2M system:
1. Creating Gaussian splats
2. Building embeddings
3. Indexing with HRM2
4. Querying for similar splats
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from time import time

from src.core.splat_types import GaussianSplat
from src.core.encoding import FullEmbeddingBuilder
from src.core.hrm2_engine import HRM2Engine, generate_test_splats
from src.memory.manager import SplatMemoryManager


def main():
    print("=" * 60)
    print("M2M GAUSSIAN SPLATTING - QUICK DEMO")
    print("=" * 60)
    print()
    
    # Configuration
    N_SPLATS = 10000
    N_COARSE = 50
    N_FINE = 200
    K_RESULTS = 10
    
    print(f"Configuration:")
    print(f"  Splats: {N_SPLATS:,}")
    print(f"  Coarse clusters: {N_COARSE}")
    print(f"  Fine clusters: {N_FINE}")
    print()
    
    # Step 1: Generate test splats
    print("Step 1: Generating test splats...")
    start = time()
    splats = generate_test_splats(N_SPLATS, seed=42)
    print(f"  Generated {len(splats):,} splats in {time()-start:.2f}s")
    print()
    
    # Step 2: Build embeddings
    print("Step 2: Building embeddings...")
    start = time()
    
    positions = np.array([s.position for s in splats])
    colors = np.array([s.color for s in splats])
    opacities = np.array([s.opacity for s in splats])
    scales = np.array([s.scale for s in splats])
    rotations = np.array([s.rotation for s in splats])
    
    encoder = FullEmbeddingBuilder()
    embeddings = encoder.build(positions, colors, opacities, scales, rotations)
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Build time: {time()-start:.2f}s")
    print()
    
    # Step 3: Index with HRM2
    print("Step 3: Building HRM2 index...")
    start = time()
    
    engine = HRM2Engine(
        n_coarse=N_COARSE,
        n_fine=N_FINE,
        n_probe=3
    )
    engine.add_splats(splats)
    build_time = engine.index()
    
    print(f"  Index built in {build_time:.2f}s")
    
    stats = engine.get_stats()
    print(f"  Coarse clusters: {stats.n_coarse_clusters}")
    print(f"  Fine clusters: {stats.n_fine_clusters}")
    print()
    
    # Step 4: Query for similar splats
    print(f"Step 4: Querying for top {K_RESULTS} similar splats...")
    
    # Use first splat as query
    query_idx = 0
    query_embedding = embeddings[query_idx]
    
    start = time()
    results = engine.query(query_embedding, k=K_RESULTS)
    query_time = (time() - start) * 1000
    
    print(f"  Query time: {query_time:.2f}ms")
    print(f"  Results:")
    
    for i, (splat, dist) in enumerate(results):
        print(f"    {i+1}. Splat {splat.id}: distance={dist:.4f}")
    print()
    
    # Step 5: Memory management demo
    print("Step 5: Memory management demo...")
    
    memory = SplatMemoryManager(vram_limit=1000, ram_limit=5000)
    memory.add_splats(splats[:10000])
    
    print(f"  Added {len(splats):,} splats")
    
    # Access some splats
    for i in range(100):
        memory.get_splat(i)
    
    mem_stats = memory.get_stats()
    print(f"  Cache hits: {mem_stats.cache_hits}")
    print(f"  VRAM: {mem_stats.vram_usage:,}")
    print(f"  RAM: {mem_stats.ram_usage:,}")
    print()
    
    # Summary
    print("=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Key features demonstrated:")
    print("  ✓ Gaussian splat creation")
    print("  ✓ 640D embedding generation")
    print("  ✓ HRM2 hierarchical indexing")
    print("  ✓ Fast similarity search")
    print("  ✓ Memory management")


if __name__ == "__main__":
    main()
