"""
smoothing_processor.py の最適化テスト

バッチクエリと従来のループ版の結果が一致することを確認
"""

import numpy as np
import sys
import os
import time

# smoothing_processor.py をインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    '..', 'MochFitter-unity-addon', 'OutfitRetargetingSystem', 'Editor'))

from smoothing_processor import (
    apply_smoothing_sequential,
    query_neighbors_batched,
    blend_weights_vectorized,
    BATCH_QUERY_CHUNK_SIZE
)
from scipy.spatial import cKDTree


def test_query_neighbors_batched():
    """query_neighbors_batched が正しく動作することを確認"""
    print("=" * 60)
    print("Test 1: query_neighbors_batched")
    print("=" * 60)

    np.random.seed(42)
    num_vertices = 5000
    vertex_coords = np.random.rand(num_vertices, 3).astype(np.float32)
    radius = 0.05

    kdtree = cKDTree(vertex_coords)

    # 従来方式
    loop_start = time.perf_counter()
    loop_neighbors = []
    for i in range(num_vertices):
        neighbors = kdtree.query_ball_point(vertex_coords[i], radius)
        loop_neighbors.append(neighbors)
    loop_time = time.perf_counter() - loop_start

    # バッチ方式
    batch_start = time.perf_counter()
    batch_neighbors = query_neighbors_batched(vertex_coords, kdtree, radius)
    batch_time = time.perf_counter() - batch_start

    # 結果を比較
    match = True
    for i in range(num_vertices):
        if set(loop_neighbors[i]) != set(batch_neighbors[i]):
            print(f"  Mismatch at vertex {i}")
            match = False
            break

    print(f"  頂点数: {num_vertices}")
    print(f"  ループ版: {loop_time*1000:.2f} ms")
    print(f"  バッチ版: {batch_time*1000:.2f} ms")
    print(f"  高速化: {loop_time/batch_time:.2f}x")
    print(f"  結果一致: {'OK' if match else 'NG'}")

    return match


def test_apply_smoothing_sequential():
    """apply_smoothing_sequential のバッチ/非バッチ結果が一致することを確認"""
    print("\n" + "=" * 60)
    print("Test 2: apply_smoothing_sequential (batch vs loop)")
    print("=" * 60)

    np.random.seed(42)
    num_vertices = 3000
    vertex_coords = np.random.rand(num_vertices, 3).astype(np.float32)
    weights = np.random.rand(num_vertices).astype(np.float32)
    radius = 0.05

    kdtree = cKDTree(vertex_coords)

    # 従来方式 (use_batch_query=False)
    loop_start = time.perf_counter()
    loop_result = apply_smoothing_sequential(
        vertex_coords, weights, kdtree, radius,
        use_distance_weighting=True, gaussian_falloff=True,
        use_batch_query=False
    )
    loop_time = time.perf_counter() - loop_start

    # バッチ方式 (use_batch_query=True)
    batch_start = time.perf_counter()
    batch_result = apply_smoothing_sequential(
        vertex_coords, weights, kdtree, radius,
        use_distance_weighting=True, gaussian_falloff=True,
        use_batch_query=True
    )
    batch_time = time.perf_counter() - batch_start

    # 結果を比較
    max_diff = np.max(np.abs(loop_result - batch_result))
    match = max_diff < 1e-6

    print(f"  頂点数: {num_vertices}")
    print(f"  ループ版: {loop_time*1000:.2f} ms")
    print(f"  バッチ版: {batch_time*1000:.2f} ms")
    print(f"  高速化: {loop_time/batch_time:.2f}x")
    print(f"  最大差分: {max_diff:.2e}")
    print(f"  結果一致: {'OK' if match else 'NG'}")

    return match


def test_blend_weights_vectorized():
    """blend_weights_vectorized が従来のループと同じ結果を返すことを確認"""
    print("\n" + "=" * 60)
    print("Test 3: blend_weights_vectorized")
    print("=" * 60)

    np.random.seed(42)
    num_vertices = 10000
    original = np.random.rand(num_vertices).astype(np.float32)
    smoothed = np.random.rand(num_vertices).astype(np.float32)
    mask = np.random.rand(num_vertices).astype(np.float32)

    # 従来方式
    loop_start = time.perf_counter()
    loop_result = np.zeros(num_vertices, dtype=np.float32)
    for i in range(num_vertices):
        blend_factor = mask[i]
        loop_result[i] = original[i] * (1.0 - blend_factor) + smoothed[i] * blend_factor
    loop_time = time.perf_counter() - loop_start

    # ベクトル化版
    vec_start = time.perf_counter()
    vec_result = blend_weights_vectorized(original, smoothed, mask)
    vec_time = time.perf_counter() - vec_start

    # 結果を比較
    max_diff = np.max(np.abs(loop_result - vec_result))
    match = max_diff < 1e-6

    print(f"  頂点数: {num_vertices}")
    print(f"  ループ版: {loop_time*1000:.4f} ms")
    print(f"  ベクトル版: {vec_time*1000:.4f} ms")
    print(f"  高速化: {loop_time/vec_time:.0f}x")
    print(f"  最大差分: {max_diff:.2e}")
    print(f"  結果一致: {'OK' if match else 'NG'}")

    return match


def run_all_tests():
    """すべてのテストを実行"""
    print("\n" + "=" * 60)
    print("smoothing_processor.py 最適化テスト")
    print("=" * 60 + "\n")

    results = []
    results.append(("query_neighbors_batched", test_query_neighbors_batched()))
    results.append(("apply_smoothing_sequential", test_apply_smoothing_sequential()))
    results.append(("blend_weights_vectorized", test_blend_weights_vectorized()))

    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("すべてのテストが成功しました！")
    else:
        print("一部のテストが失敗しました。")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
