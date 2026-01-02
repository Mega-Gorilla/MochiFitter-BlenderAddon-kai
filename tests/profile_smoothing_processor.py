"""
smoothing_processor.py のプロファイリングスクリプト

現在の実装と最適化案の性能を比較測定する
"""

import numpy as np
import time
import sys
import os
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict
import cProfile
import pstats
from io import StringIO

# テストパラメータ（実際の使用状況に近い値）
TEST_VERTEX_COUNT = 25000  # 典型的なメッシュの頂点数
TEST_GROUP_COUNT = 40      # 頂点グループ数
SMOOTHING_RADIUS = 0.02    # スムージング半径
ITERATION_COUNT = 3        # イテレーション回数


def generate_test_data(num_vertices: int, num_groups: int) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """テスト用のデータを生成"""
    np.random.seed(42)

    # 頂点座標（単位立方体内にランダム分布）
    vertex_coords = np.random.rand(num_vertices, 3).astype(np.float32)

    # 頂点グループのウェイト（各グループにランダムなウェイト）
    group_weights = []
    for _ in range(num_groups):
        weights = np.random.rand(num_vertices).astype(np.float32)
        # 約50%の頂点をゼロに
        weights[np.random.rand(num_vertices) < 0.5] = 0
        group_weights.append(weights)

    # マスクウェイト
    mask_weights = np.random.rand(num_vertices).astype(np.float32)

    return vertex_coords, group_weights, mask_weights


# ============================================================
# 現在の実装（smoothing_processor.py から抽出）
# ============================================================

def gaussian_weights(distances, sigma):
    """ガウシアン減衰による重み計算"""
    return np.exp(-(distances ** 2) / (2 * sigma ** 2))


def current_apply_smoothing_sequential(vertex_coords, current_weights, kdtree,
                                        smoothing_radius, sigma):
    """現在の実装: 頂点ごとのループ"""
    num_vertices = len(vertex_coords)
    smoothed_weights = np.zeros(num_vertices, dtype=np.float32)

    for i in range(num_vertices):
        neighbor_indices = kdtree.query_ball_point(vertex_coords[i], smoothing_radius)

        if len(neighbor_indices) > 1:
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
            neighbor_coords = vertex_coords[neighbor_indices]
            distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
            neighbor_weights = current_weights[neighbor_indices]

            weights = gaussian_weights(distances, sigma)
            weights_sum = np.sum(weights)
            if weights_sum > 0.001:
                smoothed_weights[i] = neighbor_weights @ weights / weights_sum
            else:
                smoothed_weights[i] = current_weights[i]
        else:
            smoothed_weights[i] = current_weights[i]

    return smoothed_weights


def current_mask_blending(original_weights, smoothed_weights, mask_weights):
    """現在の実装: Python for ループによるマスク合成"""
    num_vertices = len(original_weights)
    final_weights = np.zeros(num_vertices, dtype=np.float32)

    for i in range(num_vertices):
        blend_factor = mask_weights[i]
        final_weights[i] = original_weights[i] * (1.0 - blend_factor) + smoothed_weights[i] * blend_factor

    return final_weights


# ============================================================
# 最適化案
# ============================================================

def optimized_apply_smoothing_batch(vertex_coords, current_weights, kdtree,
                                     smoothing_radius, sigma):
    """最適化案: query_ball_point を一括呼び出し"""
    num_vertices = len(vertex_coords)
    smoothed_weights = np.zeros(num_vertices, dtype=np.float32)

    # 全頂点の近傍を一括取得
    all_neighbors = kdtree.query_ball_point(vertex_coords, smoothing_radius)

    for i in range(num_vertices):
        neighbor_indices = all_neighbors[i]

        if len(neighbor_indices) > 1:
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
            neighbor_coords = vertex_coords[neighbor_indices]
            distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
            neighbor_weights = current_weights[neighbor_indices]

            weights = gaussian_weights(distances, sigma)
            weights_sum = np.sum(weights)
            if weights_sum > 0.001:
                smoothed_weights[i] = neighbor_weights @ weights / weights_sum
            else:
                smoothed_weights[i] = current_weights[i]
        else:
            smoothed_weights[i] = current_weights[i]

    return smoothed_weights


def optimized_mask_blending(original_weights, smoothed_weights, mask_weights):
    """最適化案: NumPy ベクトル化によるマスク合成"""
    return original_weights * (1.0 - mask_weights) + smoothed_weights * mask_weights


# ============================================================
# プロファイリング関数
# ============================================================

def profile_kdtree_construction(vertex_coords: np.ndarray, num_iterations: int = 10) -> float:
    """KDTree 構築時間を計測"""
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        kdtree = cKDTree(vertex_coords)
        times.append(time.perf_counter() - start)
    return np.mean(times)


def profile_query_ball_point_loop(vertex_coords: np.ndarray, kdtree: cKDTree,
                                   radius: float, num_iterations: int = 3) -> float:
    """頂点ごとの query_ball_point 時間を計測"""
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        for i in range(len(vertex_coords)):
            neighbors = kdtree.query_ball_point(vertex_coords[i], radius)
        times.append(time.perf_counter() - start)
    return np.mean(times)


def profile_query_ball_point_batch(vertex_coords: np.ndarray, kdtree: cKDTree,
                                    radius: float, num_iterations: int = 3) -> float:
    """一括 query_ball_point 時間を計測"""
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        all_neighbors = kdtree.query_ball_point(vertex_coords, radius)
        times.append(time.perf_counter() - start)
    return np.mean(times)


def profile_smoothing_iteration(vertex_coords: np.ndarray, weights: np.ndarray,
                                 smoothing_radius: float, use_batch: bool = False) -> Tuple[float, float]:
    """スムージング1イテレーションの時間を計測（KDTree構築含む）"""
    sigma = smoothing_radius / 3.0

    # KDTree 構築
    kdtree_start = time.perf_counter()
    kdtree = cKDTree(vertex_coords)
    kdtree_time = time.perf_counter() - kdtree_start

    # スムージング
    smoothing_start = time.perf_counter()
    if use_batch:
        result = optimized_apply_smoothing_batch(vertex_coords, weights, kdtree, smoothing_radius, sigma)
    else:
        result = current_apply_smoothing_sequential(vertex_coords, weights, kdtree, smoothing_radius, sigma)
    smoothing_time = time.perf_counter() - smoothing_start

    return kdtree_time, smoothing_time


def profile_mask_blending(original: np.ndarray, smoothed: np.ndarray,
                          mask: np.ndarray, num_iterations: int = 100) -> Tuple[float, float]:
    """マスク合成の時間を計測"""
    # 現在の実装
    times_current = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = current_mask_blending(original, smoothed, mask)
        times_current.append(time.perf_counter() - start)

    # 最適化版
    times_optimized = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = optimized_mask_blending(original, smoothed, mask)
        times_optimized.append(time.perf_counter() - start)

    return np.mean(times_current), np.mean(times_optimized)


def run_full_profile():
    """完全なプロファイリングを実行"""
    print("=" * 70)
    print("smoothing_processor.py プロファイリング")
    print("=" * 70)
    print(f"\nテストパラメータ:")
    print(f"  頂点数: {TEST_VERTEX_COUNT:,}")
    print(f"  グループ数: {TEST_GROUP_COUNT}")
    print(f"  スムージング半径: {SMOOTHING_RADIUS}")
    print(f"  イテレーション数: {ITERATION_COUNT}")

    # テストデータ生成
    print("\nテストデータ生成中...")
    vertex_coords, group_weights, mask_weights = generate_test_data(TEST_VERTEX_COUNT, TEST_GROUP_COUNT)

    # KDTree 構築
    print("\n" + "-" * 70)
    print("1. KDTree 構築時間")
    print("-" * 70)
    kdtree_time = profile_kdtree_construction(vertex_coords)
    print(f"  1回あたり: {kdtree_time*1000:.2f} ms")
    print(f"  グループ数分 ({TEST_GROUP_COUNT}回): {kdtree_time*TEST_GROUP_COUNT*1000:.2f} ms")
    print(f"  → KDTree再利用で削減可能: {kdtree_time*(TEST_GROUP_COUNT-1)*1000:.2f} ms")

    # KDTree を事前構築
    kdtree = cKDTree(vertex_coords)

    # query_ball_point 比較
    print("\n" + "-" * 70)
    print("2. query_ball_point 比較")
    print("-" * 70)
    loop_time = profile_query_ball_point_loop(vertex_coords, kdtree, SMOOTHING_RADIUS, num_iterations=1)
    batch_time = profile_query_ball_point_batch(vertex_coords, kdtree, SMOOTHING_RADIUS, num_iterations=3)
    print(f"  ループ版 (現在):  {loop_time*1000:.2f} ms")
    print(f"  バッチ版 (最適化): {batch_time*1000:.2f} ms")
    print(f"  → 改善率: {(loop_time - batch_time) / loop_time * 100:.1f}%")
    print(f"  → 削減時間: {(loop_time - batch_time)*1000:.2f} ms/イテレーション")

    # スムージング全体（1グループ、1イテレーション）
    print("\n" + "-" * 70)
    print("3. スムージング全体 (1グループ, 1イテレーション)")
    print("-" * 70)

    test_weights = group_weights[0]

    # 現在の実装
    kd_time_curr, smooth_time_curr = profile_smoothing_iteration(
        vertex_coords, test_weights, SMOOTHING_RADIUS, use_batch=False)
    total_curr = kd_time_curr + smooth_time_curr

    # 最適化版
    kd_time_opt, smooth_time_opt = profile_smoothing_iteration(
        vertex_coords, test_weights, SMOOTHING_RADIUS, use_batch=True)
    total_opt = kd_time_opt + smooth_time_opt

    print(f"  現在の実装:")
    print(f"    KDTree構築: {kd_time_curr*1000:.2f} ms")
    print(f"    スムージング: {smooth_time_curr*1000:.2f} ms")
    print(f"    合計: {total_curr*1000:.2f} ms")
    print(f"  最適化版:")
    print(f"    KDTree構築: {kd_time_opt*1000:.2f} ms")
    print(f"    スムージング: {smooth_time_opt*1000:.2f} ms")
    print(f"    合計: {total_opt*1000:.2f} ms")
    print(f"  → 改善率: {(total_curr - total_opt) / total_curr * 100:.1f}%")

    # マスク合成
    print("\n" + "-" * 70)
    print("4. マスク合成 (全グループ分)")
    print("-" * 70)

    smoothed = np.random.rand(TEST_VERTEX_COUNT).astype(np.float32)
    original = test_weights

    mask_current, mask_optimized = profile_mask_blending(original, smoothed, mask_weights)
    print(f"  現在 (for loop): {mask_current*1000:.4f} ms × {TEST_GROUP_COUNT} = {mask_current*1000*TEST_GROUP_COUNT:.2f} ms")
    print(f"  最適化 (NumPy):  {mask_optimized*1000:.4f} ms × {TEST_GROUP_COUNT} = {mask_optimized*1000*TEST_GROUP_COUNT:.2f} ms")
    print(f"  → 削減時間: {(mask_current - mask_optimized)*1000*TEST_GROUP_COUNT:.2f} ms")

    # 全体の見積もり
    print("\n" + "=" * 70)
    print("5. 全体削減見積もり (subprocess 1回あたり)")
    print("=" * 70)

    # 現在の推定時間
    # グループあたり: KDTree構築 + iteration回のスムージング
    current_per_group = kdtree_time + (smooth_time_curr * ITERATION_COUNT)
    current_total = current_per_group * TEST_GROUP_COUNT + (mask_current * TEST_GROUP_COUNT)

    # 最適化後の推定時間
    # KDTree 1回 + グループあたり iteration回のスムージング（バッチ版）
    optimized_total = kdtree_time + (smooth_time_opt * ITERATION_COUNT * TEST_GROUP_COUNT) + (mask_optimized * TEST_GROUP_COUNT)

    print(f"\n  現在の実装 (推定):")
    print(f"    KDTree構築: {kdtree_time*1000:.2f} ms × {TEST_GROUP_COUNT} = {kdtree_time*1000*TEST_GROUP_COUNT:.2f} ms")
    print(f"    スムージング: {smooth_time_curr*1000:.2f} ms × {ITERATION_COUNT} × {TEST_GROUP_COUNT} = {smooth_time_curr*1000*ITERATION_COUNT*TEST_GROUP_COUNT:.2f} ms")
    print(f"    マスク合成: {mask_current*1000*TEST_GROUP_COUNT:.2f} ms")
    print(f"    合計: {current_total*1000:.2f} ms ({current_total:.2f} 秒)")

    print(f"\n  最適化版 (推定):")
    print(f"    KDTree構築: {kdtree_time*1000:.2f} ms × 1 = {kdtree_time*1000:.2f} ms (再利用)")
    print(f"    スムージング: {smooth_time_opt*1000:.2f} ms × {ITERATION_COUNT} × {TEST_GROUP_COUNT} = {smooth_time_opt*1000*ITERATION_COUNT*TEST_GROUP_COUNT:.2f} ms")
    print(f"    マスク合成: {mask_optimized*1000*TEST_GROUP_COUNT:.2f} ms")
    print(f"    合計: {optimized_total*1000:.2f} ms ({optimized_total:.2f} 秒)")

    reduction = current_total - optimized_total
    reduction_percent = (reduction / current_total) * 100

    print(f"\n  削減見積もり:")
    print(f"    削減時間: {reduction*1000:.2f} ms ({reduction:.2f} 秒)")
    print(f"    削減率: {reduction_percent:.1f}%")

    # 実際の subprocess 呼び出し回数を考慮
    print("\n" + "=" * 70)
    print("6. 実運用での削減見積もり")
    print("=" * 70)

    # ベンチマークから: smoothing 処理は2ペアで4回呼ばれる（各メッシュで1回ずつ、2ペア）
    # 実際にはもっと多いかもしれないが、保守的に見積もる
    subprocess_calls = 4  # 保守的な見積もり

    print(f"\n  subprocess 呼び出し回数 (推定): {subprocess_calls}")
    print(f"  1回あたりの削減: {reduction:.2f} 秒")
    print(f"  合計削減時間: {reduction * subprocess_calls:.2f} 秒")

    return {
        'kdtree_time': kdtree_time,
        'loop_time': loop_time,
        'batch_time': batch_time,
        'smooth_time_curr': smooth_time_curr,
        'smooth_time_opt': smooth_time_opt,
        'mask_current': mask_current,
        'mask_optimized': mask_optimized,
        'current_total': current_total,
        'optimized_total': optimized_total,
        'reduction': reduction,
        'reduction_percent': reduction_percent
    }


if __name__ == '__main__':
    results = run_full_profile()

    print("\n" + "=" * 70)
    print("結論")
    print("=" * 70)
    print(f"""
現実的な削減見積もり:

  楽観的シナリオ: {results['reduction'] * 6:.1f} 秒 (6回の subprocess 呼び出し想定)
  現実的シナリオ: {results['reduction'] * 4:.1f} 秒 (4回の subprocess 呼び出し想定)
  保守的シナリオ: {results['reduction'] * 2:.1f} 秒 (2回の subprocess 呼び出し想定)

主な改善ポイント:
  1. query_ball_point 一括化: {(results['loop_time'] - results['batch_time'])*1000:.1f} ms/イテレーション削減
  2. KDTree 再利用: {results['kdtree_time']*1000*(TEST_GROUP_COUNT-1):.1f} ms 削減
  3. マスク合成ベクトル化: {(results['mask_current'] - results['mask_optimized'])*1000*TEST_GROUP_COUNT:.2f} ms 削減
""")
