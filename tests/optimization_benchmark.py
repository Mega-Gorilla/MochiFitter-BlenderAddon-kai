#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最適化手法のベンチマークテスト

実際のメッシュデータを使用して、各最適化手法の効果を検証する。
- P1-1: Numba JIT 距離計算
- P1-2: スムージング処理のベクトル化（query_ball_tree）
- P1-3: KDTree 共有による再構築回避

使用方法:
    python tests/optimization_benchmark.py
    python tests/optimization_benchmark.py --num-vertices 10000
    python tests/optimization_benchmark.py --with-numba
"""

import numpy as np
import time
import argparse
from pathlib import Path
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

# scipy は必須
from scipy.spatial import cKDTree

# Numba はオプション
NUMBA_AVAILABLE = False
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""
    name: str
    duration_ms: float
    mse: Optional[float] = None  # baseline との比較
    max_error: Optional[float] = None
    speedup: Optional[float] = None  # baseline 比


# =============================================================================
# データ読み込み
# =============================================================================

def load_mesh_data(npz_path: str, num_vertices: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    NPZ ファイルから頂点データを読み込む

    Returns:
        vertex_coords: 頂点座標 (N, 3)
        weights: 疑似ウェイト値 (N,)
    """
    npz = np.load(npz_path, allow_pickle=True)
    all_points = npz['all_field_points']

    # 最初のステップの頂点データを使用
    full_coords = all_points[0].astype(np.float32)

    # サブセットを取得（ランダムサンプリング）
    if num_vertices < len(full_coords):
        np.random.seed(42)  # 再現性のため固定シード
        indices = np.random.choice(len(full_coords), num_vertices, replace=False)
        vertex_coords = full_coords[indices]
    else:
        vertex_coords = full_coords
        num_vertices = len(full_coords)

    # 疑似ウェイト値を生成（座標から）
    weights = np.linalg.norm(vertex_coords, axis=1).astype(np.float32)
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    print(f"Loaded {len(vertex_coords)} vertices from {Path(npz_path).name}")
    return vertex_coords, weights


# =============================================================================
# Baseline 実装（現在の smoothing_processor.py）
# =============================================================================

def gaussian_weights(distances: np.ndarray, sigma: float) -> np.ndarray:
    """ガウシアン減衰による重み計算"""
    return np.exp(-(distances ** 2) / (2 * sigma ** 2))


def smoothing_baseline(
    vertex_coords: np.ndarray,
    weights: np.ndarray,
    smoothing_radius: float = 0.05,
    use_distance_weighting: bool = True,
    gaussian_falloff: bool = True
) -> np.ndarray:
    """
    Baseline 実装: 現在の smoothing_processor.py と同等
    - Python ループで頂点を逐次処理
    - 各頂点で np.linalg.norm() を呼び出し
    """
    num_vertices = len(vertex_coords)
    sigma = smoothing_radius / 3.0
    smoothed_weights = np.zeros(num_vertices, dtype=np.float32)

    # KDTree 構築
    kdtree = cKDTree(vertex_coords)

    for i in range(num_vertices):
        neighbor_indices = kdtree.query_ball_point(vertex_coords[i], smoothing_radius)

        if len(neighbor_indices) > 1:
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
            neighbor_coords = vertex_coords[neighbor_indices]
            # ★ ボトルネック: 各頂点で距離計算
            distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
            neighbor_weights = weights[neighbor_indices]

            if use_distance_weighting and gaussian_falloff:
                w = gaussian_weights(distances, sigma)
                weights_sum = np.sum(w)
                if weights_sum > 0.001:
                    smoothed_weights[i] = neighbor_weights @ w / weights_sum
                else:
                    smoothed_weights[i] = weights[i]
            else:
                smoothed_weights[i] = np.mean(neighbor_weights)
        else:
            smoothed_weights[i] = weights[i]

    return smoothed_weights


# =============================================================================
# P1-1: Numba JIT 距離計算
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, fastmath=True)
    def compute_distances_numba(center: np.ndarray, neighbor_coords: np.ndarray) -> np.ndarray:
        """Numba JIT版: 1頂点から近傍点群への距離計算"""
        n = len(neighbor_coords)
        distances = np.empty(n, dtype=np.float32)

        for j in range(n):
            dist = 0.0
            for k in range(3):
                d = center[k] - neighbor_coords[j, k]
                dist += d * d
            distances[j] = np.sqrt(dist)

        return distances

    @jit(nopython=True, fastmath=True)
    def gaussian_weights_numba(distances: np.ndarray, sigma: float) -> np.ndarray:
        """Numba JIT版: ガウシアン重み計算"""
        return np.exp(-(distances ** 2) / (2 * sigma ** 2))


def smoothing_p1_1_numba(
    vertex_coords: np.ndarray,
    weights: np.ndarray,
    smoothing_radius: float = 0.05,
    use_distance_weighting: bool = True,
    gaussian_falloff: bool = True
) -> np.ndarray:
    """
    P1-1: Numba JIT による距離計算高速化
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available")

    num_vertices = len(vertex_coords)
    sigma = smoothing_radius / 3.0
    smoothed_weights = np.zeros(num_vertices, dtype=np.float32)

    kdtree = cKDTree(vertex_coords)

    for i in range(num_vertices):
        neighbor_indices = kdtree.query_ball_point(vertex_coords[i], smoothing_radius)

        if len(neighbor_indices) > 1:
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
            neighbor_coords = vertex_coords[neighbor_indices]
            # ★ Numba JIT による距離計算
            distances = compute_distances_numba(vertex_coords[i], neighbor_coords)
            neighbor_weights = weights[neighbor_indices]

            if use_distance_weighting and gaussian_falloff:
                w = gaussian_weights_numba(distances, sigma)
                weights_sum = np.sum(w)
                if weights_sum > 0.001:
                    smoothed_weights[i] = neighbor_weights @ w / weights_sum
                else:
                    smoothed_weights[i] = weights[i]
            else:
                smoothed_weights[i] = np.mean(neighbor_weights)
        else:
            smoothed_weights[i] = weights[i]

    return smoothed_weights


# =============================================================================
# P1-2: ベクトル化（query_ball_tree による一括近傍取得）
# =============================================================================

def smoothing_p1_2_vectorized(
    vertex_coords: np.ndarray,
    weights: np.ndarray,
    smoothing_radius: float = 0.05,
    use_distance_weighting: bool = True,
    gaussian_falloff: bool = True
) -> np.ndarray:
    """
    P1-2: query_ball_tree による一括近傍取得
    - 全頂点の近傍を一括取得
    - ループは残るが、近傍検索のオーバーヘッドを削減
    """
    num_vertices = len(vertex_coords)
    sigma = smoothing_radius / 3.0
    smoothed_weights = np.zeros(num_vertices, dtype=np.float32)

    kdtree = cKDTree(vertex_coords)

    # ★ 全頂点の近傍を一括取得
    all_neighbors = kdtree.query_ball_tree(kdtree, smoothing_radius)

    for i in range(num_vertices):
        neighbor_indices = all_neighbors[i]

        if len(neighbor_indices) > 1:
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
            neighbor_coords = vertex_coords[neighbor_indices]
            distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
            neighbor_weights = weights[neighbor_indices]

            if use_distance_weighting and gaussian_falloff:
                w = gaussian_weights(distances, sigma)
                weights_sum = np.sum(w)
                if weights_sum > 0.001:
                    smoothed_weights[i] = neighbor_weights @ w / weights_sum
                else:
                    smoothed_weights[i] = weights[i]
            else:
                smoothed_weights[i] = np.mean(neighbor_weights)
        else:
            smoothed_weights[i] = weights[i]

    return smoothed_weights


# =============================================================================
# P1-3: KDTree 共有（バッチ処理でKDTree再構築を回避）
# =============================================================================

def smoothing_p1_3_kdtree_shared(
    vertex_coords: np.ndarray,
    weights: np.ndarray,
    smoothing_radius: float = 0.05,
    use_distance_weighting: bool = True,
    gaussian_falloff: bool = True,
    num_iterations: int = 3
) -> np.ndarray:
    """
    P1-3: KDTree 共有による再構築回避
    - 複数イテレーションでKDTreeを再利用
    - 実際の smoothing_processor.py では各バッチでKDTreeを再構築している
    """
    num_vertices = len(vertex_coords)
    sigma = smoothing_radius / 3.0
    current_weights = weights.copy()

    # ★ KDTree は1回だけ構築（イテレーション間で共有）
    kdtree = cKDTree(vertex_coords)
    all_neighbors = kdtree.query_ball_tree(kdtree, smoothing_radius)

    for iteration in range(num_iterations):
        smoothed_weights = np.zeros(num_vertices, dtype=np.float32)

        for i in range(num_vertices):
            neighbor_indices = all_neighbors[i]

            if len(neighbor_indices) > 1:
                neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
                neighbor_coords = vertex_coords[neighbor_indices]
                distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
                neighbor_weights = current_weights[neighbor_indices]

                if use_distance_weighting and gaussian_falloff:
                    w = gaussian_weights(distances, sigma)
                    weights_sum = np.sum(w)
                    if weights_sum > 0.001:
                        smoothed_weights[i] = neighbor_weights @ w / weights_sum
                    else:
                        smoothed_weights[i] = current_weights[i]
                else:
                    smoothed_weights[i] = np.mean(neighbor_weights)
            else:
                smoothed_weights[i] = current_weights[i]

        current_weights = smoothed_weights

    return smoothed_weights


def smoothing_p1_3_kdtree_rebuild(
    vertex_coords: np.ndarray,
    weights: np.ndarray,
    smoothing_radius: float = 0.05,
    use_distance_weighting: bool = True,
    gaussian_falloff: bool = True,
    num_iterations: int = 3
) -> np.ndarray:
    """
    比較用: 各イテレーションでKDTreeを再構築
    """
    num_vertices = len(vertex_coords)
    sigma = smoothing_radius / 3.0
    current_weights = weights.copy()

    for iteration in range(num_iterations):
        # ★ 各イテレーションでKDTreeを再構築（非効率）
        kdtree = cKDTree(vertex_coords)
        smoothed_weights = np.zeros(num_vertices, dtype=np.float32)

        for i in range(num_vertices):
            neighbor_indices = kdtree.query_ball_point(vertex_coords[i], smoothing_radius)

            if len(neighbor_indices) > 1:
                neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
                neighbor_coords = vertex_coords[neighbor_indices]
                distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
                neighbor_weights = current_weights[neighbor_indices]

                if use_distance_weighting and gaussian_falloff:
                    w = gaussian_weights(distances, sigma)
                    weights_sum = np.sum(w)
                    if weights_sum > 0.001:
                        smoothed_weights[i] = neighbor_weights @ w / weights_sum
                    else:
                        smoothed_weights[i] = current_weights[i]
                else:
                    smoothed_weights[i] = np.mean(neighbor_weights)
            else:
                smoothed_weights[i] = current_weights[i]

        current_weights = smoothed_weights

    return smoothed_weights


# =============================================================================
# ベンチマーク実行
# =============================================================================

def run_benchmark(
    name: str,
    func: Callable,
    vertex_coords: np.ndarray,
    weights: np.ndarray,
    baseline_result: Optional[np.ndarray] = None,
    baseline_time: Optional[float] = None,
    warmup: bool = False,
    **kwargs
) -> BenchmarkResult:
    """ベンチマークを実行"""

    # ウォームアップ（JITコンパイル用）
    if warmup:
        try:
            _ = func(vertex_coords[:100], weights[:100], **kwargs)
        except Exception:
            pass

    # 本番実行
    start = time.perf_counter()
    result = func(vertex_coords, weights, **kwargs)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # baseline との比較
    mse = None
    max_error = None
    speedup = None

    if baseline_result is not None:
        diff = result - baseline_result
        mse = float(np.mean(diff ** 2))
        max_error = float(np.max(np.abs(diff)))

    if baseline_time is not None:
        speedup = baseline_time / duration_ms

    return BenchmarkResult(
        name=name,
        duration_ms=duration_ms,
        mse=mse,
        max_error=max_error,
        speedup=speedup
    )


def print_results(results: list[BenchmarkResult]):
    """結果を表示"""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\n{'Method':<40} {'Time (ms)':>12} {'Speedup':>10} {'MSE':>15} {'Max Error':>12}")
    print("-" * 80)

    for r in results:
        speedup_str = f"{r.speedup:.2f}x" if r.speedup else "-"
        mse_str = f"{r.mse:.2e}" if r.mse is not None else "-"
        max_err_str = f"{r.max_error:.2e}" if r.max_error is not None else "-"
        print(f"{r.name:<40} {r.duration_ms:>12.2f} {speedup_str:>10} {mse_str:>15} {max_err_str:>12}")

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Optimization Benchmark")
    parser.add_argument("--num-vertices", type=int, default=10000,
                        help="Number of vertices to use (default: 10000)")
    parser.add_argument("--smoothing-radius", type=float, default=0.05,
                        help="Smoothing radius (default: 0.05)")
    parser.add_argument("--with-numba", action="store_true",
                        help="Include Numba JIT benchmark")
    parser.add_argument("--npz-path", type=str, default=None,
                        help="Path to NPZ file with mesh data")
    args = parser.parse_args()

    # NPZ ファイルパス
    if args.npz_path:
        npz_path = args.npz_path
    else:
        # デフォルトパス
        repo_root = Path(__file__).parent.parent
        npz_path = repo_root / "MochFitter-unity-addon" / "OutfitRetargetingSystem" / "Editor" / "deformation_beryl_to_template.npz"

    if not Path(npz_path).exists():
        print(f"ERROR: NPZ file not found: {npz_path}")
        print("Please specify --npz-path or ensure the default file exists.")
        return

    # データ読み込み
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    vertex_coords, weights = load_mesh_data(str(npz_path), args.num_vertices)
    print(f"Smoothing radius: {args.smoothing_radius}")
    print(f"Numba available: {NUMBA_AVAILABLE}")

    results = []

    # ==========================================================================
    # Baseline
    # ==========================================================================
    print(f"\n{'='*80}")
    print("RUNNING BENCHMARKS")
    print(f"{'='*80}")

    print("\n[1/5] Running baseline...")
    baseline = run_benchmark(
        "Baseline (current implementation)",
        smoothing_baseline,
        vertex_coords, weights,
        smoothing_radius=args.smoothing_radius
    )
    results.append(baseline)
    baseline_result = smoothing_baseline(vertex_coords, weights, args.smoothing_radius)

    # ==========================================================================
    # P1-1: Numba JIT
    # ==========================================================================
    if args.with_numba and NUMBA_AVAILABLE:
        print("\n[2/5] Running P1-1 (Numba JIT)...")
        # ウォームアップでJITコンパイル
        p1_1 = run_benchmark(
            "P1-1: Numba JIT distance calculation",
            smoothing_p1_1_numba,
            vertex_coords, weights,
            baseline_result=baseline_result,
            baseline_time=baseline.duration_ms,
            warmup=True,
            smoothing_radius=args.smoothing_radius
        )
        results.append(p1_1)
    elif args.with_numba:
        print("\n[2/5] Skipping P1-1 (Numba not installed)")
    else:
        print("\n[2/5] Skipping P1-1 (use --with-numba to enable)")

    # ==========================================================================
    # P1-2: Vectorized (query_ball_tree)
    # ==========================================================================
    print("\n[3/5] Running P1-2 (query_ball_tree)...")
    p1_2 = run_benchmark(
        "P1-2: Vectorized (query_ball_tree)",
        smoothing_p1_2_vectorized,
        vertex_coords, weights,
        baseline_result=baseline_result,
        baseline_time=baseline.duration_ms,
        smoothing_radius=args.smoothing_radius
    )
    results.append(p1_2)

    # ==========================================================================
    # P1-3: KDTree shared vs rebuild
    # ==========================================================================
    print("\n[4/5] Running P1-3 (KDTree rebuild - 3 iterations)...")
    p1_3_rebuild = run_benchmark(
        "P1-3: KDTree rebuild each iteration (3x)",
        smoothing_p1_3_kdtree_rebuild,
        vertex_coords, weights,
        baseline_time=baseline.duration_ms,
        smoothing_radius=args.smoothing_radius,
        num_iterations=3
    )
    results.append(p1_3_rebuild)

    print("\n[5/5] Running P1-3 (KDTree shared - 3 iterations)...")
    p1_3_shared = run_benchmark(
        "P1-3: KDTree shared across iterations (3x)",
        smoothing_p1_3_kdtree_shared,
        vertex_coords, weights,
        baseline_time=baseline.duration_ms,
        smoothing_radius=args.smoothing_radius,
        num_iterations=3
    )
    results.append(p1_3_shared)

    # ==========================================================================
    # 結果表示
    # ==========================================================================
    print_results(results)

    # ==========================================================================
    # サマリー
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Vertices: {len(vertex_coords)}")
    print(f"Baseline time: {baseline.duration_ms:.2f} ms")

    if args.with_numba and NUMBA_AVAILABLE:
        print(f"\nP1-1 (Numba JIT):")
        print(f"  - Speedup: {results[1].speedup:.2f}x")
        print(f"  - MSE: {results[1].mse:.2e} (should be ~0 or very small)")

    print(f"\nP1-2 (query_ball_tree):")
    print(f"  - Speedup: {p1_2.speedup:.2f}x")
    print(f"  - MSE: {p1_2.mse:.2e} (should be 0)")

    print(f"\nP1-3 (KDTree caching for 3 iterations):")
    print(f"  - Rebuild each iteration: {p1_3_rebuild.duration_ms:.2f} ms")
    print(f"  - Shared KDTree: {p1_3_shared.duration_ms:.2f} ms")
    print(f"  - Speedup from caching: {p1_3_rebuild.duration_ms / p1_3_shared.duration_ms:.2f}x")


if __name__ == "__main__":
    main()
