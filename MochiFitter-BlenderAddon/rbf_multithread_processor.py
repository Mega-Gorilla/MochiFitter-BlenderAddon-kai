#!/usr/bin/env python3
"""
RBF変形の外部マルチプロセス処理スクリプト

使用方法:
python rbf_multithread_processor.py temp_rbf_data.npz

このスクリプトはBlenderから出力された一時データファイルを読み込み、
マルチプロセスでRBF補間処理を実行して結果を保存します。

必要なライブラリ:
- numpy
- scipy
- concurrent.futures (標準ライブラリ)
- psutil (メモリモニタリング用)
"""

import numpy as np
import os
import sys
import time
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from typing import Tuple, List, Dict, Any

# psutilの可用性をチェック
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    #print("警告: psutilがインストールされていません。メモリ監視機能は無効になります。")
    #print("インストールするには: pip install psutil")

def set_cpu_affinity():
    """プロセスのCPU親和性を設定して全コアを活用"""
    try:
        # 全論理プロセッサを使用
        all_cpus = list(range(psutil.cpu_count(logical=True)))
        psutil.Process().cpu_affinity(all_cpus)
        print(f"CPU親和性を設定: {len(all_cpus)}個の論理プロセッサを使用")
    except Exception as e:
        print(f"CPU親和性設定に失敗: {e}")

class MemoryMonitor:
    """メモリ使用量を監視するクラス（psutil依存）"""
    
    def __init__(self, max_memory_gb: float = None):
        if not PSUTIL_AVAILABLE:
            self.enabled = False
            self.initial_memory = 0.0
            return
        
        self.enabled = True
        self.process = psutil.Process()
        self.max_memory_bytes = max_memory_gb * 1024**3 if max_memory_gb else None
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """現在のメモリ使用量をGB単位で取得"""
        if not self.enabled:
            return 0.0
        return self.process.memory_info().rss / 1024**3
    
    def get_memory_increase(self) -> float:
        """初期状態からのメモリ増加量をGB単位で取得"""
        if not self.enabled:
            return 0.0
        return self.get_memory_usage() - self.initial_memory
    
    def is_memory_limit_exceeded(self) -> bool:
        """メモリ制限を超えているかチェック"""
        if not self.enabled or self.max_memory_bytes is None:
            return False
        return self.process.memory_info().rss > self.max_memory_bytes
    
    def get_recommended_batch_size(self, current_batch_size: int, memory_increase: float) -> int:
        """メモリ使用量に基づいて推奨バッチサイズを計算"""
        if not self.enabled:
            return current_batch_size
        
        if memory_increase > 2.0:  # 2GB以上増加した場合
            return max(1000, current_batch_size // 4)
        elif memory_increase > 1.0:  # 1GB以上増加した場合
            return max(5000, current_batch_size // 2)
        else:
            return current_batch_size


def get_optimal_worker_count(total_items: int, memory_monitor: MemoryMonitor) -> int:
    """最適なワーカー数を計算（プロセスプール用に調整）"""
    # CPUコア数を取得
    cpu_count = os.cpu_count()
    
    # psutilが利用可能な場合のみメモリベースの調整を行う
    if PSUTIL_AVAILABLE:
        # メモリ使用量に基づいて調整
        available_memory = psutil.virtual_memory().available / 1024**3  # GB単位
    else:
        # psutilが利用できない場合は保守的な値を使用
        available_memory = 8.0  # 8GBと仮定
    
    # プロセスプールでは各プロセスがメモリを独立して使用するため、より保守的に設定
    if total_items > 1000000:  # 100万頂点以上
        max_workers = min(cpu_count, 3)  # ThreadPoolよりも少なく設定
    elif total_items > 500000:  # 50万頂点以上
        max_workers = min(cpu_count, 4)
    else:
        max_workers = min(cpu_count, 6)
    
    # 利用可能メモリに基づいて調整（プロセスプール用により厳しく）
    if available_memory < 4.0:  # 4GB未満
        max_workers = min(max_workers, 1)
    elif available_memory < 8.0:  # 8GB未満
        max_workers = min(max_workers, 2)
    elif available_memory < 16.0:  # 16GB未満
        max_workers = min(max_workers, 4)
    
    return max(1, max_workers)


def multi_quadratic_biharmonic(r: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Multi-Quadratic Biharmonic RBFカーネル関数"""
    return np.sqrt(r**2 + epsilon**2)


def smooth_step(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """
    Performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
    """
    # Clamp x to the range [0, 1]
    x = np.maximum(0, np.minimum(1, (x - edge0) / (edge1 - edge0)))
    
    # Apply the smooth step formula: 3x^2 - 2x^3
    return x * x * (3 - 2 * x)


def compute_distances_batch(batch_data: Dict[str, Any]) -> Tuple[int, int, np.ndarray]:
    """
    距離計算のバッチ処理（マルチプロセス用）
    メモリ効率を改善
    """
    start_idx = batch_data['start_idx']
    end_idx = batch_data['end_idx']
    batch_targets = batch_data['batch_targets']
    
    # KDTreeの情報を取得して新しいKDTreeを構築（メモリ効率化）
    source_vertices = batch_data['source_vertices']
    kdtree = KDTree(source_vertices)
    
    # KDTreeを使用して最近接点を検索
    distances, _ = kdtree.query(batch_targets)
    
    # 使用後すぐにメモリを解放
    del kdtree
    
    return start_idx, end_idx, distances


def compute_distances_to_source_mesh(target_vertices: np.ndarray, source_vertices: np.ndarray, 
                                   batch_size: int = 5000, max_workers: int = None) -> np.ndarray:
    """
    ターゲットメッシュの各頂点からソースメッシュの最近接頂点までの距離を計算
    KDTreeと並列処理を使用して高速化、メモリ効率を改善
    
    Parameters:
    - target_vertices: ターゲット頂点配列
    - source_vertices: ソース頂点配列  
    - batch_size: バッチサイズ（デフォルト: 5000、メモリ効率化のため削減）
    - max_workers: 最大ワーカー数（Noneの場合は自動設定）
    
    Returns:
    - 距離配列
    """
    num_target = len(target_vertices)
    distances = np.zeros(num_target)
    
    # メモリモニタリングを開始
    memory_monitor = MemoryMonitor()
    
    # 最適なワーカー数を計算
    if max_workers is None:
        max_workers = get_optimal_worker_count(num_target, memory_monitor)
    
    print(f"各頂点の最近接点までの距離を並列計算中... (頂点数: {num_target:,}, ワーカー数: {max_workers})")
    
    # 小さなデータセットの場合は並列化しない
    if num_target <= batch_size:
        print("小さなデータセットのため単一処理で実行...")
        kdtree = KDTree(source_vertices)
        distances, _ = kdtree.query(target_vertices)
        print("距離計算完了")
        return distances
    
    # バッチサイズを動的に調整
    memory_increase = memory_monitor.get_memory_increase()
    if memory_increase > 0.5:  # 500MB以上増加した場合
        batch_size = memory_monitor.get_recommended_batch_size(batch_size, memory_increase)
        print(f"メモリ使用量に基づいてバッチサイズを調整: {batch_size}")
    
    # バッチデータを準備
    batch_tasks = []
    for i in range(0, num_target, batch_size):
        end_idx = min(i + batch_size, num_target)
        batch_targets = target_vertices[i:end_idx].copy()  # コピーを作成してメモリ効率化
        
        batch_data = {
            'start_idx': i,
            'end_idx': end_idx,
            'batch_targets': batch_targets,
            'source_vertices': source_vertices  # KDTreeではなくソース頂点を渡す
        }
        batch_tasks.append(batch_data)
    
    print(f"距離計算を {len(batch_tasks)} バッチでマルチプロセス処理します")
    
    # 並列処理で距離を計算
    processed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(compute_distances_batch, batch_data): batch_data for batch_data in batch_tasks}
        
        for future in as_completed(future_to_batch):
            try:
                start_idx, end_idx, batch_distances = future.result()
                distances[start_idx:end_idx] = batch_distances
                
                processed_count += (end_idx - start_idx)
                progress_percent = (processed_count / num_target) * 100
                
                # プロセスプールでの進捗表示（メモリ監視は各プロセスで独立）
                if processed_count % (batch_size * 5) == 0 or processed_count == num_target:
                    if memory_monitor.enabled:
                        current_memory = memory_monitor.get_memory_usage()
                        print(f"距離計算進捗: {processed_count:,}/{num_target:,} 頂点処理完了 ({progress_percent:.1f}%) [メインプロセスメモリ: {current_memory:.1f}GB]")
                    else:
                        print(f"距離計算進捗: {processed_count:,}/{num_target:,} 頂点処理完了 ({progress_percent:.1f}%)")
                
            except Exception as exc:
                batch_data = future_to_batch[future]
                print(f"距離計算バッチ {batch_data['start_idx']}-{batch_data['end_idx']} でエラーが発生: {exc}")
                print("スタックトレース:")
                traceback.print_exc()
                raise exc
    
    print("距離計算完了")
    return distances


def falloff_displacements(target_vertices: np.ndarray, target_displacements: np.ndarray, 
                         source_vertices: np.ndarray, max_workers: int = None) -> List[np.ndarray]:
    """
    距離に基づいて変位にフォールオフを適用
    """
    num_vertices = len(target_vertices)
    
    # 各頂点のソースメッシュの最近接頂点までの距離を計算
    print("ソースメッシュまでの距離を計算中...")
    distances = compute_distances_to_source_mesh(target_vertices, source_vertices, 
                                               batch_size=5000, max_workers=max_workers)
    
    # 距離に基づく重み付け
    distances = np.maximum(distances - 0.015, 0.0)
    weights = np.minimum(1.0, smooth_step(distances * 4.0, 0.0, 1.0))

    final_displacements = []
    
    for i in range(num_vertices):
        if weights[i] > 0:
            # 距離に応じた重み付けを適用
            blend_factor = weights[i]
            next_displacement = (1.0 - blend_factor) * target_displacements[i]
        else:
            next_displacement = target_displacements[i]
        
        final_displacements.append(next_displacement)
    
    return final_displacements


def process_vertex_batch(batch_data: Dict[str, Any]) -> Tuple[int, int, np.ndarray]:
    """
    頂点のバッチを処理する関数（マルチプロセス用）
    メモリ効率を改善
    
    Returns:
        Tuple[start_idx, end_idx, displacements]
    """
    start_idx = batch_data['start_idx']
    end_idx = batch_data['end_idx']
    batch_world_vertices = batch_data['batch_world_vertices']
    source_control_points = batch_data['source_control_points']
    rbf_weights = batch_data['rbf_weights']
    poly_weights = batch_data['poly_weights']
    epsilon = batch_data['epsilon']
    dim = batch_data['dim']
    
    current_batch_size = end_idx - start_idx
    
    try:
        # ターゲット頂点と制御点の間の距離を計算
        # メモリ効率のために一度に計算
        batch_dists = cdist(batch_world_vertices, source_control_points, 'sqeuclidean')
        batch_phi = np.sqrt(batch_dists + epsilon**2)
        
        # 多項式項の計算
        batch_P = np.ones((current_batch_size, dim + 1))
        batch_P[:, 1:] = batch_world_vertices
        
        # 各ターゲット頂点の変位を計算
        batch_displacements = np.dot(batch_phi, rbf_weights) + np.dot(batch_P, poly_weights)
        
        # 使用後すぐにメモリを解放
        del batch_dists, batch_phi, batch_P
        
        return start_idx, end_idx, batch_displacements
        
    except Exception as e:
        print(f"バッチ処理中にエラーが発生: {e}")
        raise


def rbf_interpolation_multithread(source_control_points: np.ndarray, 
                                 source_control_points_deformed: np.ndarray, 
                                 target_world_vertices: np.ndarray,
                                 epsilon: float = 1.0, 
                                 batch_size: int = 10000,  # デフォルトバッチサイズを削減
                                 max_workers: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    マルチプロセスRBF補間を使用してターゲットメッシュの新しい位置を計算
    メモリ効率を改善
    
    Parameters:
    - source_control_points: ソースメッシュの選択された制御点（基準位置）- ワールド座標
    - source_control_points_deformed: シェイプキーで変形後のソースメッシュの制御点 - ワールド座標
    - target_world_vertices: ターゲットメッシュの頂点（ワールド座標）
    - epsilon: RBFパラメータ
    - batch_size: 一度に処理するターゲット頂点の数（デフォルト値を削減）
    - max_workers: 最大ワーカー数（Noneの場合はCPUコア数に基づく）
    
    Returns:
    - 変位ベクトル
    - フォールオフ適用後の最終変位
    """
    # メモリモニタリングを開始
    memory_monitor = MemoryMonitor()
    total_vertices = len(target_world_vertices)
    
    # 最適なワーカー数を計算
    if max_workers is None:
        max_workers = get_optimal_worker_count(total_vertices, memory_monitor)
    
    print(f"マルチプロセスRBF補間を開始（ワーカー数: {max_workers}, 初期メモリ: {memory_monitor.initial_memory:.1f}GB）")
    
    # 変位ベクトルを計算（変形後の位置 - 元の位置）
    displacements = source_control_points_deformed - source_control_points
    
    # スケーリング係数を計算：距離の標準偏差に基づく値を使用
    if epsilon <= 0:
        # 平均距離に基づいて適切なepsilonを計算
        dists = cdist(source_control_points, source_control_points)
        mean_dist = np.mean(dists[dists > 0])
        epsilon = mean_dist  # 平均距離をepsilonとして使用
        print(f"自動計算されたepsilon: {epsilon}")
    
    # 制御点間の距離行列を計算
    print("RBF行列を計算中...")
    dist_matrix = cdist(source_control_points, source_control_points, 'sqeuclidean')
    
    # RBF行列を計算
    phi = np.sqrt(dist_matrix + epsilon**2)
    
    num_pts, dim = source_control_points.shape
    P = np.ones((num_pts, dim + 1))
    P[:, 1:] = source_control_points  # 多項式項のための拡張行列
    
    # 完全な線形システムを構築
    A = np.zeros((num_pts + dim + 1, num_pts + dim + 1))
    A[:num_pts, :num_pts] = phi
    A[:num_pts, num_pts:] = P
    A[num_pts:, :num_pts] = P.T
    
    # 右辺を設定
    b = np.zeros((num_pts + dim + 1, dim))
    b[:num_pts] = displacements
    
    # 解を求める
    print("線形システムを解いています...")
    try:
        # 通常の解法を試みる
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 行列が特異な場合、正則化して疑似逆行列を使用
        print("行列が特異です - 正則化を適用します")
        reg = np.eye(A.shape[0]) * 1e-6
        x = np.linalg.lstsq(A + reg, b, rcond=None)[0]
    
    # 重みを抽出
    rbf_weights = x[:num_pts]
    poly_weights = x[num_pts:]
    
    # 不要な変数を削除してメモリを解放
    del dist_matrix, phi, A, b, x
    
    # メモリ使用量をチェックしてバッチサイズを調整
    memory_increase = memory_monitor.get_memory_increase()
    if memory_increase > 1.0:  # 1GB以上増加した場合
        batch_size = memory_monitor.get_recommended_batch_size(batch_size, memory_increase)
        print(f"メモリ使用量に基づいてバッチサイズを調整: {batch_size}")
    
    # 結果を格納する配列を初期化
    target_displacements = np.zeros_like(target_world_vertices)
    
    # バッチデータを準備
    batch_tasks = []
    for batch_start in range(0, total_vertices, batch_size):
        batch_end = min(batch_start + batch_size, total_vertices)
        batch_world_vertices = target_world_vertices[batch_start:batch_end].copy()  # コピーを作成
        
        batch_data = {
            'start_idx': batch_start,
            'end_idx': batch_end,
            'batch_world_vertices': batch_world_vertices,
            'source_control_points': source_control_points,
            'rbf_weights': rbf_weights,
            'poly_weights': poly_weights,
            'epsilon': epsilon,
            'dim': dim
        }
        batch_tasks.append(batch_data)
    
    print(f"ターゲットメッシュの頂点を {len(batch_tasks)} バッチでマルチプロセス処理します（全 {total_vertices} 頂点）")

    # プロセス開始時にスレッド数を1に制限（各プロセスで実行）
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
    os.environ['NUMEXPR_NUM_THREADS'] = '2'
    
    # マルチプロセス処理
    processed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # バッチを並列処理
        future_to_batch = {executor.submit(process_vertex_batch, batch_data): batch_data for batch_data in batch_tasks}
        
        for future in as_completed(future_to_batch):
            try:
                start_idx, end_idx, batch_displacements = future.result()
                target_displacements[start_idx:end_idx] = batch_displacements
                
                processed_count += (end_idx - start_idx)
                progress_percent = (processed_count / total_vertices) * 100
                
                # プロセスプールでの進捗表示（メモリ監視は各プロセスで独立）
                if processed_count % (batch_size * 20) == 0 or processed_count == total_vertices:
                    if memory_monitor.enabled:
                        current_memory = memory_monitor.get_memory_usage()
                        print(f"進捗: {processed_count}/{total_vertices} 頂点処理完了 ({progress_percent:.1f}%) [メインプロセスメモリ: {current_memory:.1f}GB]")
                    else:
                        print(f"進捗: {processed_count}/{total_vertices} 頂点処理完了 ({progress_percent:.1f}%)")
                
            except Exception as exc:
                batch_data = future_to_batch[future]
                print(f"バッチ {batch_data['start_idx']}-{batch_data['end_idx']} でエラーが発生: {exc}")
                print("スタックトレース:")
                traceback.print_exc()
                raise exc
    
    print("マルチプロセス処理が完了しました")

    # フォールオフ処理を適用
    print("フォールオフ処理を適用中...")
    final_displacements = falloff_displacements(
        target_world_vertices, 
        target_displacements, 
        source_control_points,
        max_workers
    )

    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(os.cpu_count())
    os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())
    
    if memory_monitor.enabled:
        final_memory = memory_monitor.get_memory_usage()
        print(f"最終メモリ使用量: {final_memory:.1f}GB (増加: {memory_monitor.get_memory_increase():.1f}GB)")
    else:
        print("最終メモリ使用量: psutilが利用できないため表示できません")
    
    return target_displacements, np.array(final_displacements)


def process_temp_file(temp_file_path: str, max_workers: int = None, old_version: bool = False) -> str:
    """
    一時ファイルを処理してマルチプロセスRBF補間を実行
    
    Parameters:
    - temp_file_path: 一時データファイルのパス
    - max_workers: 最大ワーカー数
    - old_version: 旧バージョン形式で保存するかどうか
    
    Returns:
    - 出力ファイルのパス
    """
    print(f"一時データファイルを読み込み中: {temp_file_path}")
    
    # 一時データを読み込み
    data = np.load(temp_file_path, allow_pickle=True)
    
    # 新形式と旧形式の両方に対応
    if 'all_field_world_vertices' in data:
        # 新形式：ステップごとのフィールド
        all_field_world_vertices = data['all_field_world_vertices']
        print("新形式の一時データを検出（ステップごとのフィールド）")
    elif 'field_world_vertices' in data:
        # 旧形式：単一フィールド
        field_world_vertices = data['field_world_vertices']
        num_steps_temp = int(data['num_steps'])
        all_field_world_vertices = [field_world_vertices for _ in range(num_steps_temp)]
        print("旧形式の一時データを検出（単一フィールドを複製）")
    else:
        raise ValueError("フィールドデータが見つかりません")
    
    field_world_matrix = data['field_world_matrix']
    all_step_data = data['all_step_data']
    source_world_matrix = data['source_world_matrix']
    epsilon = float(data['epsilon'])
    num_steps = int(data['num_steps'])
    invert = bool(data['invert'])
    source_avatar_name = str(data['source_avatar_name'])
    target_avatar_name = str(data['target_avatar_name'])
    source_shape_key_name = str(data['source_shape_key_name'])
    save_shape_key_mode = bool(data['save_shape_key_mode'])
    
    print(f"読み込み完了:")
    print(f"  ステップ数: {num_steps}")
    for step in range(num_steps):
        print(f"  ステップ {step+1} フィールド頂点数: {len(all_field_world_vertices[step])}")
    print(f"  逆変形: {invert}")
    print(f"  Epsilon: {epsilon}")
    
    # 各ステップの変位を計算
    all_displacements = []
    all_target_world_vertices = []
    
    for step in range(num_steps):
        step_data = all_step_data[step]
        
        print(f"\n=== ステップ {step+1}/{num_steps} の処理 ===")
        print(f"シェイプキー値: {step_data['step_value']}")
        
        source_control_points = step_data['control_points_original']
        source_control_points_deformed = step_data['control_points_deformed']
        
        # 対応するステップのフィールドを取得
        current_field_vertices = all_field_world_vertices[step]
        print(f"使用するフィールド頂点数: {len(current_field_vertices)}")

        print(f"current_field_vertices の型: {type(current_field_vertices)}")
        print(f"current_field_vertices の形状: {current_field_vertices.shape}")
        print(f"current_field_vertices のデータ型: {current_field_vertices.dtype}")
        print(f"current_field_vertices の要素数: {len(current_field_vertices)}")
        
        # 変位の最大値をチェック
        displacements = source_control_points_deformed - source_control_points
        max_disp = np.max(np.linalg.norm(displacements, axis=1))
        print(f"制御点の最大変位: {max_disp}")

        # マルチプロセスRBF補間を実行（メモリ効率化のためバッチサイズを調整）
        target_displacements, final_displacements = rbf_interpolation_multithread(
            source_control_points, 
            source_control_points_deformed, 
            current_field_vertices,
            epsilon,
            batch_size=1000,  # メモリ効率化のためバッチサイズを削減
            max_workers=max_workers
        )
        
        all_target_world_vertices.append(current_field_vertices.copy())
        all_displacements.append(final_displacements)
        
        print(f"ステップ {step+1} の変位計算完了")
    
    # 出力ファイルパスを生成
    base_dir = os.path.dirname(temp_file_path)
    
    if save_shape_key_mode:
        direction_suffix = "_inv" if invert else ""
        output_path = os.path.join(base_dir, f"deformation_{source_avatar_name}_shape_{source_shape_key_name}{direction_suffix}.npz")
    else:
        direction_suffix = "_inv" if invert else ""
        output_path = os.path.join(base_dir, f"deformation_{source_avatar_name}_to_{target_avatar_name}{direction_suffix}.npz")
    
    # 結果を保存
    save_field_data_multi_step(
        field_world_matrix,
        output_path,
        all_target_world_vertices,
        all_displacements,
        num_steps,
        old_version=old_version,
        enable_x_mirror=data.get('enable_x_mirror', False)
    )
    
    print(f"結果を保存しました: {output_path}")
    return output_path


def save_field_data_multi_step(world_matrix, filepath, 
                              all_field_points, 
                              all_delta_positions, 
                              num_steps,
                              old_version=False,
                              enable_x_mirror=True):
    """
    複数ステップのDeformation Fieldの変形前後の差分をnumpy arrayとして直接保存
    enable_x_mirrorが有効な場合、X座標が0以上のデータのみを保存
    """
    
    kdtree_query_k = 27
    
    # RBF補間のパラメータを追加
    rbf_epsilon = 0.00001  # 固定値
    rbf_smoothing = 0.0    # スムージングパラメータ
   
    # データを保存
    if old_version:
        np.savez(filepath,
                field_points=all_field_points[0],
                delta_positions=all_delta_positions[0],
                num_steps=num_steps,
                world_matrix=world_matrix,
                kdtree_query_k=kdtree_query_k,
                rbf_epsilon=rbf_epsilon,
                rbf_smoothing=rbf_smoothing)
    else:
        # enable_x_mirrorが有効な場合、X座標が0以上のデータのみフィルタリング
        if enable_x_mirror:
            filtered_field_points = []
            filtered_delta_positions = []
            
            for step in range(num_steps):
                field_points = all_field_points[step]
                delta_positions = all_delta_positions[step]
                
                if len(field_points) > 0:
                    # X座標が0以上のインデックスを取得
                    x_positive_mask = field_points[:, 0] >= 0.0
                    filtered_field = field_points[x_positive_mask]
                    filtered_delta = delta_positions[x_positive_mask]
                    
                    filtered_field_points.append(filtered_field.astype(np.float32))
                    filtered_delta_positions.append(filtered_delta.astype(np.float32))
                    
                    print(f"ステップ {step+1}: 元の頂点数 {len(field_points)} → フィルタ後 {len(filtered_field)}")
                else:
                    filtered_field_points.append(np.array([]))
                    filtered_delta_positions.append(np.array([]))
                    print(f"ステップ {step+1}: フィールド頂点数 0")
        else:
            # ミラーが無効の場合、float32にキャストのみ行う
            filtered_field_points = []
            filtered_delta_positions = []
            
            for step in range(num_steps):
                field_points = all_field_points[step]
                delta_positions = all_delta_positions[step]
                
                if len(field_points) > 0:
                    filtered_field_points.append(field_points.astype(np.float32))
                    filtered_delta_positions.append(delta_positions.astype(np.float32))
                    print(f"ステップ {step+1}: 頂点数 {len(field_points)} (ミラーフィルタなし)")
                else:
                    filtered_field_points.append(np.array([]))
                    filtered_delta_positions.append(np.array([]))
                    print(f"ステップ {step+1}: フィールド頂点数 0")
        
        np.savez(filepath,
                all_field_points=np.array(filtered_field_points, dtype=object),
                all_delta_positions=np.array(filtered_delta_positions, dtype=object),
                num_steps=num_steps,
                world_matrix=world_matrix,
                kdtree_query_k=kdtree_query_k,
                rbf_epsilon=rbf_epsilon,
                rbf_smoothing=rbf_smoothing,
                enable_x_mirror=enable_x_mirror)
        
    print(f"Deformation Field差分データを保存しました: {filepath}")
    print(f"ステップ数: {num_steps}")
    if old_version:
        print(f"ステップ 1: 頂点数 {len(all_field_points[0])}")
    else:
        for step in range(num_steps):
            if step < len(filtered_field_points):
                print(f"ステップ {step+1}: 頂点数 {len(filtered_field_points[step])}")
    print(f"RBF関数: multi_quadratic_biharmonic, epsilon: {rbf_epsilon}, smoothing: {rbf_smoothing}")


def process_multiple_temp_files(temp_file_pattern: str, max_workers: int = None, old_version: bool = False) -> List[str]:
    """
    複数の一時ファイルを処理（順方向と逆方向）
    
    Parameters:
    - temp_file_pattern: 一時データファイルのパターン（_invサフィックスなし）
    - max_workers: 最大ワーカー数
    - old_version: 旧バージョン形式で保存するかどうか
    
    Returns:
    - 出力ファイルのパスのリスト
    """
    output_paths = []
    
    # 順方向ファイルと逆方向ファイルのパスを生成
    temp_files = []
    
    # 基本ファイル名から順方向と逆方向のファイルパスを生成
    base_path = temp_file_pattern
    if base_path.endswith('.npz'):
        base_path = base_path[:-4]
    
    forward_file = f"{base_path}.npz"
    inverse_file = f"{base_path}_inv.npz"
    
    # 存在するファイルのみを処理対象に追加
    for temp_file in [forward_file, inverse_file]:
        if os.path.exists(temp_file):
            temp_files.append(temp_file)
        else:
            print(f"警告: ファイルが見つかりません: {temp_file}")
    
    if not temp_files:
        print("エラー: 処理対象のファイルが見つかりません")
        return output_paths
    
    total_start_time = time.time()
    
    for i, temp_file in enumerate(temp_files):
        print(f"\n{'='*60}")
        print(f"ファイル {i+1}/{len(temp_files)}: {os.path.basename(temp_file)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            output_path = process_temp_file(temp_file, max_workers, old_version)
            output_paths.append(output_path)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"ファイル {i+1} 処理完了: {processing_time:.2f}秒")
            print(f"出力ファイル: {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"ファイル {i+1} でエラーが発生しました: {e}")
            print("スタックトレース:")
            traceback.print_exc()
            continue
    
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print(f"全体の処理完了")
    print(f"{'='*60}")
    print(f"処理されたファイル数: {len(output_paths)}/{len(temp_files)}")
    print(f"総処理時間: {total_processing_time:.2f}秒")
    if output_paths:
        print("出力ファイル:")
        for path in output_paths:
            print(f"  - {os.path.basename(path)}")
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description='RBF変形の外部マルチプロセス処理（メモリ効率化対応）')
    parser.add_argument('temp_file', help='一時データファイルのパス（基本ファイル名、自動的に_invファイルも処理）')
    parser.add_argument('--max-workers', type=int, default=16, 
                       help='最大プロセス数（デフォルト: CPUコア数・メモリ容量に基づく自動設定）')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='バッチサイズ（デフォルト: 10000、プロセス間での効率を考慮）')
    parser.add_argument('--single-file', action='store_true',
                       help='単一ファイルのみを処理（_invファイルを自動検出しない）')
    parser.add_argument('--memory-limit', type=float, default=None,
                       help='メモリ使用量の上限（GB単位、デフォルト: 制限なし）')
    parser.add_argument('--low-memory', action='store_true',
                       help='低メモリモード（バッチサイズとワーカー数を自動的に制限）')
    parser.add_argument('--old-version', action='store_true',
                       help='旧バージョン形式で保存（互換性のため）')
    
    args = parser.parse_args()

    if PSUTIL_AVAILABLE:
        set_cpu_affinity()

    # 低メモリモードの場合、設定を調整（プロセスプール用）
    if args.low_memory:
        print("低メモリモードが有効です。バッチサイズとプロセス数を制限します。")
        if args.batch_size > 2000:
            args.batch_size = 2000
        if args.max_workers is None or args.max_workers > 1:
            args.max_workers = 1  # プロセスプールでは1つに制限
    
    print(f"CPU数: {os.cpu_count()}")
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(os.cpu_count())
    os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())
    
    np.__config__.show()
    
    # メモリ使用量の情報を表示（psutil利用可能時のみ）
    if PSUTIL_AVAILABLE:
        memory_info = psutil.virtual_memory()
        print(f"システムメモリ情報:")
        print(f"  総メモリ: {memory_info.total / 1024**3:.1f}GB")
        print(f"  利用可能メモリ: {memory_info.available / 1024**3:.1f}GB")
        print(f"  メモリ使用率: {memory_info.percent:.1f}%")
        if args.memory_limit:
            print(f"  設定メモリ制限: {args.memory_limit:.1f}GB")
    else:
        print("システムメモリ情報: psutilが利用できないため表示できません")
        if args.memory_limit:
            print(f"設定メモリ制限: {args.memory_limit:.1f}GB")
    
    if args.single_file:
        print(f"単一ファイル処理モード")
        if not os.path.exists(args.temp_file):
            print(f"エラー: 一時データファイルが見つかりません: {args.temp_file}")
            sys.exit(1)
        
        start_time = time.time()
        
        try:
            output_path = process_temp_file(args.temp_file, args.max_workers, args.old_version)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"\n=== 処理完了 ===")
            print(f"処理時間: {processing_time:.2f}秒")
            print(f"出力ファイル: {output_path}")
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            print("スタックトレース:")
            traceback.print_exc()
            sys.exit(1)
    else:
        # 複数ファイル処理モード（デフォルト）
        try:
            print(f"複数ファイル処理モード")
            output_paths = process_multiple_temp_files(args.temp_file, args.max_workers, args.old_version)
            
            if not output_paths:
                print("エラー: 処理されたファイルがありません")
                sys.exit(1)
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            print("スタックトレース:")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main() 