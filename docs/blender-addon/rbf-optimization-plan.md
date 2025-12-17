# RBF処理 高速化・最適化 実装計画

## 1. 概要

本ドキュメントは、MochiFitter Blender Addon の RBF (Radial Basis Function) 補間処理の高速化・最適化に関する実装計画を記述します。

### 1.1 背景

Issue #19 の改善により、RBF処理は正常に動作するようになりました。しかし、大規模なメッシュ（140万頂点、1.5万制御点）の処理には約52秒を要しており、更なる高速化の余地があります。

### 1.2 目標

| 目標 | 現在 | Phase 1 | Phase 2 | Phase 3 |
|------|------|---------|---------|---------|
| 処理時間 | 52秒 | 22-25秒 | 8-12秒 | 2-5秒 |
| 短縮率 | - | 55-60% | 80% | 90% |
| メモリ使用量 | 16GB | 8GB | 8GB | 4-8GB |

### 1.3 関連Issue/PR

- Issue #19: UI ブロック問題（解決済み）
- Issue #24: 高速化・最適化
- PR #20, #21, #22, #23: Issue #19 の修正

---

## 2. 現在のパフォーマンス分析

### 2.1 処理フロー

```
┌─────────────────────────────────────────────────────────────────┐
│  RBF補間処理フロー（rbf_multithread_processor.py）              │
└─────────────────────────────────────────────────────────────────┘

├─ [フェーズ1] RBF行列構築（シングルプロセス）
│  ├─ cdist() で制御点間距離行列計算: O(n²)
│  ├─ Multi-Quadratic Biharmonic 関数適用
│  └─ 完全線形システム A 構築: (n+4) x (n+4)
│
├─ [フェーズ2] 線形システム求解（シングルプロセス）★ボトルネック
│  ├─ np.linalg.solve() 実行
│  └─ 演算量: O(n³)
│
├─ [フェーズ3] ターゲット頂点RBF評価（マルチプロセス）
│  ├─ バッチサイズ: 1,000頂点/バッチ
│  └─ ProcessPoolExecutor (max_workers=3-4)
│
└─ [フェーズ4] フォールオフ処理（並列）
   └─ KDTree 距離計算 + smooth_step() 重み付け
```

### 2.2 処理時間の内訳

| フェーズ | 処理内容 | 時間 | 割合 | 計算量 |
|---------|---------|------|------|--------|
| 1 | RBF行列構築 | 5秒 | 10% | O(n²) |
| 2 | **線形システム求解** | **27秒** | **52%** | O(n³) |
| 3 | ターゲット距離計算 | 20秒 | 38% | O(n×m) |
| - | RBF評価 | 10秒 | - | O(n×m) |
| 4 | フォールオフ | 5秒 | - | O(m) |
| | **合計** | **~52秒** | 100% | |

※ n=15,779（制御点）, m=1,400,370（ターゲット頂点）

### 2.3 ボトルネック分析

1. **線形システム求解（52%）**: LU分解のO(n³)コストが支配的
2. **ターゲット距離計算（38%）**: 2.2×10^10 回の距離計算
3. **バッチ処理オーバーヘッド**: 小さなバッチサイズ（1,000）による通信コスト

---

## 3. 最適化施策

### 3.1 Phase 1: 即時実装可能（優先度高）

#### P0-1: float32精度使用

**概要**: 倍精度（float64）から単精度（float32）へ変更

**期待効果**:
- メモリ使用量: 50%削減（16GB → 8GB）
- 処理速度: +15-20%（キャッシュ効率向上）

**実装箇所**: `rbf_multithread_processor.py`

```python
# 変更前
A = np.zeros((num_pts + dim + 1, num_pts + dim + 1), dtype=np.float64)
dist_matrix = cdist(source_control_points, source_control_points, 'sqeuclidean')

# 変更後
A = np.zeros((num_pts + dim + 1, num_pts + dim + 1), dtype=np.float32)
dist_matrix = cdist(
    source_control_points.astype(np.float32),
    source_control_points.astype(np.float32),
    'sqeuclidean'
).astype(np.float32)
```

**リスク**:
- 数値安定性の若干の低下
- RBF補間は本質的にC0連続であり、float32で十分な精度

**検証方法**:
- 同一入力での出力比較（float64 vs float32）
- 視覚的な差異確認（0.001mm以下であればOK）

---

#### P0-2: Gaussianカーネル導入

**概要**: Multi-Quadratic Biharmonic から Gaussian RBF へ変更

**期待効果**: 処理速度 +10-15%（sqrt → exp は高速）

**実装箇所**: `rbf_multithread_processor.py`

```python
# 変更前: Multi-Quadratic Biharmonic
phi = np.sqrt(dist_matrix + epsilon**2)

# 変更後: Gaussian RBF
phi = np.exp(-dist_matrix / (epsilon**2))
```

**リスク**: 極低（精度同等もしくは若干改善）

**検証方法**:
- ベンチマーク比較
- 補間精度テスト

---

#### P1-1: Cholesky分解適用

**概要**: LU分解の代わりにCholesky分解を使用

**期待効果**: 線形求解 25-30%高速化（27秒 → 19-20秒）

**理論的背景**:
- Cholesky分解: (1/3)×n³ FLOPS
- LU分解: (2/3)×n³ FLOPS
- RBF行列は対称正定値であり、Cholesky適用可能

**実装箇所**: `rbf_multithread_processor.py`

```python
import scipy.linalg

# 変更前
x = np.linalg.solve(A, b)

# 変更後
try:
    L = scipy.linalg.cholesky(A, lower=True)
    x = scipy.linalg.cho_solve((L, True), b)
except np.linalg.LinAlgError:
    # フォールバック: 条件数が悪い場合
    print("Cholesky分解失敗 - LU分解にフォールバック")
    x = np.linalg.solve(A, b)
```

**リスク**: 低（条件数確認必要、フォールバック実装で安全）

---

#### P1-2: バッチサイズ動的最適化

**概要**: 固定バッチサイズ（1,000）を動的に最適化

**期待効果**: 処理速度 +15-25%（通信オーバーヘッド削減）

**実装箇所**: `rbf_multithread_processor.py`

```python
def calculate_optimal_batch_size(available_memory_gb, num_control_pts, num_workers):
    """
    メモリ制約を考慮した最適バッチサイズ計算
    """
    # バッチあたりメモリ使用量
    # - 距離行列: batch_size × num_control_pts × 8 bytes
    # - 中間データ: batch_size × num_control_pts × 8 bytes
    bytes_per_vertex = num_control_pts * 8 * 2

    # 利用可能メモリの70%を使用（安全マージン）
    target_bytes = available_memory_gb * 0.7 * 1024**3

    # ワーカー数で分割
    bytes_per_worker = target_bytes / num_workers

    # 最適バッチサイズ
    optimal_batch = int(bytes_per_worker / bytes_per_vertex)

    # 上限・下限の設定
    return max(1000, min(optimal_batch, 20000))
```

**期待値**:
- 現在: batch_size=1,000 固定
- 最適化後: batch_size=5,000-20,000（環境依存）

---

### 3.2 Phase 2: 中期実装

#### P2-1: Numba JIT化（cdist）

**概要**: 距離計算をNumba JITでコンパイル

**期待効果**: 距離計算 3-5倍高速化（20秒 → 4-6秒）

**依存関係**: `numba` パッケージ

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True, fastmath=True)
def cdist_euclidean_numba(A, B):
    """
    Numba JIT版 ユークリッド距離計算
    """
    m, n = A.shape[0], B.shape[0]
    d = np.zeros((m, n), dtype=np.float32)

    for i in prange(m):
        for j in range(n):
            dist = 0.0
            for k in range(A.shape[1]):
                dx = A[i, k] - B[j, k]
                dist += dx * dx
            d[i, j] = np.sqrt(dist)

    return d
```

**注意点**:
- 初回呼び出し時にJITコンパイルが発生（数秒のオーバーヘッド）
- キャッシュ機能で2回目以降は高速

---

#### P2-2: 前処理付き反復ソルバー（GMRES）

**概要**: 直接法（LU/Cholesky）から反復法（GMRES）へ変更

**期待効果**: 線形求解 55-70%高速化（27秒 → 8-12秒）

**理論的背景**:
- RBF行列は通常 well-conditioned（条件数 κ~100-500）
- 反復法は少ない反復で収束可能
- 前処理により更に高速化

```python
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.linalg import lu_factor, lu_solve

def solve_with_gmres(A, b, tol=1e-6):
    """
    前処理付きGMRESソルバー
    """
    n = A.shape[0]

    # 不完全LU分解による前処理
    lu, piv = lu_factor(A)

    def preconditioner(x):
        return lu_solve((lu, piv), x)

    M = LinearOperator((n, n), matvec=preconditioner)

    # GMRES実行
    x, info = gmres(A, b, M=M, tol=tol, restart=100, maxiter=500)

    if info != 0:
        print(f"GMRES警告: 収束しませんでした (info={info})")

    return x
```

**リスク**: 中（条件数に依存、フォールバック必要）

---

#### P2-3: ハイブリッド並列化（Thread+Process）

**概要**: タスク特性に応じた最適な並列化戦略

**期待効果**: CPU使用率 40-60% → 85-95%、速度 +30-40%

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def rbf_interpolation_hybrid(control_pts, target_verts, max_workers=4):
    """
    ハイブリッド並列化によるRBF補間
    """
    # フェーズ1: 距離計算（メモリ集約）→ ProcessPool（少数）
    with ProcessPoolExecutor(max_workers=2) as proc_pool:
        distance_results = list(proc_pool.map(
            compute_distance_batch,
            distance_batches,
            chunksize=10
        ))

    # フェーズ2: RBF評価（CPU集約）→ ThreadPool（多数）
    # NumPy操作中はGILが解放されるため、ThreadPoolが効果的
    with ThreadPoolExecutor(max_workers=max_workers * 2) as thread_pool:
        rbf_results = list(thread_pool.map(
            evaluate_rbf_batch,
            rbf_batches,
            chunksize=20
        ))

    return np.concatenate(rbf_results)
```

---

### 3.3 Phase 3: 長期研究（オプショナル）

#### P3-1: GPU加速（CuPy）

**概要**: CuPyを使用したGPU加速

**期待効果**: +200-500%（GPU依存）

**依存関係**: `cupy`, NVIDIA GPU + CUDA

```python
import cupy as cp

def rbf_interpolation_gpu(control_pts, target_verts, epsilon):
    """
    GPU加速版RBF補間
    """
    # データをGPUに転送
    control_pts_gpu = cp.asarray(control_pts)
    target_verts_gpu = cp.asarray(target_verts)

    # GPU上で距離計算
    dist_matrix = cp.sqrt(
        cp.sum((target_verts_gpu[:, None, :] - control_pts_gpu[None, :, :]) ** 2, axis=2)
    )

    # RBF評価
    phi = cp.exp(-dist_matrix / (epsilon ** 2))

    # 結果をCPUに転送
    return cp.asnumpy(phi @ rbf_weights_gpu)
```

**リスク**: 高（GPU依存、互換性問題）

---

#### P3-2: コンパクト台RBF

**概要**: サポート領域外でゼロとなるRBFカーネル

**期待効果**: +40-60%（疎行列化による高速化）

```python
def wendland_c4(r, R=1.0):
    """
    Wendland C4 RBF（コンパクト台）
    サポート: [0, R], R外ではゼロ
    """
    r = np.minimum(r, R)
    ratio = 1 - r / R
    return np.where(r < R,
                    (ratio ** 6) * (35 * (r/R)**2 + 18 * (r/R) + 3),
                    0.0)
```

**リスク**: 中（精度低下の可能性、境界効果）

---

## 4. 実装ロードマップ

### 4.1 スケジュール

```
Week 1-2 (Quick Wins):
├─ [ ] P0-1: float32精度導入
├─ [ ] P0-2: Gaussianカーネル導入
├─ [ ] ベンチマーク環境構築
└─ [ ] 単体テスト・回帰テスト

Week 3-4 (Core Optimization):
├─ [ ] P1-1: Cholesky分解適用
├─ [ ] P1-2: バッチサイズ動的最適化
├─ [ ] 統合テスト
└─ [ ] パフォーマンス測定

Week 5-8 (Advanced):
├─ [ ] P2-1: Numba JIT化（オプショナル依存）
├─ [ ] P2-2: 反復ソルバー検討
├─ [ ] P2-3: ハイブリッド並列化
└─ [ ] ストレステスト

Month 2-3 (Research):
├─ [ ] P3-1: GPU加速検討
├─ [ ] P3-2: コンパクト台RBF実験
└─ [ ] ユーザーフィードバック収集
```

### 4.2 マイルストーン

| マイルストーン | 目標時間 | 達成基準 |
|---------------|---------|---------|
| M1: Phase 1完了 | Week 4 | 52秒 → 22-25秒 |
| M2: Phase 2完了 | Week 8 | → 8-12秒 |
| M3: Phase 3検証 | Month 3 | GPU環境での動作確認 |

---

## 5. ベンチマーク手法

### 5.1 テスト環境

```yaml
標準テスト環境:
  CPU: 32コア（推奨）
  RAM: 32GB以上
  OS: Windows 10/11
  Blender: 4.0+
  Python: 3.10+

テストデータ:
  制御点数: 15,779
  ターゲット頂点数: 1,400,370
  行列サイズ: 15,783 x 15,783
```

### 5.2 ベンチマークスクリプト

```python
import time
import numpy as np

def benchmark_rbf_processing(control_pts, target_verts, iterations=3):
    """
    RBF処理のベンチマーク
    """
    results = {
        'matrix_build': [],
        'linear_solve': [],
        'batch_process': [],
        'falloff': [],
        'total': []
    }

    for i in range(iterations):
        start_total = time.time()

        # フェーズ1: 行列構築
        start = time.time()
        # ... 行列構築処理 ...
        results['matrix_build'].append(time.time() - start)

        # フェーズ2: 線形求解
        start = time.time()
        # ... 線形求解処理 ...
        results['linear_solve'].append(time.time() - start)

        # ... 他のフェーズ ...

        results['total'].append(time.time() - start_total)

    # 平均値を計算
    return {k: np.mean(v) for k, v in results.items()}
```

### 5.3 測定項目

| 項目 | 測定方法 | 目標 |
|------|---------|------|
| 処理時間 | time.time() | 各フェーズ個別 + 合計 |
| メモリ使用量 | psutil.Process().memory_info() | ピーク値 |
| CPU使用率 | psutil.cpu_percent() | 平均値 |
| 精度 | np.allclose(result_new, result_old) | 差異 < 0.001mm |

---

## 6. リスクと対策

### 6.1 技術的リスク

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|---------|------|
| float32精度不足 | 中 | 低 | 閾値テスト、フォールバック |
| Cholesky分解失敗 | 中 | 低 | LU分解へのフォールバック |
| GMRES非収束 | 高 | 中 | 直接法へのフォールバック |
| Numba互換性問題 | 低 | 中 | オプショナル機能として実装 |
| GPU未対応環境 | 低 | 高 | CPU版を維持 |

### 6.2 フォールバック戦略

```python
def solve_linear_system_with_fallback(A, b):
    """
    フォールバック付き線形システム求解
    """
    # 優先度1: Cholesky分解
    try:
        L = scipy.linalg.cholesky(A, lower=True)
        return scipy.linalg.cho_solve((L, True), b)
    except np.linalg.LinAlgError:
        pass

    # 優先度2: GMRES
    try:
        x, info = gmres(A, b, tol=1e-6, maxiter=500)
        if info == 0:
            return x
    except Exception:
        pass

    # 優先度3: LU分解（確実）
    return np.linalg.solve(A, b)
```

---

## 7. 参考資料

### 7.1 関連ドキュメント

- [RBF補間の理論](https://en.wikipedia.org/wiki/Radial_basis_function)
- [SciPy線形代数](https://docs.scipy.org/doc/scipy/reference/linalg.html)
- [Numba並列処理](https://numba.readthedocs.io/en/stable/user/parallel.html)
- [CuPy GPUコンピューティング](https://docs.cupy.dev/en/stable/)

### 7.2 参考論文

1. Wendland, H. (1995). "Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree"
2. Fasshauer, G. E. (2007). "Meshfree Approximation Methods with MATLAB"

---

## 更新履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2024-12-17 | 1.0 | 初版作成 |
