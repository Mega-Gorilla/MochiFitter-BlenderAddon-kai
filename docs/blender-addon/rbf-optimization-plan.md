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
- 3Dメッシュ変形用途ではfloat32で十分な精度（視覚的に差異が認識できない）

**検証方法**:
- 同一入力での出力比較（float64 vs float32）
- 視覚的な差異確認（0.001mm以下であればOK）

---

#### P0-2: Gaussianカーネル導入

**概要**: Multi-Quadratic Biharmonic から Gaussian RBF へ変更

**期待効果**: 数値安定性向上、条件数改善

**実装箇所**: `rbf_multithread_processor.py`

```python
# 変更前: Multi-Quadratic Biharmonic
phi = np.sqrt(dist_matrix + epsilon**2)

# 変更後: Gaussian RBF
phi = np.exp(-dist_matrix / (epsilon**2))
```

**Gaussianカーネルの利点**:
- **正定値性**: Gaussian RBF は常に正定値行列を生成（phi ブロックのみ考慮時）
- **局所的影響**: 距離が増すと急速に減衰し、遠方制御点の影響が抑制される
- **無限回微分可能**: 補間結果が滑らかになる
- **epsilon調整が容易**: 形状パラメータの直感的な解釈（影響半径）

**注意**: 計算速度は sqrt と exp でほぼ同等（環境依存）。
主な採用理由は数値的性質の改善。

**リスク**: 低（epsilon パラメータの再調整が必要な場合あり）

**検証方法**:
- 補間精度テスト（視覚的確認）
- 条件数の比較（`np.linalg.cond(A)`）

---

#### P1-1: LU分解の再利用（3成分同時求解）

**概要**: x, y, z 各成分の線形求解でLU分解を再利用

**期待効果**: 線形求解 60-65%高速化（27秒 → 10-12秒）

**理論的背景**:

RBF補間行列は以下のサドルポイント構造を持つ：
```
A = [[phi, P ],
     [P^T, 0 ]]
```
- `phi`: RBFカーネル値行列（n×n、対称）
- `P`: 多項式項（n×4、[1, x, y, z]）
- `0`: ゼロブロック（4×4）

この構造は**対称であるが正定値ではない**（ゼロブロックの存在により不定値）。
したがって Cholesky 分解は適用不可。

しかし、現在 x, y, z 成分ごとに3回 `np.linalg.solve()` を呼んでおり、
毎回同じ行列 A の LU 分解を再計算している。**行列 A は共通**なので、
LU 分解を1回行い、後方代入を3回行うことで大幅な高速化が可能。

**実装箇所**: `rbf_multithread_processor.py`

```python
import scipy.linalg

# 変更前: 3回のsolve（各回でLU分解）
weights_x = np.linalg.solve(A, b_x)  # LU分解 + 後方代入
weights_y = np.linalg.solve(A, b_y)  # LU分解 + 後方代入
weights_z = np.linalg.solve(A, b_z)  # LU分解 + 後方代入

# 変更後: LU分解1回 + 後方代入3回
lu, piv = scipy.linalg.lu_factor(A)  # LU分解1回のみ
weights_x = scipy.linalg.lu_solve((lu, piv), b_x)
weights_y = scipy.linalg.lu_solve((lu, piv), b_y)
weights_z = scipy.linalg.lu_solve((lu, piv), b_z)

# または、右辺を結合して一度に求解
B = np.column_stack([b_x, b_y, b_z])  # (n+4, 3)
weights = scipy.linalg.lu_solve((lu, piv), B)  # (n+4, 3)
weights_x, weights_y, weights_z = weights[:, 0], weights[:, 1], weights[:, 2]
```

**計算量の比較**:
- 変更前: LU分解 O(n³) × 3回 = O(3n³)
- 変更後: LU分解 O(n³) × 1回 + 後方代入 O(n²) × 3回 ≈ O(n³)
- n=15,783 の場合、約66%の計算量削減

**リスク**: 極低（既存ロジックの単純な最適化）

---

#### P1-2: バッチサイズ動的最適化

**概要**: 固定バッチサイズ（1,000）を動的に最適化

**現状**: `rbf_multithread_processor.py` の `batch_size=1000` はハードコードされている
（`process_batches()` 関数内）

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

**概要**: 直接法（LU）から反復法（GMRES）へ変更

**期待効果**: 線形求解 40-60%高速化（10秒 → 4-6秒）※P1-1適用後

**理論的背景**:
- RBF行列は通常 well-conditioned（条件数 κ~100-500）
- 反復法は少ない反復で収束可能
- **不完全LU分解（ILU）** による前処理で収束を加速

**注意**: 完全LU分解を前処理に使うと直接法と同等のコストがかかるため、
疎行列化した不完全LU（spilu）を使用する必要がある。

```python
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres, spilu, LinearOperator

def solve_with_gmres(A, b, tol=1e-6):
    """
    不完全LU前処理付きGMRESソルバー
    """
    n = A.shape[0]

    # 密行列を疎行列に変換（ILU用）
    A_sparse = csc_matrix(A)

    # 不完全LU分解（ILU）による前処理
    # drop_tol: 小さい要素を無視する閾値
    # fill_factor: 元の非ゼロ要素数に対する許容倍率
    ilu = spilu(A_sparse, drop_tol=1e-4, fill_factor=10)

    def preconditioner(x):
        return ilu.solve(x)

    M = LinearOperator((n, n), matvec=preconditioner)

    # GMRES実行
    x, info = gmres(A, b, M=M, tol=tol, restart=100, maxiter=500)

    if info != 0:
        print(f"GMRES警告: 収束しませんでした (info={info})")

    return x
```

**注意点**:
- RBF行列は密行列のため、ILUの効果は限定的な場合がある
- drop_tol の調整が必要（大きすぎると精度低下、小さすぎると高コスト）
- 直接法（P1-1のLU再利用）の方が安定する可能性あり

**リスク**: 中（条件数・パラメータ調整に依存、フォールバック必要）

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
| float32精度不足 | 中 | 低 | 閾値テスト、float64へのフォールバック |
| LU分解の数値不安定性 | 中 | 極低 | ピボット選択、条件数チェック |
| GMRES非収束 | 高 | 中 | 直接法へのフォールバック |
| Numba互換性問題 | 低 | 中 | オプショナル機能として実装 |
| GPU未対応環境 | 低 | 高 | CPU版を維持 |

### 6.2 フォールバック戦略

```python
import scipy.linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres, spilu, LinearOperator

def solve_linear_system_with_fallback(A, b):
    """
    フォールバック付き線形システム求解

    注意: RBF行列はサドルポイント構造のため、Cholesky分解は不可。
    """
    # 優先度1: LU分解（最も安定）
    try:
        lu, piv = scipy.linalg.lu_factor(A)
        return scipy.linalg.lu_solve((lu, piv), b)
    except np.linalg.LinAlgError:
        pass

    # 優先度2: ILU前処理付きGMRES
    try:
        n = A.shape[0]
        A_sparse = csc_matrix(A)
        ilu = spilu(A_sparse, drop_tol=1e-4)
        M = LinearOperator((n, n), matvec=ilu.solve)
        x, info = gmres(A, b, M=M, tol=1e-6, maxiter=500)
        if info == 0:
            return x
    except Exception:
        pass

    # 優先度3: np.linalg.solve（最終手段）
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
| 2025-12-17 | 1.0 | 初版作成 |
