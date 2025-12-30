# retarget_script CPU最適化 実装計画

## 1. 概要

本ドキュメントは、`retarget_script2_14.py` の処理高速化に関する実装計画を記述します。
GPU化は環境依存が大きいため、まずはCPU側の最適化を優先します。

### 1.1 背景

Issue #36 のリファクタリング完了後、次のステップとして処理速度の改善を検討。
ベンチマーク結果から、ウェイト転送・スムージング処理が主要なボトルネックであることが判明。

### 1.2 目標

| マイルストーン | 現在 | 目標 | 短縮率 |
|---------------|------|------|--------|
| Phase 1 完了 | 287秒 | 180-200秒 | 30-40% |
| Phase 2 完了 | - | 120-150秒 | 45-60% |
| (Phase 3 GPU) | - | 60-90秒 | 70-80% |

### 1.3 関連Issue/PR

- Issue #48: retarget_script CPU最適化
- Issue #36: コード品質改善（完了）
- Issue #24: RBF処理の高速化（Blenderアドオン側）

---

## 2. 現在のパフォーマンス分析

### 2.1 ベンチマーク結果

テストケース: Beryl → Template → mao（チェーン処理）

| 日時 | バージョン | 処理時間 | 備考 |
|------|-----------|----------|------|
| 2025-12-30 11:52 | Phase 2 完了 | **287.62秒** | ベースライン |
| 2025-12-30 14:04 | Phase 3.1 | 291.31秒 | RetargetContext導入 |
| 2025-12-30 16:22 | Phase 3.2 + PR#47 | 343.59秒 | データ整合性修正含む |

### 2.2 処理フロー

```
全体処理時間: ~280秒（Blenderスクリプト内部）

├─ ペア1 (Beryl→Template): ~94秒
│  ├─ サイクル1（メッシュ変形）: ~14秒
│  ├─ サイクル2前処理: ~5秒
│  ├─ サイクル2（ウェイト転送）: ~70秒 ★主要ボトルネック
│  └─ FBXエクスポート: ~2秒
│
└─ ペア2 (Template→mao): ~185秒
   ├─ サイクル1: ~14秒
   ├─ サイクル2前処理: ~5秒
   ├─ サイクル2（ウェイト転送）: ~160秒 ★主要ボトルネック
   └─ FBXエクスポート: ~2秒
```

### 2.3 ボトルネック詳細分析

#### 2.3.1 スムージング処理（smoothing_processor.py）

**現在の実装:**
```python
def apply_smoothing_sequential(vertex_coords, current_weights, kdtree, ...):
    num_vertices = len(vertex_coords)
    smoothed_weights = np.zeros(num_vertices, dtype=np.float32)

    for i in range(num_vertices):  # ★ Pythonループ（遅い）
        neighbor_indices = kdtree.query_ball_point(vertex_coords[i], smoothing_radius)

        if len(neighbor_indices) > 1:
            neighbor_coords = vertex_coords[neighbor_indices]
            distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)  # ★ 毎回計算
            # ...
```

**問題点:**
1. Pythonの `for` ループによる逐次処理
2. 各頂点で `np.linalg.norm()` を個別に呼び出し
3. 各プロセスで `cKDTree` を再構築

**処理時間の内訳（例: TempMerged_Costume_Gloves）:**
| 処理 | 時間 |
|------|------|
| スムージング並列処理 | 23.50秒 |
| ウェイト転送全体 | 46.68秒 |

#### 2.3.2 ウェイト転送処理（retarget_script2_14.py）

**現在の実装:**
```python
def transfer_weights_from_nearest_vertex(base_mesh, target_obj, ...):
    # BVHツリー作成
    bvh_tree = BVHTree.FromBMesh(body_bm)

    # 衣装メッシュの各頂点の法線処理
    for i, vertex in enumerate(cloth_bm.verts):  # ★ Pythonループ
        cloth_vert_world = vertex.co
        original_normal_world = (cloth_normal_matrix @ Vector(...)).xyz.normalized()

        nearest_result = bvh_tree.find_nearest(cloth_vert_world)
        # ...
```

**問題点:**
1. 頂点ごとのPythonループ
2. 毎回 `Vector` オブジェクトを生成
3. 法線変換の繰り返し計算

---

## 3. 最適化施策

### 3.1 Phase 1: 低リスク・即効性のある改善

#### P1-1: Numba JIT による距離計算高速化

**対象ファイル:** `smoothing_processor.py`

**現在の実装:**
```python
distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
```

**最適化後:**
```python
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True)
def compute_distances_batch(vertices, target_points):
    """Numba JIT版 距離計算"""
    n, m = len(vertices), len(target_points)
    distances = np.empty((n, m), dtype=np.float32)

    for i in prange(n):
        for j in range(m):
            dist = 0.0
            for k in range(3):
                d = vertices[i, k] - target_points[j, k]
                dist += d * d
            distances[i, j] = np.sqrt(dist)

    return distances
```

**期待効果:** +100-300%（距離計算部分）
**リスク:** 低（Numba未インストール時はフォールバック）
**実装難度:** 低

---

#### P1-2: スムージング処理のベクトル化

**対象ファイル:** `smoothing_processor.py`

**現在の実装:**
```python
for i in range(num_vertices):
    neighbor_indices = kdtree.query_ball_point(vertex_coords[i], smoothing_radius)
    # 個別処理...
```

**最適化後:**
```python
def apply_smoothing_vectorized(vertex_coords, current_weights, smoothing_radius, ...):
    """ベクトル化されたスムージング処理"""
    kdtree = cKDTree(vertex_coords)

    # 全頂点の近傍を一括取得
    all_neighbors = kdtree.query_ball_tree(kdtree, smoothing_radius)

    # NumPyベクトル演算で一括処理
    smoothed_weights = np.zeros(len(vertex_coords), dtype=np.float32)

    for i, neighbors in enumerate(all_neighbors):
        if len(neighbors) > 1:
            neighbor_coords = vertex_coords[neighbors]
            distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
            weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
            smoothed_weights[i] = np.dot(current_weights[neighbors], weights) / np.sum(weights)
        else:
            smoothed_weights[i] = current_weights[i]

    return smoothed_weights
```

**期待効果:** +30-50%
**リスク:** 低
**実装難度:** 低

---

#### P1-3: KDTree共有による再構築回避

**対象ファイル:** `smoothing_processor.py`

**現在の実装:**
```python
def process_vertex_batch(args):
    # プロセス内でcKDTreeを構築（毎回）
    kdtree = cKDTree(vertex_coords)
```

**最適化後:**
```python
# グローバルにKDTreeをキャッシュ
_kdtree_cache = {}

def get_cached_kdtree(vertex_coords_hash, vertex_coords):
    if vertex_coords_hash not in _kdtree_cache:
        _kdtree_cache[vertex_coords_hash] = cKDTree(vertex_coords)
    return _kdtree_cache[vertex_coords_hash]
```

**期待効果:** +20-30%（KDTree構築コスト削減）
**リスク:** 低（メモリ使用量増加に注意）
**実装難度:** 低

---

### 3.2 Phase 2: 中リスク・アルゴリズム改善

#### P2-1: バッチ処理の最適化

**対象ファイル:** `smoothing_processor.py`

**現在の実装:**
- 固定バッチサイズ: 1,000頂点
- 各バッチでKDTreeを再構築

**最適化後:**
```python
def calculate_optimal_batch_size(num_vertices, num_workers, available_memory_gb):
    """動的バッチサイズ計算"""
    # メモリと頂点数に基づいて最適化
    bytes_per_vertex = 3 * 4 + 4  # coords(float32×3) + weight(float32)
    max_batch_by_memory = int(available_memory_gb * 0.5 * 1024**3 / bytes_per_vertex / num_workers)

    # 最適なバッチサイズを計算
    optimal = min(max(5000, num_vertices // (num_workers * 2)), max_batch_by_memory, 20000)
    return optimal
```

**期待効果:** +20-40%
**リスク:** 中
**実装難度:** 中

---

#### P2-2: 法線計算の事前バッチ処理

**対象ファイル:** `retarget_script2_14.py`

**現在の実装:**
```python
for i, vertex in enumerate(cloth_bm.verts):
    original_normal_world = (cloth_normal_matrix @ Vector(...)).xyz.normalized()
```

**最適化後:**
```python
# 事前に全頂点の法線を一括計算
def precompute_normals_batch(vertices, normal_matrix):
    """NumPyで法線を一括変換"""
    normals = np.array([v.normal[:] for v in vertices], dtype=np.float32)
    # 行列変換を一括適用
    transformed = (normal_matrix @ np.hstack([normals, np.zeros((len(normals), 1))]).T).T[:, :3]
    # 正規化
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    return transformed / np.maximum(norms, 1e-8)
```

**期待効果:** +30-50%（法線計算部分）
**リスク:** 中
**実装難度:** 中

---

#### P2-3: 増分更新による再計算回避

**対象ファイル:** `retarget_script2_14.py`

**概要:** 変更された頂点のみ再計算

```python
def incremental_weight_update(changed_vertices, bvh_tree, ...):
    """変更頂点のみウェイト更新"""
    for v_idx in changed_vertices:
        # 該当頂点のみ処理
        update_single_vertex_weight(v_idx, bvh_tree, ...)
```

**期待効果:** +30-50%（変更箇所依存）
**リスク:** 中（正確な変更検出が必要）
**実装難度:** 中

---

### 3.3 Phase 3: GPU加速（将来検討）

| 手法 | 期待効果 | 前提条件 | リスク |
|------|----------|----------|--------|
| CuPy | +200-500% | NVIDIA GPU + CUDA | 高 |
| PyTorch | +100-300% | GPU または CPU | 中 |
| OpenCL | +100-200% | 任意の GPU | 高 |

**現時点では Phase 1-2 に注力し、GPU化は将来的なオプションとする。**

---

## 4. 実装ロードマップ

### 4.1 スケジュール

```
Phase 1 (低リスク・即効):
├─ [ ] P1-1: Numba JIT による距離計算高速化
├─ [ ] P1-2: スムージング処理のベクトル化
├─ [ ] P1-3: KDTree共有による再構築回避
└─ [ ] ベンチマーク・回帰テスト

Phase 2 (中リスク・アルゴリズム改善):
├─ [ ] P2-1: バッチ処理の最適化
├─ [ ] P2-2: 法線計算の事前バッチ処理
├─ [ ] P2-3: 増分更新による再計算回避
└─ [ ] 統合テスト・ベンチマーク

Phase 3 (GPU加速・将来検討):
├─ [ ] GPU加速の実現可能性調査
├─ [ ] プロトタイプ実装
└─ [ ] 互換性テスト
```

### 4.2 マイルストーン

| マイルストーン | 目標時間 | 達成基準 |
|---------------|---------|---------|
| M1: Phase 1完了 | - | 287秒 → 180-200秒 |
| M2: Phase 2完了 | - | → 120-150秒 |
| M3: Phase 3検証 | - | GPU環境での動作確認 |

---

## 5. 依存関係とリスク

### 5.1 新規依存パッケージ

| パッケージ | 用途 | 必須/オプション |
|-----------|------|----------------|
| numba | JITコンパイル | オプション |
| (cupy) | GPU加速 | オプション（Phase 3） |

### 5.2 フォールバック戦略

```python
# Numbaが利用できない場合のフォールバック
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    prange = range

@jit(nopython=True, parallel=True)
def compute_distances(A, B):
    # Numbaがあれば高速化、なければ通常のPython
    ...
```

### 5.3 リスク管理

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|---------|------|
| Numba互換性問題 | 低 | 中 | フォールバック実装 |
| メモリ使用量増加 | 中 | 中 | キャッシュサイズ制限 |
| 精度低下 | 高 | 低 | float64フォールバック |
| 並列処理デッドロック | 高 | 低 | タイムアウト設定 |

---

## 6. ベンチマーク手法

### 6.1 テスト環境

```yaml
標準テスト環境:
  CPU: マルチコア（推奨8コア以上）
  RAM: 32GB以上
  OS: Windows 10/11
  Blender: 4.0.2
  Python: 3.10+

テストケース:
  - Beryl → Template → mao（チェーン処理）
  - 衣装メッシュ: 7個
  - 総処理時間: 280-350秒
```

### 6.2 測定項目

| 項目 | 測定方法 | 目標 |
|------|---------|------|
| 処理時間 | `time.time()` | 各フェーズ個別 + 合計 |
| メモリ使用量 | `psutil.Process().memory_info()` | ピーク値 |
| CPU使用率 | `psutil.cpu_percent()` | 平均値 |
| 精度 | 出力FBXの比較 | 視覚的差異なし |

### 6.3 ベンチマークスクリプト

```bash
# CLI実行
python run_retarget.py --preset beryl_to_mao --benchmark

# 結果は Outputs/benchmark_results.json に保存
```

---

## 7. 参考資料

### 7.1 関連ドキュメント

- [RBF処理 高速化・最適化 実装計画](../blender-addon/rbf-optimization-plan.md)
- [リファクタリング計画](../refactoring/retarget_script_refactor_plan.md)

### 7.2 外部リソース

- [Numba ドキュメント](https://numba.readthedocs.io/)
- [SciPy cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)
- [CuPy ドキュメント](https://docs.cupy.dev/)

---

## 更新履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-12-30 | 1.0 | 初版作成 |
