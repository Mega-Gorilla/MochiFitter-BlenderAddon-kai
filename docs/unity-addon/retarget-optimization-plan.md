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
| Phase 1 完了（保守的） | 287秒 | 150-200秒 | 30-50% |
| Phase 1 完了（楽観的） | 287秒 | 100-150秒 | 50-65% |
| Phase 2 完了 | - | 80-120秒 | 60-70% |
| (Phase 3 GPU) | - | 40-80秒 | 70-85% |

> ⚠️ **注意（レビュアー指摘 2025-12-30）:**
> - ミクロベンチマーク（~20x）は実処理全体への寄与率と異なる
> - foreach_get は Mesh API 向けであり、BMesh 箇所（約35箇所）には適用不可
> - 配列確保コストが支配的になりやすいため、事前確保が必要
> - **フェーズ別の実測ログで効果を確認してから目標を再評価すること**

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
| 2025-12-30 14:52 | Phase 3.2 | 298.08秒 | 関数分離（PR#45） |
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

> **優先順位**: ベンチマーク結果に基づき、効果の高い順に並べ替え済み（2025-12-30 更新）

#### P1-0: foreach_get + NumPy batch 一括変換 ★★★最優先

**対象ファイル:** `retarget_script2_14.py`

**レビュアー指摘により追加**: 頂点ごとのPythonループ内で Vector 生成や行列変換を繰り返している箇所を配列化→一括変換に寄せる。

> ⚠️ **重要な制約（レビュアー指摘 2025-12-30）:**
> - `foreach_get` は **Mesh API 向け**であり、**BMesh には適用不可**
> - BMesh で回している箇所は直接置き換えできない
> - 「Meshで取得→NumPy処理→必要最小限だけBMeshへ戻す」方針が安全

**適用可能箇所の分類:**

| 分類 | 対象 | foreach_get | 箇所数 |
|------|------|-------------|--------|
| A: Mesh API | `mesh.vertices`, `evaluated_mesh.vertices` | ✅ 適用可能 | 約50箇所 |
| B: BMesh | `bm.verts`, `cloth_bm.verts` | ❌ 不可 | 約35箇所 |
| C: NumPy配列 | すでに NumPy 配列のデータ | - (行列演算のみ) | 約15箇所 |

**現在の実装（遅い）:**
```python
# パターン1: Mesh API + 行列変換 ← foreach_get 適用可能
vertices_world = np.array([matrix @ v.co for v in mesh.vertices])

# パターン2: BMesh + 行列変換 ← foreach_get 不可、行列演算のみ最適化
coords = np.array([v.co for v in bm.verts])
transformed = np.array([matrix @ Vector(v) for v in coords])

# パターン3: NumPy配列 + 行列変換 ← 行列演算のみ最適化
transformed = np.array([matrix @ Vector(v) for v in vertices])
```

**最適化後（高速）:**
```python
# パターン1: Mesh API → foreach_get + NumPy batch（10-20x 高速）
num_verts = len(mesh.vertices)
coords = np.empty(num_verts * 3, dtype=np.float64)
mesh.vertices.foreach_get("co", coords)
coords = coords.reshape(-1, 3)
rotation = np.array(matrix)[:3, :3]
translation = np.array(matrix)[:3, 3]
vertices_world = coords @ rotation.T + translation

# パターン2: BMesh → リスト内包 + NumPy batch（行列演算部分のみ高速化）
coords = np.array([v.co for v in bm.verts], dtype=np.float64)  # BMeshはリスト内包のまま
vertices_world = coords @ rotation.T + translation  # 行列演算は一括化

# パターン3: NumPy配列 → NumPy batch のみ（100-250x 高速）
# 配列の事前確保（繰り返し処理で重要）
vertices_world = coords @ rotation.T + translation
```

**配列事前確保パターン（繰り返し処理で重要）:**
```python
# 悪い例: 毎回配列を確保
for i in range(iterations):
    coords = np.empty(num_verts * 3, dtype=np.float64)  # 毎回確保 ← オーバーヘッド
    mesh.vertices.foreach_get("co", coords)

# 良い例: 事前確保して再利用
coords = np.empty(num_verts * 3, dtype=np.float64)  # 1回だけ確保
result = np.empty((num_verts, 3), dtype=np.float64)  # 結果も事前確保
for i in range(iterations):
    mesh.vertices.foreach_get("co", coords)  # 再利用
    np.dot(coords.reshape(-1, 3), rotation.T, out=result)  # in-place演算
    result += translation
```

**期待効果（修正版）:**
- **Mesh API 箇所（約50箇所）**: 10-20x 高速化
- **BMesh 箇所（約35箇所）**: 行列演算部分のみ 2-5x 高速化（foreach_get 不可）
- **全体への寄与率**: 処理内容により異なる（後述の注意参照）

**リスク:** 低（Blender標準API、追加依存なし）
**実装難度:** 中（適用箇所が多い、BMesh/Mesh の区別が必要）

> **ベンチマーク結果（2025-12-30 実測）- Mesh API 箇所:**
> | 頂点数 | 現行パターン | foreach_get + NumPy | **Speedup** |
> |--------|-------------|---------------------|-------------|
> | 10,000 | 18.83 ms | 0.91 ms | **20.76x** |
> | 30,000 | 59.98 ms | 2.79 ms | **21.46x** |
> | 100,000 | 229.76 ms | 11.76 ms | **19.53x** |
>
> ⚠️ **注意**: 上記はミクロベンチマーク。実処理全体での寄与率は別途検証が必要。

**主な適用箇所（retarget_script2_14.py）:**

| 分類 | 行番号 | パターン | 適用可能 |
|------|--------|----------|----------|
| A | 1483 | `[matrix @ v.co for v in mesh.vertices]` | foreach_get ✅ |
| A | 833, 3858, 5427 | `[v.co for v in eval_mesh.vertices]` | foreach_get ✅ |
| B | 4242, 16085 | `[v.co for v in cloth_bm.verts]` | 行列演算のみ |
| B | 16349-16350 | `[v.co for v in cloth_bm.verts]` + normals | 行列演算のみ |
| C | 882, 5969 | `[matrix @ Vector(v) for v in vertices]` | 行列演算のみ |

---

#### P1-1: KDTree共有による再構築回避

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

**期待効果:** +40-65%（実測: 10k頂点で1.66x、30k頂点で1.44x、3イテレーション時）
**リスク:** 低（メモリ使用量増加に注意）
**実装難度:** 低

> **Note**: イテレーション数が多いほど効果大。最も効果が高く実装リスクが低いため、**最優先で実装推奨**。

---

#### P1-2: Numba JIT による距離計算高速化

**対象ファイル:** `smoothing_processor.py`

**現在の実装:**
```python
distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
```

**最適化後:**
```python
from numba import jit

@jit(nopython=True, fastmath=True)
def compute_distances_from_point(center, neighbor_coords):
    """Numba JIT版: 1頂点から近傍点群への距離計算

    スムージング処理では各頂点について、KDTreeで取得した近傍点との距離を計算する。
    この関数は1頂点(center)から近傍点群(neighbor_coords)への距離配列を返す。
    """
    n = len(neighbor_coords)
    distances = np.empty(n, dtype=np.float32)

    for j in range(n):
        dist = 0.0
        for k in range(3):
            d = center[k] - neighbor_coords[j, k]
            dist += d * d
        distances[j] = np.sqrt(dist)

    return distances
```

> **Note**: この関数は `kdtree.query_ball_point()` で取得した近傍点に対して使用します。
> 全頂点×全頂点の距離行列を計算するのではなく、各頂点の近傍のみを処理します。

**期待効果:** +20-30%（実測: 10k頂点で1.30x、30k頂点で1.24x）
**リスク:** 低（Numba未インストール時はフォールバック）
**実装難度:** 低

---

#### P1-3: スムージング処理のベクトル化（条件付き採用）

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

**期待効果:** +0-35%（実測: 10k頂点で1.33x、30k頂点で1.04x）
**リスク:** 低
**実装難度:** 低

> ⚠️ **条件付き運用（レビュアー指摘 2025-12-30）:**
>
> `query_ball_tree` は一括取得のオーバーヘッドにより、**大規模データでは逆効果になる可能性があります。**
>
> **採用条件（以下のいずれかを満たす場合のみ）:**
> 1. **小規模限定**: 頂点数が 10,000 以下のメッシュ
> 2. **k近傍制限**: `query_ball_tree` の代わりに `query(k=N)` を使用し、固定数の近傍のみを取得
>
> **実装時の判断ロジック:**
> ```python
> QUERY_BALL_TREE_THRESHOLD = 10000  # この頂点数以下でのみ使用
>
> if num_vertices <= QUERY_BALL_TREE_THRESHOLD:
>     # 小規模: query_ball_tree で一括取得
>     all_neighbors = kdtree.query_ball_tree(kdtree, radius)
> else:
>     # 大規模: 従来の query_ball_point ループ（または k近傍）
>     for i in range(num_vertices):
>         neighbors = kdtree.query_ball_point(vertex_coords[i], radius)
> ```
>
> **代替案（k近傍）:**
> ```python
> # 半径ベースではなく、固定数の近傍を取得
> MAX_NEIGHBORS = 32  # 近傍数の上限
> distances, indices = kdtree.query(vertex_coords, k=MAX_NEIGHBORS)
> ```

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

## 3.4 Phase 1 最適化ベンチマーク結果（2025-12-30 実測）

### 3.4.1 foreach_get + NumPy batch ベンチマーク

`tests/foreach_get_benchmark.py` を使用して Blender 4.0.2 で実測。

#### Vertex Extraction: foreach_get vs list comprehension

| 頂点数 | list_comprehension | foreach_get | **Speedup** |
|--------|-------------------|-------------|-------------|
| 10,000 | 2.25 ms | 0.15 ms | **15.15x** |
| 30,000 | 6.93 ms | 0.43 ms | **16.05x** |
| 100,000 | 25.51 ms | 1.82 ms | **14.02x** |

#### Matrix Transformation: Loop vs NumPy batch

| 頂点数 | loop_vector | numpy_batch | **Speedup** |
|--------|-------------|-------------|-------------|
| 10,000 | 25.48 ms | 0.10 ms | **244x** |
| 30,000 | 76.63 ms | 0.31 ms | **247x** |
| 100,000 | 302.15 ms | 1.82 ms | **166x** |

#### Combined (Real-world pattern)

| 頂点数 | 現行パターン | foreach_get + NumPy | **Speedup** |
|--------|-------------|---------------------|-------------|
| 10,000 | 18.83 ms | 0.91 ms | **20.76x** |
| 30,000 | 59.98 ms | 2.79 ms | **21.46x** |
| 100,000 | 229.76 ms | 11.76 ms | **19.53x** |

### 3.4.2 スムージング処理ベンチマーク

`tests/optimization_benchmark.py` を使用して実測したベンチマーク結果。

### テスト環境

- データ: `deformation_beryl_to_template.npz` から抽出した実頂点データ
- Python: Blender 4.0.2 付属 Python 3.11
- Numba: 0.61.0（pip install で追加）

### 10,000 頂点

| 最適化 | 処理時間 | Speedup | MSE | Max Error |
|--------|----------|---------|-----|-----------|
| Baseline | 481.68 ms | - | - | - |
| P1-1: Numba JIT | 370.12 ms | **1.30x** | 7.26e-15 | 4.77e-07 |
| P1-2: query_ball_tree | 360.92 ms | **1.33x** | 1.09e-14 | 5.96e-07 |
| P1-3: KDTree rebuild (3回) | 1418.06 ms | - | - | - |
| P1-3: KDTree shared (3回) | 853.16 ms | **1.66x** | - | - |

### 30,000 頂点

| 最適化 | 処理時間 | Speedup | MSE | Max Error |
|--------|----------|---------|-----|-----------|
| Baseline | 2273.79 ms | - | - | - |
| P1-1: Numba JIT | 1836.99 ms | **1.24x** | 1.71e-14 | 8.94e-07 |
| P1-2: query_ball_tree | 2195.88 ms | **1.04x** | 2.47e-14 | 1.13e-06 |
| P1-3: KDTree rebuild (3回) | 6727.60 ms | - | - | - |
| P1-3: KDTree shared (3回) | 4681.66 ms | **1.44x** | - | - |

### 3.4.3 考察

1. **P1-0 (foreach_get + NumPy batch)**: **約20倍高速化**。最も効果が大きく、追加依存なし。レビュアー指摘通り**最優先**で実装すべき。
2. **P1-1 (KDTree caching)**: 複数イテレーションで **44-66% 高速化**。イテレーション数が多いほど効果大。
3. **P1-2 (Numba JIT)**: 安定して **20-30% 高速化**。距離計算のJITコンパイルが効果的。オプション依存。
4. **P1-3 (query_ball_tree)**: 小規模データでは効果的だが、**大規模データでは効果が薄れる**（一括取得のオーバーヘッド）。
5. **精度**: MSE は 10^-14 〜 10^-16 レベルで、**実質的に精度劣化なし**。

### 3.4.4 推奨実装優先順位

| 優先度 | 施策 | 効果 | 依存 | リスク |
|--------|------|------|------|--------|
| ★★★ | **P1-0: foreach_get + NumPy batch** | **~20x** | なし | 低 |
| ★★ | P1-1: KDTree caching | +40-65% | なし | 低 |
| ★ | P1-2: Numba JIT | +20-30% | numba | 低 |
| - | P1-3: query_ball_tree | +0-35% | なし | 低 |

> **Note**: P1-0 だけで大幅な高速化が期待できるため、まずこれに集中し、効果を実測してから P1-1 以降を検討することを推奨。

---

## 4. 実装ロードマップ

### 4.1 スケジュール

```
Phase 1 (低リスク・即効):
├─ [ ] P1-0: foreach_get + NumPy batch ★★★最優先 (+1900-2100%)
├─ [ ] P1-1: KDTree共有による再構築回避 (+40-65%)
├─ [ ] P1-2: Numba JIT による距離計算高速化（オプション）(+20-30%)
├─ [ ] P1-3: スムージング処理のベクトル化（限定的）(+0-35%)
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
| psutil | メモリ/CPU使用量の計測 | オプション（未インストール時は計測スキップ） |
| (cupy) | GPU加速 | オプション（Phase 3） |

### 5.2 Numba 互換性調査結果（2025-12-30）

#### 5.2.1 Blender 4.0.2 環境との互換性

| 項目 | Blender 4.0.2 | Numba 0.61.0 要件 | 互換性 |
|------|---------------|-------------------|--------|
| Python | 3.11 | 3.10 - 3.13 | ✅ |
| NumPy | 1.24.3 | 1.24 - 2.1 | ✅ |

**結論: Blender 4.0.2 環境では Numba との互換性あり**

#### 5.2.2 インストール方法

pip wheel に LLVM がバンドルされているため、システム LLVM のインストールは不要:

```bash
# Blender付属Pythonへのインストール（Unity アドオン同梱 Blender の場合）
BlenderTools/blender-4.0.2-windows-x64/4.0/python/bin/python.exe -m pip install numba
```

#### 5.2.3 過去の問題と現在の状況

| 時期 | Blender | 状況 |
|------|---------|------|
| 2022年 | 3.1 | NumPy/Python/Numba の三すくみ問題（解決済み） |
| 現在 | 4.0.2 | **互換性あり、pip install 可能** |
| 現在 | 4.2+ | 一部で問題報告あり（原因調査中） |

> **参考資料:**
> - [Numba Installation Documentation](https://numba.readthedocs.io/en/stable/user/installing.html)
> - [Tissue #132 - Numba/NumPy compatibility](https://github.com/alessandro-zomparelli/tissue/issues/132)
> - [Blender Artists - Tissue numba issues](https://blenderartists.org/t/tissue-add-on-problem-installing-numba-blender-4-2lts/1605607)

#### 5.2.4 推奨方針

Blender 4.0.2 では互換性が確認されたため、Numba をオプション依存として採用可能。
ただし、**フォールバック設計は必須**（ユーザー環境差異への対応）。

### 5.3 フォールバック戦略

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

### 5.4 BMesh/並列処理の制約

**重要: BMesh はスレッドセーフではありません。**

並列処理を行う場合は、以下の設計パターンを守る必要があります:

```
┌─────────────────────────────────────────────────────────────┐
│ メインスレッド                                                │
│   ├─ BMesh → NumPy配列に抽出                                │
│   ├─ ワーカープロセス起動（NumPy配列のみ渡す）                 │
│   │     └─ NumPy演算のみ（BMesh操作禁止）                    │
│   └─ NumPy配列 → BMeshに書き戻し                            │
└─────────────────────────────────────────────────────────────┘
```

`smoothing_processor.py` は既にこの設計に従っています。

### 5.5 foreach_get の活用（前提条件）

Blender API → NumPy 変換は `foreach_get` を使用することで高速化できます:

```python
# 遅い（Pythonループ）- 避けるべき
coords = np.array([v.co[:] for v in mesh.vertices])

# 速い（foreach_get）- 10-50倍高速
coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
mesh.vertices.foreach_get('co', coords)
coords = coords.reshape(-1, 3)

# 法線も同様
normals = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
mesh.vertices.foreach_get('normal', normals)
normals = normals.reshape(-1, 3)
```

**Phase 1-2 の最適化は `foreach_get` の使用を前提としています。**

### 5.6 リスク管理

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|---------|------|
| Numba互換性問題 | 低 | 低〜中 | フォールバック実装 |
| BMesh並列アクセス | 高 | 中 | メインスレッドでのみBMesh操作 |
| foreach_get未使用 | 中 | 中 | コードレビューで確認 |
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
| メモリ使用量 | `psutil.Process().memory_info()` | ピーク値（psutil がない場合はスキップ） |
| CPU使用率 | `psutil.cpu_percent()` | 平均値（psutil がない場合はスキップ） |
| 精度 | 出力FBXの比較 | 視覚的差異なし |

### 6.3 ベンチマーク実行手順

ベンチマークは Unity アドオンを通じて、または Blender CLI で直接実行できます。

**Blender CLI 実行例（チェーン処理）:**

```bash
blender --background --python retarget_script2_14.py -- \
    --input="Beryl_Costumes.fbx" \
    --output="benchmark_output.fbx" \
    --base="base_project.blend" \
    --base-fbx="Template.fbx;mao.fbx" \
    --config="config_beryl2template.json;config_template2mao.json" \
    --init-pose="initial_pose.json" \
    --hips-position="0.00000000,0.00955725,0.93028500" \
    --target-meshes="Costume_Body;Costume_Socks;HighHeel"
```

> **Note**: 詳細なパラメータは [Execute Retargeting コマンド詳細](execute_retargeting.md) を参照してください。
> 処理時間はスクリプト内部で計測され、コンソールに出力されます。

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
| 2025-12-30 | 1.1 | PR #49 レビュー対応: ベンチマーク表追加、compute_distances修正、psutil依存追記、ベンチマーク手順修正 |
| 2025-12-30 | 1.2 | Numba互換性調査結果追加、BMesh並列処理制約、foreach_get前提条件を明記 |
| 2025-12-30 | 1.3 | Phase 1 最適化ベンチマーク実測結果追加（10k/30k頂点、Numba JIT、query_ball_tree、KDTree caching）|
| 2025-12-30 | 1.4 | 各最適化施策の「期待効果」を実測値に基づいて修正、優先度順に並び替え |
| 2025-12-30 | 1.5 | **P1-0: foreach_get + NumPy batch 追加（~20x効果）**、レビュアー意見に基づく再調査、目標時間上方修正 |
| 2025-12-30 | 1.6 | レビュアー追加指摘対応: BMesh/Mesh API 分類表、配列事前確保パターン、目標時間保守的見直し、query_ball_tree 条件付き運用を明記 |
