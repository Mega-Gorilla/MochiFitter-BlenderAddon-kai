# MochiFitter データフォーマット仕様

## 概要

MochiFitterは以下のデータファイルを使用します：

| ファイル種別 | 拡張子 | 用途 |
|------------|-------|------|
| avatar_data | .json | ボーン階層、Humanoidマッピング |
| pose_basis | .json | アバターのベースポーズ |
| posediff | .json | 2アバター間のポーズ差分 |
| deformation | .npz | 変形フィールドデータ |

## avatar_data_*.json

アバターのボーン構造とHumanoidボーンマッピングを定義します。

### 構造

```json
{
  "boneHierarchy": {
    "name": "Hips",
    "children": [
      {
        "name": "Spine",
        "children": [...]
      },
      {
        "name": "LeftUpperLeg",
        "children": [...]
      }
    ]
  },
  "humanoidBones": [
    {
      "humanoidBoneName": "Hips",
      "boneName": "Hips"
    },
    {
      "humanoidBoneName": "Spine",
      "boneName": "Spine"
    }
  ]
}
```

### フィールド説明

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `boneHierarchy` | object | ボーンの階層構造（ルートから再帰的） |
| `boneHierarchy.name` | string | ボーン名 |
| `boneHierarchy.children` | array | 子ボーンのリスト |
| `humanoidBones` | array | Humanoidボーンマッピングのリスト |
| `humanoidBones[].humanoidBoneName` | string | Unity Humanoidの標準ボーン名 |
| `humanoidBones[].boneName` | string | 実際のボーン名 |

## pose_basis_*.json

アバターのベースポーズ（レストポーズからの変形）を保存します。

### 構造

```json
{
  "Hips": {
    "location": [0.0, 0.0, 0.0],
    "rotation": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0],
    "head_world": [0.0, 0.009557, 0.930285],
    "head_world_transformed": [0.0, 0.009557, 0.930285]
  },
  "Spine": {
    ...
  }
}
```

### フィールド説明

| フィールド | 型 | 単位 | 説明 |
|-----------|-----|------|------|
| `location` | [x, y, z] | メートル | ボーンヘッドの移動量 |
| `rotation` | [x, y, z] | 度 | オイラー角（XYZ順） |
| `scale` | [x, y, z] | 倍率 | スケール |
| `head_world` | [x, y, z] | メートル | 変形前のボーンヘッド位置（ワールド座標） |
| `head_world_transformed` | [x, y, z] | メートル | 変形後のボーンヘッド位置（ワールド座標） |

## posediff_*.json

2つのアバター間のポーズ差分を保存します。

### 新形式（MochiFitter-Kai v0.1.0以降）

```json
{
  "Hips": {
    "location": [0.0, 0.0, 0.07627],
    "rotation": [0.0, 0.0, 0.0],
    "scale": [1.3, 1.3, 1.3],
    "head_world": [0.0, -0.00477, 0.8476],
    "head_world_transformed": [0.0, -0.00477, 0.9238]
  }
}
```

### 旧形式（delta_matrix含む）

```json
{
  "Hips": {
    "location": [0.0, -0.0157, -0.0207],
    "rotation": [-0.00000278, 0.0, 0.0],
    "scale": [1.0169, 1.0169, 1.0169],
    "head_world": [0.0, 0.00956, 0.9303],
    "head_world_transformed": [0.0, -0.00599, 0.9252],
    "delta_matrix": [
      [1.0169, 0.0, 0.0, 0.0],
      [0.0, 1.0169, 0.0, -0.0157],
      [0.0, 0.0, 1.0169, -0.0207],
      [0.0, 0.0, 0.0, 1.0]
    ]
  }
}
```

### フィールド説明

| フィールド | 型 | 単位 | 説明 |
|-----------|-----|------|------|
| `location` | [x, y, z] | メートル | ボーンヘッドの移動量 |
| `rotation` | [x, y, z] | 度（新形式）/ラジアン（旧形式） | オイラー角（XYZ順） |
| `scale` | [x, y, z] | 倍率 | スケール |
| `head_world` | [x, y, z] | メートル | ソースアバターのボーンヘッド位置 |
| `head_world_transformed` | [x, y, z] | メートル | ターゲットアバターのボーンヘッド位置 |
| `delta_matrix` | 4x4 matrix | - | 変換行列（旧形式のみ） |

### 新形式 vs 旧形式

| 項目 | 新形式 | 旧形式 |
|------|--------|--------|
| rotation単位 | 度 | ラジアン |
| delta_matrix | なし | あり |
| 編集の容易さ | 高い | 低い |
| 精度 | 再構築による誤差あり | 行列直接使用で高精度 |

**読み込み時の優先順位**（v0.1.0以降）:
1. `delta_matrix` が存在 → 直接使用（最も正確）
2. `delta_matrix` なし → `location/rotation/scale` から行列を再構築

> **注意**: `delta_matrix`がなく、`rotation`がラジアン単位の旧形式JSONは
> 正しく読み込めません（新形式と誤認され度として解釈されるため）。
> 旧形式JSONには必ず`delta_matrix`が含まれている必要があります。

## deformation_*.npz

RBF補間による変形フィールドデータを保存します（NumPy圧縮形式）。

### 構造

```python
{
    'all_field_points': array,      # 各ステップのフィールド座標
    'all_delta_positions': array,   # 各ステップの変位ベクトル
    'num_steps': int,               # ステップ数
    'world_matrix': array,          # ワールド行列（現在は単位行列）
    'kdtree_query_k': int,          # KDTree検索のk値
    'rbf_epsilon': float,           # RBFのεパラメータ
    'rbf_smoothing': float,         # スムージングパラメータ
    'enable_x_mirror': bool         # Xミラーリング有効フラグ（オプション）
}
```

### フィールド説明

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `all_field_points` | ndarray | フィールドグリッドの3D座標（shape: [num_steps, num_points, 3]） |
| `all_delta_positions` | ndarray | 各点での変位ベクトル（shape: [num_steps, num_points, 3]） |
| `num_steps` | int | 変形を分割したステップ数 |
| `world_matrix` | ndarray | 4x4のワールド変換行列 |
| `kdtree_query_k` | int | 最近傍検索で使用する点の数（デフォルト: 27）※現在未使用 |
| `rbf_epsilon` | float | RBFカーネルのεパラメータ |
| `rbf_smoothing` | float | スムージング係数 ※現在未使用 |
| `enable_x_mirror` | bool | Xミラーリング有効フラグ（オプション、デフォルト: False） |

> **注意**: `kdtree_query_k`と`rbf_smoothing`は保存されますが、現在の`apply_field_data()`では
> 参照されていません（近傍点数は`k = min(8, len(field_points))`で固定）。
> これらは将来の拡張用、またはマルチスレッドプロセッサ用のパラメータです。

### Xミラーリング

`rbf_enable_x_mirror=True` の場合：
- 保存時: X座標が0以上のデータのみ保存（NPZキー: `enable_x_mirror=True`）
- 読み込み時: `enable_x_mirror`フラグを確認し、自動的にX軸でミラーリング

## データフロー図

```
┌─────────────────────────────────────────────────────────────────┐
│                        データ生成フロー                          │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐
│ Avatar A     │     │ Avatar B     │
│ (Armature)   │     │ (Armature)   │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│pose_basis_   │     │pose_basis_   │
│A.json        │     │B.json        │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
                ▼
        ┌──────────────┐
        │posediff_     │
        │A_to_B.json   │
        └──────────────┘


┌──────────────────────────────────────────────────────────────────┐
│                      変形フィールド生成フロー                      │
└──────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐
│ Source Mesh  │     │ ShapeKey     │
│ (Basis)      │     │ (Target)     │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
                ▼
        ┌──────────────────┐
        │ 制御点抽出        │
        │ (before/after)   │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ RBF補間          │
        │ (Multi-Quadratic │
        │  Biharmonic)     │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ deformation_     │
        │ A_to_B.npz       │
        └──────────────────┘
```

## ファイル命名規則

| 種別 | パターン | 例 |
|------|---------|-----|
| アバターデータ | `avatar_data_{avatar_name}.json` | `avatar_data_template.json` |
| ベースポーズ | `pose_basis_{avatar_name}.json` | `pose_basis_beryl.json` |
| ポーズ差分 | `posediff_{source}_to_{target}.json` | `posediff_beryl_to_template.json` |
| 変形フィールド | `deformation_{source}_to_{target}.npz` | `deformation_beryl_to_template.npz` |
| 逆変形フィールド | `deformation_{source}_to_{target}_inv.npz` | `deformation_beryl_to_template_inv.npz` |
| シェイプキー変形 | `deformation_{avatar}_shape_{key}.npz` | `deformation_template_shape_Breasts.npz` |

## 関連ドキュメント

- [アーキテクチャ概要](overview.md)
- [オペレーター詳細](operators.md)
