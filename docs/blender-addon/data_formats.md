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

ポーズ情報を保存します。用途によって2種類の形式があります。

### 用途1: 基準アバターの Rest Pose

`pose_basis_template.json` など、基準となるアバター（Template）の Rest Pose を保存します。

```json
{
  "Hips": {
    "delta_matrix": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
    "location": [0.0, 0.0, 0.0],
    "rotation": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0],
    "head_world": [0.0, -0.00477, 0.8476],
    "head_world_transformed": [0.0, -0.00477, 0.8476]
  }
}
```

**特徴:**
- `scale` ≈ [1.0, 1.0, 1.0]
- `delta_matrix` ≈ 単位行列
- `head_world` = `head_world_transformed`（同じ値）

### 用途2: Template → Target への変換情報

`pose_basis_mao.json` など、Template からターゲットアバターへの変換情報を保存します。
実質的に `posediff_template_to_*.json` と同じ形式です。

```json
{
  "Hips": {
    "delta_matrix": [[1.02,0,0,0], [0,1.02,0,0], [0,0,1.02,0.0595], [0,0,0,1]],
    "location": [0.0, 0.0, 0.0763],
    "rotation": [0.0, 0.0, 0.0],
    "scale": [1.0198, 1.0198, 1.0198],
    "head_world": [0.0, -0.00477, 0.8476],
    "head_world_transformed": [0.0, -0.00477, 0.9238]
  }
}
```

**特徴:**
- `scale` ≠ [1.0, 1.0, 1.0]（ターゲットのサイズに応じたスケール）
- `delta_matrix` に変換情報を含む
- `head_world` = Template の Hips 位置（変換前）
- `head_world_transformed` = Target の Hips 位置（変換後）

> **Note**: `pose_basis_<target>.json` は、ターゲットアバターが Template より
> 大きい/小さい場合、そのサイズ比に応じた scale 値を持ちます。
> 例: mao が Template より約2%大きい場合、scale ≈ 1.02

### フィールド説明

| フィールド | 型 | 単位 | 説明 |
|-----------|-----|------|------|
| `delta_matrix` | 4x4 matrix | - | 変換行列（必須） |
| `location` | [x, y, z] | メートル | ボーンヘッドの移動量（参考値） |
| `rotation` | [x, y, z] | 度 | オイラー角（XYZ順、参考値） |
| `scale` | [x, y, z] | 倍率 | スケール（参考値） |
| `head_world` | [x, y, z] | メートル | 変形前のボーンヘッド位置（ワールド座標） |
| `head_world_transformed` | [x, y, z] | メートル | 変形後のボーンヘッド位置（ワールド座標） |

## posediff_*.json

2つのアバター間のポーズ差分を保存します。

### 現在の形式（MochiFitter-Kai v0.2.x以降）

```json
{
  "Hips": {
    "delta_matrix": [
      [1.3, 0.0, 0.0, 0.0],
      [0.0, 1.3, 0.0, 0.0],
      [0.0, 0.0, 1.3, 0.07627],
      [0.0, 0.0, 0.0, 1.0]
    ],
    "location": [0.0, 0.0, 0.07627],
    "rotation": [0.0, 0.0, 0.0],
    "scale": [1.3, 1.3, 1.3],
    "head_world": [0.0, -0.00477, 0.8476],
    "head_world_transformed": [0.0, -0.00477, 0.9238]
  }
}
```

> **Note**: `delta_matrix` は Unity パッケージ同梱スクリプトとの互換性のため必須です。
> `location/rotation/scale` は参考値として同時に保存されます。

### 旧形式（本家 MochiFitter）

```json
{
  "Hips": {
    "delta_matrix": [
      [1.0169, 0.0, 0.0, 0.0],
      [0.0, 1.0169, 0.0, -0.0157],
      [0.0, 0.0, 1.0169, -0.0207],
      [0.0, 0.0, 0.0, 1.0]
    ],
    "location": [0.0, -0.0157, -0.0207],
    "rotation": [-0.00000278, 0.0, 0.0],
    "scale": [1.0169, 1.0169, 1.0169],
    "head_world": [0.0, 0.00956, 0.9303],
    "head_world_transformed": [0.0, -0.00599, 0.9252]
  }
}
```

### フィールド説明

| フィールド | 型 | 単位 | 説明 |
|-----------|-----|------|------|
| `delta_matrix` | 4x4 matrix | - | 変換行列（Unity 互換性のため必須） |
| `location` | [x, y, z] | メートル | ボーンヘッドの移動量（参考値） |
| `rotation` | [x, y, z] | 度（現在）/ラジアン（旧形式） | オイラー角（XYZ順、参考値） |
| `scale` | [x, y, z] | 倍率 | スケール（参考値） |
| `head_world` | [x, y, z] | メートル | ソースアバターのボーンヘッド位置 |
| `head_world_transformed` | [x, y, z] | メートル | ターゲットアバターのボーンヘッド位置 |

### 形式の比較

| 項目 | 現在の形式 (v0.2.x+) | 旧形式（本家） |
|------|---------------------|---------------|
| delta_matrix | あり（必須） | あり |
| rotation単位 | 度 | ラジアン |
| location/rotation/scale | 参考値として保存 | 保存 |

### 読み込み時の優先順位

1. `delta_matrix` が存在 → 直接使用（最も正確、Unity 互換）
2. `delta_matrix` なし → `location/rotation/scale` から行列を再構築（フォールバック）

> **注意**: `delta_matrix` が存在する場合、`location/rotation/scale` を手動編集しても
> 変形には反映されません。値を調整したい場合は Blender UI から操作してください。

> **注意（フォールバック時）**: `delta_matrix` がなく、`rotation` がラジアン単位の旧形式JSONは
> 正しく読み込めません（度として誤解釈されるため）。
> `delta_matrix` のないJSONを使用する場合は、`rotation` が度単位であることを確認してください。

### バージョン履歴

| バージョン | 変更内容 |
|-----------|---------|
| v0.1.0 | `delta_matrix` を廃止（ユーザー編集を可能にするため） |
| v0.2.x | `delta_matrix` を復活（Unity 互換性のため） |

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
