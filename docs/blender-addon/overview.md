# MochiFitter Blenderアドオン - アーキテクチャ概要

## 概要

MochiFitter Blenderアドオンは、VRChat/VRMアバター向けの衣装リターゲティングツールです。
RBF（Radial Basis Function）補間を使用して、異なるアバター体型間でメッシュ変形を転送します。

## ファイル構成

```
MochiFitter-BlenderAddon/
├── SaveAndApplyFieldAuto.py      # メインアドオンモジュール
├── rbf_multithread_processor.py  # マルチスレッド処理用外部スクリプト
└── deps/                         # 依存ライブラリ（scipy等）
```

## 主要コンポーネント

### 1. UIパネル

**場所**: View3D > Sidebar > MochiFitter

```
┌─────────────────────────────────────┐
│ アバター設定                         │
│  - ソース/ターゲットアバター名        │
│  - アバターデータファイル             │
├─────────────────────────────────────┤
│ ベースポーズ差分                     │
│  - ベースポーズ保存/適用             │
├─────────────────────────────────────┤
│ ポーズ差分                          │
│  - ポーズ差分保存/適用               │
├─────────────────────────────────────┤
│ 変形フィールド設定                   │
│  - シェイプキー選択                  │
│  - RBFパラメータ                    │
│  - グリッド/距離設定                │
├─────────────────────────────────────┤
│ 実行                               │
│  - シングルスレッド処理              │
│  - マルチスレッド処理                │
├─────────────────────────────────────┤
│ 変形データ適用                      │
├─────────────────────────────────────┤
│ フィールド可視化                    │
├─────────────────────────────────────┤
│ NumPy/SciPy管理                    │
├─────────────────────────────────────┤
│ デバッグ                           │
└─────────────────────────────────────┘
```

### 2. 処理フロー

```
┌──────────────────────────────────────────────────────────────┐
│                    データ生成フロー                           │
└──────────────────────────────────────────────────────────────┘

Avatar A (Source)                    Avatar B (Target)
      │                                    │
      ▼                                    ▼
┌─────────────┐                    ┌─────────────┐
│ ベースポーズ │                    │ ベースポーズ │
│   保存      │                    │   保存      │
└─────────────┘                    └─────────────┘
      │                                    │
      ▼                                    ▼
pose_basis_A.json                  pose_basis_B.json

      │                                    │
      └──────────────┬─────────────────────┘
                     ▼
              ┌─────────────┐
              │ ポーズ差分  │
              │   計算      │
              └─────────────┘
                     │
                     ▼
         posediff_A_to_B.json


┌──────────────────────────────────────────────────────────────┐
│                  変形フィールド生成フロー                      │
└──────────────────────────────────────────────────────────────┘

Source Mesh + ShapeKey
      │
      ▼
┌─────────────────────────────┐
│ 制御点抽出                   │
│ (変形前/変形後の頂点位置)    │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ RBF補間                      │
│ (Multi-Quadratic Biharmonic) │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ 変形フィールド生成           │
│ (3Dグリッド上の変位ベクトル) │
└─────────────────────────────┘
      │
      ▼
deformation_A_to_B.npz
```

### 3. 主要クラス/関数

#### オペレーター

| クラス名 | bl_idname | 説明 |
|---------|-----------|------|
| `SAVE_OT_BasePoseDiff` | `object.save_base_pose_diff` | ベースポーズ保存 |
| `APPLY_OT_BasePoseDiff` | `object.apply_base_pose_diff` | ベースポーズ適用 |
| `SAVE_OT_PoseDiff` | `object.save_pose_diff` | ポーズ差分保存 |
| `APPLY_OT_PoseDiff` | `object.apply_pose_diff` | ポーズ差分適用 |
| `CREATE_OT_RBFDeformation` | `object.create_rbf_deformation` | 変形データ作成（シングルスレッド） |
| `EXPORT_OT_RBFTempData` | `object.export_rbf_temp_data` | 変形データ作成（マルチスレッド） |
| `APPLY_OT_FieldData` | `rbf.apply_field_data` | 変形フィールド適用 |

#### コア関数

| 関数名 | 説明 |
|--------|------|
| `save_armature_pose()` | Armatureのポーズをワールド座標系でJSON保存 |
| `add_pose_from_json()` | JSONからポーズデータを読み込み適用 |
| `rbf_interpolation()` | RBF補間計算（Multi-Quadratic Biharmonic kernel） |
| `create_shape_key_from_rbf()` | RBFを使用してシェイプキー作成 |
| `save_field_data_multi_step()` | 変形フィールドをNPZ形式で保存 |
| `apply_field_data()` | 保存された変形フィールドをメッシュに適用 |

### 4. 依存関係

```
┌─────────────────┐
│   Blender 4.0+  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ NumPy │ │ SciPy │
└───────┘ └───────┘
              │
         ┌────┴────┐
         ▼         ▼
    ┌────────┐ ┌────────┐
    │ cKDTree│ │ cdist  │
    └────────┘ └────────┘
```

- **Blender 4.0+**: 必須（APIの互換性）
- **NumPy**: 配列演算、NPZファイル入出力
- **SciPy**:
  - `cKDTree`: 高速な最近傍探索
  - `cdist`: 距離行列計算

### 5. 出力ファイル

| ファイル形式 | 命名規則 | 説明 |
|------------|---------|------|
| `avatar_data_*.json` | avatar_data_{avatar_name}.json | ボーン階層、Humanoidマッピング |
| `pose_basis_*.json` | pose_basis_{avatar_name}.json | アバターのベースポーズ |
| `posediff_*.json` | posediff_{source}_to_{target}.json | 2アバター間のポーズ差分 |
| `deformation_*.npz` | deformation_{source}_to_{target}.npz | 変形フィールドデータ |

## RBFアルゴリズム

### Multi-Quadratic Biharmonic Kernel

```python
def multi_quadratic_biharmonic(r, epsilon=1.0):
    """
    φ(r) = sqrt(r² + ε²)
    """
    return np.sqrt(r**2 + epsilon**2)
```

### 補間処理

1. **制御点の抽出**: シェイプキーの変形前/変形後の頂点位置を取得
2. **重み計算**: RBFカーネルを使用して重み係数を計算
3. **補間**: ターゲット位置での変位ベクトルを計算

## マルチスレッド処理

BlenderのGIL（Global Interpreter Lock）を回避するため、重い計算は外部プロセスで実行：

```
┌─────────────────┐      temp_rbf_*.npz      ┌────────────────────────┐
│  Blender        │ ─────────────────────▶   │ rbf_multithread_       │
│  (メインプロセス) │                          │ processor.py           │
│                 │ ◀─────────────────────   │ (ProcessPoolExecutor)  │
└─────────────────┘   deformation_*.npz      └────────────────────────┘
```

## 座標系

- **Blender**: Z-up, 右手座標系
- **保存形式**: ワールド座標系で保存
- **注意**: Unity（Y-up, 左手座標系）との変換はUnity側で処理

## 関連ドキュメント

- [データフォーマット仕様](data_formats.md)
- [オペレーター詳細](operators.md)
