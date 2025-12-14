# MochiFitter オペレーター詳細

## 概要

MochiFitter Blenderアドオンは以下のオペレーターを提供します。

## ポーズ関連オペレーター

### SAVE_OT_BasePoseDiff

**bl_idname**: `object.save_base_pose_diff`

アクティブなArmatureのベースポーズをJSONファイルに保存します。

#### 前提条件
- アクティブオブジェクトがArmatureであること
- ソースアバター名が設定されていること
- ソースアバターデータファイルが設定されていること

#### 出力
- ファイル: `pose_basis_{source_avatar_name}.json`
- 場所: Blendファイルと同じフォルダ

#### 内部処理
```python
# SaveAndApplyFieldAuto.py: save_armature_pose()
def save_armature_pose(armature_obj, filename, avatar_data_file):
    # 1. アバターデータからHumanoidボーンマッピングを取得
    # 2. 各ボーンのワールド空間での変換を計算
    # 3. delta_matrix = world_matrix @ base_world_matrix.inverted()
    # 4. location, rotation(度), scaleを抽出してJSON保存
```

---

### APPLY_OT_BasePoseDiff

**bl_idname**: `object.apply_base_pose_diff`

保存されたベースポーズをArmatureに適用します。

#### 前提条件
- アクティブオブジェクトがArmatureであること
- ソースアバターデータファイルが設定されていること

#### パラメータ
- **Invert Pose**: 逆変換を適用するかどうか

---

### SAVE_OT_PoseDiff

**bl_idname**: `object.save_pose_diff`

2つのアバター間のポーズ差分をJSONファイルに保存します。

#### 前提条件
- アクティブオブジェクトがArmatureであること
- ソース/ターゲットアバター名が設定されていること
- ソースアバターデータファイルが設定されていること

#### 出力
- ファイル: `posediff_{source}_to_{target}.json`
- 場所: Blendファイルと同じフォルダ

---

### APPLY_OT_PoseDiff

**bl_idname**: `object.apply_pose_diff`

保存されたポーズ差分をArmatureに適用します。

#### 前提条件
- アクティブオブジェクトがArmatureであること
- ターゲットアバターデータファイルが設定されていること

#### パラメータ
- **Invert Pose**: 逆変換を適用するかどうか

#### 内部処理（delta_matrix読み込み優先順位）
```python
# SaveAndApplyFieldAuto.py: add_pose_from_json()
def add_pose_from_json(...):
    # 1. delta_matrixが存在する場合 → 直接使用（最も正確）
    if 'delta_matrix' in bone_pose:
        delta_matrix = list_to_matrix(bone_pose['delta_matrix'])
    # 2. delta_matrixがない場合 → location/rotation/scaleから再構築
    elif 'location' in bone_pose and 'rotation' in bone_pose and 'scale' in bone_pose:
        delta_loc = Vector(bone_pose['location'])
        delta_rot = Euler([math.radians(x) for x in bone_pose['rotation']], 'XYZ')
        delta_scale = Vector(bone_pose['scale'])
        delta_matrix = Translation @ Rotation @ Scale
```

## 変形フィールド関連オペレーター

### CREATE_OT_RBFDeformation

**bl_idname**: `object.create_rbf_deformation`

シングルスレッドでRBF変形データを作成・保存します。

#### 前提条件
- ソースメッシュオブジェクトが選択されていること
- シェイプキーが設定されていること
- ソース/ターゲットアバター名が設定されていること

#### パラメータ
| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| Epsilon | RBFカーネルのε値 | 0.00001 |
| Number of Steps | 変形分割ステップ数 | 1 |
| Selected Vertices Only | 選択頂点のみを制御点として使用 | False |
| Save Shape Key Transform | シェイプキー変形モード | False |
| Enable X Mirror | X軸ミラーリング | True |

#### 出力
- ファイル: `deformation_{source}_to_{target}.npz`

---

### EXPORT_OT_RBFTempData

**bl_idname**: `object.export_rbf_temp_data`

マルチスレッド処理用の一時データをエクスポートし、外部プロセスでRBF計算を実行します。

#### 処理フロー
```
┌─────────────────┐
│ 1. 一時データ    │
│    エクスポート  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 2. 外部プロセス起動                   │
│    rbf_multithread_processor.py     │
│    (ProcessPoolExecutor使用)         │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ 3. 結果読み込み  │
│    NPZ保存       │
└─────────────────┘
```

#### 外部プロセス
- スクリプト: `rbf_multithread_processor.py`
- ワーカー数: `min(8, os.cpu_count())`（BrokenProcessPool対策）

---

### APPLY_OT_FieldData

**bl_idname**: `rbf.apply_field_data`

保存された変形フィールドをメッシュに適用します。

#### 前提条件
- アクティブオブジェクトがメッシュであること
- 対応する変形データファイルが存在すること

#### パラメータ
- **Apply Shape Key Name**: 作成するシェイプキーの名前

#### 内部処理
```python
# SaveAndApplyFieldAuto.py: apply_field_data()
def apply_field_data(target_obj, field_data_path, shape_key_name):
    # 1. NPZファイルから変形フィールドを読み込み
    # 2. KDTreeで最近傍点を検索
    # 3. 距離加重補間で変位を計算
    # 4. シェイプキーとして適用
```

---

### APPLY_OT_InverseFieldData

**bl_idname**: `rbf.apply_inverse_field_data`

逆変形フィールド（`_inv.npz`）をメッシュに適用します。

## ユーティリティオペレーター

### SWAP_OT_AvatarSettings

**bl_idname**: `object.swap_avatar_settings`

ソースとターゲットのアバター設定を入れ替えます。

#### 入れ替え対象
- アバター名
- アバターデータファイル

---

### SET_OT_HumanoidBoneInheritScale

**bl_idname**: `object.set_humanoid_bone_inherit_scale`

Humanoidボーンの`Inherit Scale`を`Average`に設定します。

#### 目的
ボーンスケールの継承方法を統一し、変形の一貫性を確保します。

---

### SELECT_OT_RBFShapeKey

**bl_idname**: `object.select_rbf_shape_key`

シェイプキー選択ドロップダウンメニューを表示します。

---

### SET_OT_RBFShapeKey

**bl_idname**: `object.set_rbf_shape_key`

選択されたシェイプキーを設定します。

## デバッグ/ユーティリティオペレーター

### CREATE_OT_FieldVisualization

**bl_idname**: `rbf.create_field_visualization`

変形フィールドを3Dオブジェクトとして可視化します。

#### パラメータ
- **Field Step**: 可視化するステップ番号
- **Use Inverse Data**: 逆変形データを使用
- **Field Object Name**: 作成するオブジェクト名

---

### DEBUG_OT_ShowPythonPaths

**bl_idname**: `rbf.debug_show_python_paths`

Pythonパス情報をコンソールに表示します。

---

### DEBUG_OT_TestExternalPython

**bl_idname**: `rbf.debug_test_external_python`

外部Pythonプロセスの実行テストを行います。

---

### REINSTALL_OT_NumpyScipyMultithreaded

**bl_idname**: `rbf.reinstall_numpy_scipy_multithreaded`

マルチスレッド対応のNumPy/SciPyを再インストールします。

#### 処理内容
1. 既存のNumPy/SciPyをアンインストール
2. `deps/`フォルダからマルチスレッド対応版をインストール
3. Blender再起動を促す

## プロパティ一覧

### アバター設定

| プロパティ | 型 | 説明 |
|-----------|-----|------|
| `rbf_source_avatar_name` | String | ソースアバター名 |
| `rbf_target_avatar_name` | String | ターゲットアバター名 |
| `rbf_source_avatar_data_file` | String (FILE_PATH) | ソースアバターデータファイル |
| `rbf_target_avatar_data_file` | String (FILE_PATH) | ターゲットアバターデータファイル |
| `rbf_source_obj` | PointerProperty (Object) | ソースメッシュオブジェクト |

### シェイプキー設定

| プロパティ | 型 | 説明 | デフォルト |
|-----------|-----|------|-----------|
| `rbf_source_shape_key` | String | ソースシェイプキー名 | - |
| `rbf_shape_key_start_value` | Float | シェイプキー開始値 | 0.0 |
| `rbf_shape_key_end_value` | Float | シェイプキー終了値 | 1.0 |
| `rbf_save_shape_key_mode` | Bool | シェイプキー変形モード | False |

### RBF設定

| プロパティ | 型 | 説明 | デフォルト |
|-----------|-----|------|-----------|
| `rbf_epsilon` | Float | RBFカーネルのε値 | 0.00001 |
| `rbf_num_steps` | Int | 変形分割ステップ数 | 1 |
| `rbf_selected_only` | Bool | 選択頂点のみ使用 | False |
| `rbf_enable_x_mirror` | Bool | X軸ミラーリング | True |
| `rbf_add_normal_control_points` | Bool | 法線制御点追加 | False |
| `rbf_normal_distance` | Float | 法線方向距離 | -0.0002 |

### グリッド設定

| プロパティ | 型 | 説明 | デフォルト |
|-----------|-----|------|-----------|
| `rbf_base_grid_spacing` | Float | 基本グリッド間隔 | 0.0025 |
| `rbf_bbox_scale_factor` | Float | BBox拡大係数 | 1.5 |
| `rbf_surface_distance` | Float | 表面からの最大距離 | 2.0 |
| `rbf_max_distance` | Float | 最大ウェイト距離 | 0.2 |
| `rbf_min_distance` | Float | 最小ウェイト距離 | 0.005 |
| `rbf_density_falloff` | Float | 密度減衰率 | 4.0 |

### その他

| プロパティ | 型 | 説明 | デフォルト |
|-----------|-----|------|-----------|
| `rbf_pose_invert` | Bool | ポーズ逆変換 | False |
| `rbf_apply_shape_key_name` | String | 適用シェイプキー名 | "RBF_Deform" |
| `rbf_show_debug_info` | Bool | デバッグ情報表示 | False |

## 関連ドキュメント

- [アーキテクチャ概要](overview.md)
- [データフォーマット仕様](data_formats.md)
