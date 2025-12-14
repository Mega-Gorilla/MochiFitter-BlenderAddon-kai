# リターゲットスクリプト解説

> **ライセンス**: このスクリプトは GPL v3 ライセンスで公開されています。
> 本ドキュメントは Unity パッケージ同梱の Blender スクリプトの解説です。

## 概要

`retarget_script` は Unity アドオンから呼び出される Blender Python スクリプトで、
衣装メッシュのリターゲット処理を実行します。

> **Note**: スクリプト名はバージョンにより異なります（例: `retarget_script2_12.py`）。
> Unity アドオンが適切なバージョンを自動で展開・使用します。

## 呼び出し方法

Unity から以下のように呼び出されます：

```bash
blender --background --python retarget_script2_XX.py -- \
    --input clothing.fbx \
    --output output.fbx \
    --base scene.blend \
    --base-fbx "avatar1.fbx;avatar2.fbx" \
    --config "config1.json;config2.json" \
    --init-pose init_pose.json \
    [オプション...]
```

### 必須引数

| 引数 | 説明 |
|------|------|
| `--input` | 入力衣装 FBX ファイル |
| `--output` | 出力 FBX ファイル |
| `--base` | ベース Blender ファイル (.blend) |
| `--base-fbx` | ベースアバター FBX（セミコロン区切りで複数指定可） |
| `--config` | config JSON（セミコロン区切りで複数指定可） |
| `--init-pose` | 初期ポーズ JSON ファイル |

### 複数ペア処理（チェーン処理）

`--base-fbx` と `--config` はセミコロン区切りで複数指定できます。
複数指定時は、前段の処理結果が次段の入力として使用されるチェーン処理になります：

```
[入力 FBX] → [config1 処理] → [中間結果] → [config2 処理] → [出力 FBX]
```

## 処理フロー

```
┌─────────────────────────────────────────────────────────────────┐
│                    リターゲット処理フロー                         │
└─────────────────────────────────────────────────────────────────┘

1. 初期化
   ├── アドオン有効化 (robust-weight-transfer)
   └── 引数解析

2. ベースアバター処理
   ├── FBX インポート
   ├── avatar_data.json 読み込み
   ├── ボーンウェイトのマージ
   └── ベースポーズ適用

3. 衣装アバター処理
   ├── FBX インポート
   ├── avatar_data.json 読み込み
   ├── シェイプキー名の同期
   ├── ボーン名の正規化
   └── Hips 位置の調整

4. ポーズ適用
   ├── 初期ポーズ適用 (init_pose.json)
   └── ポーズ差分適用 (posediff.json)

5. 変形フィールド適用
   ├── NPZ ファイル読み込み
   ├── delta_matrix による変換
   ├── RBF 補間による変形
   └── シェイプキー生成

6. ウェイト転送
   ├── Humanoid ボーンのウェイト転送
   ├── 補助ボーンのウェイト処理
   └── ウェイトの正規化

7. ボーン置換
   ├── Humanoid ボーンの置換
   └── 補助ボーンの処理

8. 出力
   ├── シェイプキーのマージ・クリーンアップ
   └── FBX エクスポート
```

## 主要な関数

### メイン処理

| 関数名 | 説明 |
|--------|------|
| `main()` | エントリーポイント、複数 config ペアを順次処理 |
| `process_single_config()` | 1つの config に対する全処理を実行 |
| `parse_args()` | コマンドライン引数の解析 |

### アバター処理

| 関数名 | 説明 |
|--------|------|
| `process_base_avatar()` | ベースアバターの読み込みと初期処理 |
| `process_clothing_avatar()` | 衣装アバターの読み込みと初期処理 |
| `import_fbx()` | FBX ファイルのインポート |
| `export_fbx()` | FBX ファイルのエクスポート |

### ポーズ適用

| 関数名 | 説明 |
|--------|------|
| `add_pose_from_json()` | posediff JSON からポーズを適用 |
| `apply_initial_pose_to_armature()` | 初期ポーズを適用 |
| `add_clothing_pose_from_json()` | 衣装用ポーズ適用（初期ポーズ + posediff） |

### 変形フィールド

| 関数名 | 説明 |
|--------|------|
| `apply_symmetric_field_delta()` | 対称変形フィールドの適用（メイン） |
| `process_field_deformation()` | 変形フィールド処理 |
| `get_deformation_field()` | NPZ から変形フィールドを読み込み |
| `batch_process_vertices_multi_step()` | マルチステップ頂点処理 |

### ウェイト転送

| 関数名 | 説明 |
|--------|------|
| `process_weight_transfer()` | ウェイト転送のメイン処理 |
| `transfer_weights_from_nearest_vertex()` | 最近傍頂点からのウェイト転送 |
| `normalize_bone_weights()` | ボーンウェイトの正規化 |
| `merge_humanoid_bone_weights()` | Humanoid ボーンウェイトのマージ |

### ボーン操作

| 関数名 | 説明 |
|--------|------|
| `replace_humanoid_bones()` | Humanoid ボーンの置換 |
| `normalize_clothing_bone_names()` | 衣装ボーン名の正規化 |
| `get_humanoid_bone_hierarchy()` | Humanoid ボーン階層の取得 |
| `clear_humanoid_bone_relations_preserve_pose()` | ボーン親子関係の解除（ポーズ保持） |

### ブレンドシェイプ

| 関数名 | 説明 |
|--------|------|
| `apply_blend_shape_settings()` | ブレンドシェイプ値の適用 |
| `apply_blendshape_deformation_fields()` | ブレンドシェイプ変形フィールドの適用 |
| `create_blendshape_mask()` | ブレンドシェイプマスクの作成 |

## delta_matrix の処理

`add_pose_from_json()` での delta_matrix 処理：

```python
# delta_matrix を直接取得（必須）
delta_matrix = list_to_matrix(pose_data[source_humanoid_bone]['delta_matrix'])

if invert:
    delta_matrix = delta_matrix.inverted()

# 現在の行列に適用
combined_matrix = delta_matrix @ current_world_matrix

# ローカル空間に変換して適用
bone.matrix = armature_obj.matrix_world.inverted() @ combined_matrix
```

> **重要**: スクリプトは `delta_matrix` を**必須**として直接参照します。
> `location/rotation/scale` からのフォールバック処理は**存在しません**。
> `delta_matrix` が欠けている場合は例外が発生します。
>
> このため、MochiFitter-Kai は `delta_matrix` を常に出力します。

## コマンドライン引数（全体）

### 必須引数

| 引数 | 説明 |
|------|------|
| `--input` | 入力衣装 FBX ファイル |
| `--output` | 出力 FBX ファイルパス |
| `--base` | ベース Blender ファイル (.blend) |
| `--base-fbx` | ベースアバター FBX（セミコロン区切りで複数可） |
| `--config` | config JSON（セミコロン区切りで複数可） |
| `--init-pose` | 初期ポーズ JSON ファイル |

### オプション引数

| 引数 | 説明 |
|------|------|
| `--hips-position` | ターゲット Hips ボーン位置（x,y,z 形式） |
| `--blend-shapes` | 適用するブレンドシェイプ（セミコロン区切り） |
| `--blend-shape-values` | ブレンドシェイプ強度（セミコロン区切り） |
| `--blend-shape-mappings` | ブレンドシェイプマッピング（label,name ペア） |
| `--target-meshes` | 処理対象メッシュ名（セミコロン区切り） |
| `--cloth-metadata` | 衣装メタデータファイル |
| `--mesh-material-data` | メッシュマテリアルデータ |
| `--mesh-renderers` | メッシュレンダラー情報 |
| `--shape-name-file` | シェイプキー名 JSON ファイル |
| `--name-conv` | ボーン名変換データ |
| `--no-subdivision` | サブディビジョンを無効化 |
| `--no-triangle` | 三角形化を無効化 |

## 出力ファイル

処理完了時に以下のファイルが生成されます：

| ファイル | 説明 |
|---------|------|
| `{output_base}.fbx` | リターゲット済みメッシュ |
| `{output_base}.blend` | 処理後の Blender シーン（デバッグ用） |
| `{output_base}_error_XXX.blend` | エラー発生時のシーン（デバッグ用） |

> **Note**: `{output_base}` は `--output` 引数から拡張子を除いたベース名です。
> 例: `--output result.fbx` の場合、`result.fbx`、`result.blend` が生成されます。
> `.blend` ファイルは処理の最終状態を保存しており、問題発生時のデバッグに役立ちます。

## 依存関係

- Blender 4.0+
- NumPy
- SciPy (cKDTree)
- robust-weight-transfer アドオン

## 処理時間の目安

典型的な処理時間（参考値）：

| 処理 | 時間 |
|------|------|
| ベースファイル読み込み | 1-2秒 |
| アバター処理 | 2-5秒 |
| 変形フィールド適用 | 5-15秒 |
| ウェイト転送 | 10-30秒 |
| FBX エクスポート | 1-3秒 |
| **合計** | **20-60秒** |

※ メッシュの複雑さ、ブレンドシェイプの数により大きく変動します。

## エラーハンドリング

処理中にエラーが発生した場合：
- エラー詳細とスタックトレースが出力されます
- エラー時のシーンが `*_error_*.blend` として保存されます
- デバッグに役立つ中間状態を確認できます

## 関連ドキュメント

- [処理フロー概要](overview.md)
- [config JSON 仕様](config_format.md)
- [Blender アドオン - データフォーマット](../blender-addon/data_formats.md)
