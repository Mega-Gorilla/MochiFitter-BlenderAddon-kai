# Execute Retargeting コマンド詳細

> **注意**: Unity アドオン (OutfitRetargetingSystem) は有料コンテンツのため、
> 本リポジトリの修正範囲外です。このドキュメントは Blender スクリプト開発者向けの
> 参考情報として、実際の呼び出しパターンと引数の詳細を記載しています。

## 概要

Unity アドオンの GUI から「Execute Retargeting」を実行すると、Blender がバックグラウンドで
起動され、`retarget_script2_XX.py` が実行されます。このドキュメントでは、実際のコマンド
呼び出しとパラメータの詳細を解説します。

## 実際の呼び出し例

### 単一変換（Source → Target）

```bash
blender --background --python retarget_script2_12.py -- \
    --input="Beryl_Costumes.fbx" \
    --output="retargeted_Beryl_Costumes.fbx" \
    --base="base_project.blend" \
    --base-fbx="Template.fbx" \
    --config="config_beryl2template.json" \
    --init-pose="initial_pose.json" \
    --hips-position="0.00000000,0.00955725,0.93028500" \
    --target-meshes="Costume_Body;Costume_Socks;HighHeel" \
    --blend-shapes="Highheel" \
    --blend-shape-values="1.000"
```

### チェーン変換（Source → Intermediate → Target）

```bash
blender --background --python retarget_script2_12.py -- \
    --input="Beryl_Costumes.fbx" \
    --output="retargeted_Beryl_Costumes.fbx" \
    --base="base_project.blend" \
    --base-fbx="Template.fbx;mao.fbx" \
    --config="config_beryl2template.json;config_template2mao.json" \
    --init-pose="initial_pose.json" \
    --hips-position="0.00000000,0.00955725,0.93028500" \
    --target-meshes="Costume_Body;Costume_Socks;HighHeel" \
    --blend-shape-mappings="Highheel,Highheel" \
    --blend-shapes="Highheel" \
    --blend-shape-values="1.000" \
    --mesh-material-data="material.json" \
    --name-conv="bone_mapping.json" \
    --shape-name-file="blendshape_names.json" \
    --no-subdivision
```

## パラメータ詳細

### 必須パラメータ

| パラメータ | 説明 | 値の由来 |
|-----------|------|---------|
| `--input` | 変換対象の衣装 FBX | Unity で選択した衣装 |
| `--output` | 出力 FBX パス | Unity が自動生成 |
| `--base` | ベース Blender プロジェクト | Unity アドオン同梱 |
| `--base-fbx` | ターゲットアバター FBX | config の `baseAvatarDataPath` から取得 |
| `--config` | リターゲット設定 JSON | Unity アドオンが生成 |
| `--init-pose` | 初期ポーズ JSON | Unity アドオンが生成 |

### 実運用上ほぼ必須のパラメータ

以下のパラメータは argparse 上は optional ですが、`retarget_script2_12.py` 内部で
実質必須として使用されています。Unity アドオンは常にこれらを渡します。

| パラメータ | 説明 | 備考 |
|-----------|------|------|
| `--cloth-metadata` | 衣装メタデータ JSON | `load_cloth_metadata()` で使用 |
| `--mesh-material-data` | マテリアル情報 JSON | `load_mesh_material_data()` で使用 |
| `--target-meshes` | 処理対象メッシュ名 | 指定しないと全メッシュが対象 |

> **Note**: CLI から直接実行する場合も、これらのパラメータを適切に指定してください。

### 位置調整パラメータ

| パラメータ | 説明 | 値の由来 |
|-----------|------|---------|
| `--hips-position` | Hips ボーン位置 (x,y,z) | **ソースアバター**の `pose_basis.json` の `head_world` |

> **重要**: `--hips-position` は**ソースアバター**（衣装元）の Hips 位置が渡されます。
> これは `adjust_armature_hips_position()` 関数で使用されます。
> 詳細は [hips-position の動作](#hips-position-の動作) を参照。

> **座標系**: この値は **Blender 座標系**（Z-up, 右手系）で指定します。
> Unity アドオンは内部で Blender 座標系に変換してからこの引数を渡しています。
> 具体的には `armature_obj.matrix_world @ pose_bone.head` と同じ空間です。

### メッシュ関連パラメータ

| パラメータ | 説明 |
|-----------|------|
| `--target-meshes` | 処理対象メッシュ名（セミコロン区切り） |
| `--mesh-material-data` | マテリアル情報 JSON |
| `--mesh-renderers` | メッシュレンダラー情報 |

### ブレンドシェイプ関連パラメータ

| パラメータ | 説明 |
|-----------|------|
| `--blend-shapes` | 適用するブレンドシェイプ名（セミコロン区切り） |
| `--blend-shape-values` | ブレンドシェイプ強度（セミコロン区切り） |
| `--blend-shape-mappings` | ソース→ターゲットのマッピング（label,name ペア） |

### その他オプション

| パラメータ | 説明 |
|-----------|------|
| `--name-conv` | ボーン名変換マッピング JSON |
| `--shape-name-file` | シェイプキー名同期用 JSON |
| `--no-subdivision` | サブディビジョンを無効化 |
| `--no-triangle` | 三角形化を無効化 |

## チェーン処理の詳細フロー

チェーン処理（複数の config を連続適用）の場合、`--base-fbx` と `--config` をセミコロンで
区切って複数指定します。

### 例: Beryl → Template → mao

```
--base-fbx="Template.fbx;mao.fbx"
--config="config_beryl2template.json;config_template2mao.json"
```

### 処理フロー図

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      チェーン処理フロー                                   │
└─────────────────────────────────────────────────────────────────────────┘

入力: Beryl_Costumes.fbx (Beryl の衣装)

Step 1: ペア 1/2 (Beryl → Template)
┌─────────────────────────────────────────────────────────────────────────┐
│ Base FBX: Template.fbx                                                  │
│ Config: config_beryl2template.json                                      │
│                                                                         │
│ 処理内容:                                                               │
│   1. Template アバターを読み込み                                         │
│   2. Beryl 衣装を読み込み                                                │
│   3. posediff_beryl_to_template.json でポーズ適用                        │
│   4. deformation_beryl_to_template.npz で変形適用                        │
│   5. ウェイト転送・ボーン置換                                             │
│                                                                         │
│ hips_position: --hips-position から取得 (Beryl の Hips 位置)             │
│ Hip Offset: (0, 0, 0) → スキップ                                        │
│   ※ Beryl 衣装の Hips ≈ Beryl の Hips なので offset が 0                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            [中間結果: Template 向け衣装]
                                    │
                                    ▼
Step 2: ペア 2/2 (Template → mao)
┌─────────────────────────────────────────────────────────────────────────┐
│ Base FBX: mao.fbx                                                       │
│ Config: config_template2mao.json                                        │
│                                                                         │
│ 処理内容:                                                               │
│   1. mao アバターを読み込み                                              │
│   2. Step 1 の結果を衣装として使用                                       │
│   3. posediff_template_to_mao.json でポーズ適用                          │
│   4. deformation_template_to_mao.npz で変形適用                          │
│   5. ウェイト転送・ボーン置換                                             │
│                                                                         │
│ hips_position: --hips-position で指定（または None）                      │
│ 位置調整: posediff の delta_matrix に含まれる平行移動で自動適用            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
出力: retargeted_Beryl_Costumes.fbx (mao 向け衣装)
```

## hips-position の動作

### パラメータの意味

`--hips-position` は衣装の Armature を指定位置に移動させるために使用されます。

```python
# retarget_script2_12.py: adjust_armature_hips_position()
def adjust_armature_hips_position(armature_obj, target_position, clothing_avatar_data):
    # 現在の Hips 位置を取得
    current_position = armature_obj.matrix_world @ pose_bone.head

    # オフセット計算
    offset = target_position - current_position

    # Armature を移動
    armature_obj.location += offset
```

### チェーン処理での動作

| ステップ | hips_position の値 | 位置調整の仕組み |
|---------|-------------------|-----------------|
| i = 0 | `--hips-position` の値 | Unity が渡す値で調整（offset ≈ 0 でスキップされることが多い） |
| i > 0 | `None`（通常） | `posediff` の `delta_matrix` に含まれる平行移動で自動調整 |

> **Note**: チェーン処理での位置調整は、`posediff_*.json` の `delta_matrix` に
> 平行移動成分（例: Z +5.95cm）が含まれているため、`--hips-position` がなくても
> 正しく動作します。詳細は [Issue #15](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/issues/15) を参照。

### 関連 Issue

- [Issue #15](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/issues/15): チェーン処理時の Hips 位置調整（生成プロセス修正で解決）

## ログ出力の解説

### 主要なログメッセージ

```
処理開始: ペア 1/2
Base FBX: Template.fbx
Config: config_beryl2template.json

# Hips 位置調整のログ
Hip Offset: <Vector (0.0000, 0.0000, 0.0000)>
Hips position is already at target position, skipping adjustment

# ポーズ適用のログ
Pose data added to armature 'Armature' from posediff_beryl_to_template.json

処理完了: 合計 96.41秒
```

### チェーン処理時の特徴的なログ

```
# Step 2 では posediff の delta_matrix で位置調整が行われる
Pose data added to armature 'Armature' from posediff_template_to_mao.json
```

> **Note**: 位置調整は `posediff` の `delta_matrix` に含まれる平行移動成分で
> 自動的に行われます。`--hips-position` による明示的な調整ログは出力されません。

## config JSON とパラメータの関係

### config JSON の構造

```json
{
    "poseDataPath": "posediff_beryl_to_template.json",
    "fieldDataPath": "deformation_beryl_to_template.npz",
    "baseAvatarDataPath": "avatar_data_template.json",
    "clothingAvatarDataPath": "avatar_data_beryl.json",
    "sourceBlendShapeSettings": [...],
    "targetBlendShapeSettings": [...]
}
```

### パラメータと config の対応

| Blender 引数 | config JSON フィールド |
|-------------|----------------------|
| (内部で使用) | `poseDataPath` |
| (内部で使用) | `fieldDataPath` |
| (内部で使用) | `baseAvatarDataPath` |
| (内部で使用) | `clothingAvatarDataPath` |
| `--blend-shapes` | (Unity が `sourceBlendShapeSettings` から生成) |

## データファイルの流れ

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        データファイルの流れ                               │
└─────────────────────────────────────────────────────────────────────────┘

[Blender アドオンで生成]
    │
    ├── avatar_data_beryl.json ──────────────────────┐
    ├── avatar_data_template.json ───────────────────┤
    ├── avatar_data_mao.json ────────────────────────┤
    │                                                │
    ├── pose_basis_beryl.json ───────────────────────┤
    │   └── head_world: [0, 0.009557, 0.930285] ─────┼── → --hips-position
    ├── pose_basis_template.json (※1) ──────────────┤
    ├── pose_basis_mao.json (※2) ───────────────────┤
    │                                                │
    ├── posediff_beryl_to_template.json ─────────────┤
    ├── posediff_template_to_mao.json ───────────────┤
    │                                                │
    ├── deformation_beryl_to_template.npz ───────────┤
    └── deformation_template_to_mao.npz ─────────────┤
                                                     │
                                                     ▼
[Unity アドオンで生成]                          [config_*.json]
    │                                                │
    ├── config_beryl2template.json ──────────────────┤
    └── config_template2mao.json ────────────────────┤
                                                     │
                                                     ▼
[Unity → Blender subprocess]                 [retarget_script]
                                                     │
                                                     ▼
                                           [リターゲット済み FBX]
```

### pose_basis ファイルの種類

| ファイル | 内容 | 特徴 |
|---------|------|------|
| (※1) pose_basis_template.json | Template の Rest Pose | scale = 1.0, delta_matrix ≈ 単位行列 |
| (※2) pose_basis_mao.json | Template → mao の変換情報 | scale ≈ 1.02（mao は Template より約2%大きい） |

> **Note**: `pose_basis_<target>.json` は Template からの変換情報を含むため、
> `posediff_template_to_<target>.json` と実質的に同じ形式です。
> 詳細は [Blender アドオン - データフォーマット](../blender-addon/data_formats.md#pose_basisjson) を参照。

## トラブルシューティング

### ログで確認すべきポイント

1. **処理ペア数の確認**
   ```
   処理開始: ペア 1/2  ← チェーン処理の場合は複数ペア
   ```

2. **Hips 位置調整の確認**
   ```
   Hip Offset: <Vector (x, y, z)>
   ```
   - offset が (0, 0, 0) の場合、位置調整はスキップされます
   - チェーン処理の Step 2 以降では自動計算されます

3. **ポーズ適用の確認**
   ```
   Pose data added to armature '...' from posediff_*.json
   ```

4. **処理時間の確認**
   ```
   処理完了: 合計 XX.XX秒
   ```

### よくある問題

| 症状 | 原因 | 対処 |
|------|------|------|
| Hips 位置がずれる | チェーン処理の Step 1 で offset = 0 | 最終結果を確認（Step 2 で調整される） |
| ポーズが適用されない | posediff JSON パスが不正 | config の `poseDataPath` を確認 |
| 変形が適用されない | NPZ ファイルパスが不正 | config の `fieldDataPath` を確認 |

### FBX 比較スクリプトについて

`scripts/compare_fbx_hips.py` を使用して FBX ファイルの位置を比較できます。

```bash
blender --background --python scripts/compare_fbx_hips.py -- before.fbx after.fbx
```

> **注意**: このスクリプトは **FBX 内部のメッシュ/アーマチュア位置** を比較します。
> Unity での **Prefab 配置問題**（Transform の位置・スケール）は直接検出できません。
> Unity 上で位置がずれる場合は、FBX 比較結果だけでなく、
> Prefab の Transform 設定や親オブジェクトのスケールも確認してください。

## 関連ドキュメント

- [処理フロー概要](overview.md)
- [config JSON 仕様](config_format.md)
- [リターゲットスクリプト解説](blender_script.md)
- [Blender アドオン - データフォーマット](../blender-addon/data_formats.md)
