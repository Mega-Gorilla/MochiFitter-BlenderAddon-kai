# config JSON 仕様

> **注意**: このドキュメントは Unity アドオンが生成する config JSON の仕様です。
> Blender アドオンユーザー向けの参考情報として提供されています。

## 概要

`config_*.json` は Unity アドオンがリターゲット処理の設定を保存するファイルです。
Blender subprocess に渡され、リターゲットスクリプトが読み込みます。

## ファイル命名規則

ファイル名は任意ですが、一般的なパターン：

```
config_{source}2{target}.json
config_{source}_to_{target}.json
```

例:
- `config_template2mao.json`（template → mao）
- `config_beryl2template.json`（beryl → template）

## 構造

### 基本構造

```json
{
    "poseDataPath": "posediff_source_to_target.json",
    "fieldDataPath": "deformation_source_to_target.npz",
    "sourceBlendShapeSettings": [],
    "targetBlendShapeSettings": [],
    "blendShapeFields": [],
    "baseAvatarDataPath": "avatar_data_target.json",
    "clothingAvatarDataPath": "avatar_data_source.json",
    "clothingBlendShapeSettings": [],
    "clothingBlendShapeSettingsInv": []
}
```

### ブレンドシェイプフィールドを含む構造

```json
{
    "poseDataPath": "posediff_source_to_target.json",
    "fieldDataPath": "deformation_source_to_target.npz",
    "sourceBlendShapeSettings": [],
    "targetBlendShapeSettings": [],
    "blendShapeFields": [
        {
            "label": "Breasts_big",
            "sourceLabel": "Breasts",
            "path": "deformation_source_to_target_breasts.npz",
            "maskBones": ["Chest", "LeftBreast", "RightBreast"],
            "sourceBlendShapeSettings": [
                {"name": "Breasts", "value": 1.0}
            ],
            "targetBlendShapeSettings": [
                {"name": "Breasts_big", "value": 1.0}
            ]
        }
    ],
    "baseAvatarDataPath": "avatar_data_target.json",
    "clothingAvatarDataPath": "avatar_data_source.json",
    "clothingBlendShapeSettings": [],
    "clothingBlendShapeSettingsInv": []
}
```

## フィールド説明

### 必須フィールド

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `poseDataPath` | string | posediff JSON ファイルへの相対パス |
| `fieldDataPath` | string | deformation NPZ ファイルへの相対パス |
| `baseAvatarDataPath` | string | ターゲットアバターの avatar_data JSON |
| `clothingAvatarDataPath` | string | ソースアバター（衣装側）の avatar_data JSON |

### 任意フィールド

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `doNotUseBasePose` | int | 0 | 1 の場合、ベースポーズを使用しない |

### ブレンドシェイプ設定

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `sourceBlendShapeSettings` | array | ソースメッシュのブレンドシェイプ初期値 |
| `targetBlendShapeSettings` | array | ターゲットメッシュのブレンドシェイプ初期値 |
| `clothingBlendShapeSettings` | array | 衣装側のブレンドシェイプ設定 |
| `clothingBlendShapeSettingsInv` | array | 衣装側の逆変換ブレンドシェイプ設定 |

### BlendShapeSetting オブジェクト

```json
{
    "name": "Breasts",
    "value": 0.5
}
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `name` | string | ブレンドシェイプ名 |
| `value` | float | 値（0.0〜1.0） |

### blendShapeFields 配列

複数のブレンドシェイプ変形フィールドを定義します。

```json
{
    "label": "Breasts_big",
    "sourceLabel": "Breasts",
    "path": "deformation_breasts.npz",
    "maskBones": ["Chest", "LeftBreast", "RightBreast"],
    "sourceBlendShapeSettings": [...],
    "targetBlendShapeSettings": [...]
}
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `label` | string | ターゲット側のブレンドシェイプ名 |
| `sourceLabel` | string | ソース側のブレンドシェイプ名 |
| `path` | string | この変形用の NPZ ファイルパス |
| `maskBones` | array | 変形を適用するボーンのリスト（Humanoid ボーン名） |
| `sourceBlendShapeSettings` | array | ソース側の設定 |
| `targetBlendShapeSettings` | array | ターゲット側の設定 |

> **注意**: `label` または `sourceLabel` が重複している場合、スクリプトは自動的に
> `___0`, `___1` などのサフィックスを付加します（例: `Breasts___0`, `Breasts___1`）。
> 生成されるシェイプキー名に影響するため注意してください。

## サンプル

### シンプルな設定

```json
{
    "poseDataPath": "posediff_template_to_mao.json",
    "fieldDataPath": "deformation_template_to_mao.npz",
    "sourceBlendShapeSettings": [
        {"name": "Breasts_flat", "value": 0.469},
        {"name": "Highheels", "value": 0.856}
    ],
    "targetBlendShapeSettings": [],
    "blendShapeFields": [],
    "baseAvatarDataPath": "avatar_data_mao.json",
    "clothingAvatarDataPath": "avatar_data_template.json",
    "clothingBlendShapeSettings": [],
    "clothingBlendShapeSettingsInv": []
}
```

### ブレンドシェイプフィールド付き設定

```json
{
    "poseDataPath": "posediff_beryl_to_template.json",
    "fieldDataPath": "deformation_beryl_to_template.npz",
    "sourceBlendShapeSettings": [],
    "targetBlendShapeSettings": [],
    "blendShapeFields": [
        {
            "label": "Breasts_big",
            "sourceLabel": "Breasts",
            "path": "deformation_beryl_to_template_breasts_big.npz",
            "maskBones": [
                "Chest", "LeftBreast", "LeftShoulder",
                "RightBreast", "RightShoulder", "Spine"
            ],
            "sourceBlendShapeSettings": [
                {"name": "Breasts", "value": 1.0}
            ],
            "targetBlendShapeSettings": [
                {"name": "Breasts_big", "value": 1.0}
            ]
        }
    ],
    "baseAvatarDataPath": "avatar_data_template.json",
    "clothingAvatarDataPath": "avatar_data_beryl4.json",
    "clothingBlendShapeSettings": [],
    "clothingBlendShapeSettingsInv": []
}
```

## パス解決

すべてのパスは config JSON ファイルからの相対パスとして解決されます。
通常、これらのファイルは Unity プロジェクトの同じディレクトリに配置されます。

```
Assets/OutfitRetargetingSystem/Editor/
├── config_source_to_target.json
├── posediff_source_to_target.json
├── deformation_source_to_target.npz
├── avatar_data_source.json
└── avatar_data_target.json
```

## 関連ドキュメント

- [処理フロー概要](overview.md)
- [Blender アドオン - データフォーマット](../blender-addon/data_formats.md)
