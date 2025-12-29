# BlenderTools/dev

Unity Outfit Retargeting System から呼び出される Blender スクリプト群です。

## ライセンス

このディレクトリ内のスクリプトは **GPL v3** ライセンスの下で公開されています。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `retarget_script2_14.py` | メインのリターゲット処理スクリプト |

## retarget_script2_14.py

### 概要

Unity の Outfit Retargeting System から subprocess として呼び出され、
衣装メッシュを異なるアバター体型にリターゲットするスクリプトです。

### 主な機能

- FBX ファイルのインポート/エクスポート
- posediff JSON からのポーズ適用（delta_matrix）
- deformation NPZ からの変形フィールド適用
- ウェイト転送
- Humanoid ボーン置換
- BlendShape フィールド処理
- 多段階リターゲット（チェーン処理）

### 必要環境

- Blender 4.0+
- Python 3.10+ (Blender 内蔵)
- NumPy, SciPy

### 使用方法

```bash
blender --background --python retarget_script2_14.py -- \
    --input <衣装FBX> \
    --output <出力FBX> \
    --base-fbx <ベースアバターFBX> \
    --config <config.json> \
    [その他オプション]
```

### 主要な引数

| 引数 | 説明 |
|------|------|
| `--input` | 入力衣装 FBX ファイルパス |
| `--output` | 出力 FBX ファイルパス |
| `--base-fbx` | ベースアバター FBX（セミコロン区切りで複数指定可） |
| `--config` | config JSON ファイル（セミコロン区切りで複数指定可） |
| `--hips-position` | Hips ボーンのターゲット位置 (x,y,z) |
| `--init-pose` | 初期ポーズ JSON |
| `--target-meshes` | 処理対象メッシュ名（セミコロン区切り） |

### チェーン処理

複数の config を指定することで、チェーン処理（例: Rurune → Template → mao）が可能です：

```bash
blender --background --python retarget_script2_14.py -- \
    --input clothing.fbx \
    --output output.fbx \
    --base-fbx "template.fbx;mao.fbx" \
    --config "config_rurune2template.json;config_template2mao.json"
```

## 関連ドキュメント

- [Unity アドオン処理フロー](../../../docs/unity-addon/overview.md)
- [config JSON 仕様](../../../docs/unity-addon/config_format.md)

## 既知の問題

- チェーン処理時にメモリが解放されず、大規模処理でメモリ不足になる可能性がある
