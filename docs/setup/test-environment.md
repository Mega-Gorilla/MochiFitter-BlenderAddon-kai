# テスト環境構築ガイド

MochiFitter の開発・テスト環境を構築するためのガイドです。

## 概要

MochiFitter は以下のコンポーネントで構成されています：

```
┌─────────────────────────────────────────────────────────────┐
│                    MochiFitter 構成                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Blender アドオン]          [Unity アドオン]                │
│  MochiFitter-BlenderAddon/   MochFitter-unity-addon/        │
│  ├── SaveAndApplyFieldAuto.py    ├── BlenderScripts/        │
│  └── (公開・GPL v3)              │   └── retarget_script    │
│                                  │       (公開・GPL v3)      │
│                                  │                          │
│                                  └── OutfitRetargetingSystem/│
│                                      └── (非公開・有料)      │
│                                                             │
│  [テストデータ]                                              │
│  profile_data/                                              │
│  └── (非公開・著作権あり)                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 必要なソフトウェア

| ソフトウェア | バージョン | 用途 |
|-------------|-----------|------|
| Blender | 4.0+ | リターゲット処理実行 |
| Python | 3.10+ | スクリプト実行 |
| Git | 最新 | バージョン管理 |
| Unity | 2019.4+ | (オプション) Unity アドオン使用時 |

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai.git
cd MochiFitter-BlenderAddon-kai
```

### 2. profile_data フォルダの作成

テスト用データを格納するフォルダを作成します：

```bash
mkdir -p profile_data/fbx
```

### 3. テストデータの配置

`profile_data/` フォルダに以下のファイルを配置します：

#### 必須ファイル（Issue #15 検証用）

| ファイル | 説明 | 入手方法 |
|---------|------|---------|
| **Avatar Data** |||
| `avatar_data_beryl4.json` | Beryl アバター情報 | Blender アドオンで生成 |
| `avatar_data_template.json` | Template アバター情報 | MochiFitter に付属 |
| `avatar_data_mao.json` | mao アバター情報 | Blender アドオンで生成 |
| **Pose Data** |||
| `pose_basis_beryl.json` | Beryl ベースポーズ | Blender アドオンで生成 |
| `pose_basis_template.json` | Template ベースポーズ | MochiFitter に付属 |
| `pose_basis_mao.json` | mao ベースポーズ | Blender アドオンで生成 |
| **Pose Diff** |||
| `posediff_beryl_to_template.json` | Beryl → Template 差分 | Blender アドオンで生成 |
| `posediff_template_to_mao.json` | Template → mao 差分 | Blender アドオンで生成 |
| **Deformation Fields** |||
| `deformation_beryl_to_template.npz` | Beryl → Template 変形 | Blender アドオンで生成 |
| `deformation_template_to_mao.npz` | Template → mao 変形 | Blender アドオンで生成 |
| **Config** |||
| `config_beryl2template.json` | Beryl → Template 設定 | Unity アドオンで生成 |
| `config_template2mao.json` | Template → mao 設定 | Unity アドオンで生成 |
| **FBX (fbx/ サブフォルダ)** |||
| `fbx/Beryl.fbx` | Beryl アバター | 有料アバター購入 |
| `fbx/Beryl_Costumes.fbx` | Beryl 衣装 | 有料アバター購入 |
| `fbx/Template.fbx` | Template アバター | MochiFitter に付属 |
| `fbx/mao.fbx` | mao アバター | 有料アバター購入 |

### 4. Unity アドオンのセットアップ（オプション）

Unity 経由でテストする場合：

```bash
# OutfitRetargetingSystem フォルダに有料アドオンを配置
cp -r /path/to/purchased/OutfitRetargetingSystem/* \
    MochFitter-unity-addon/OutfitRetargetingSystem/
```

## テスト実行

### Blender CLI でのリターゲット実行

```bash
# 単一変換（Beryl → Template）
blender --background --python MochFitter-unity-addon/BlenderScripts/retarget_script2_12.py -- \
    --input profile_data/fbx/Beryl_Costumes.fbx \
    --output output.fbx \
    --base-fbx profile_data/fbx/Template.fbx \
    --config profile_data/config_beryl2template.json

# チェーン変換（Beryl → Template → mao）
blender --background --python MochFitter-unity-addon/BlenderScripts/retarget_script2_12.py -- \
    --input profile_data/fbx/Beryl_Costumes.fbx \
    --output output.fbx \
    --base-fbx "profile_data/fbx/Template.fbx;profile_data/fbx/mao.fbx" \
    --config "profile_data/config_beryl2template.json;profile_data/config_template2mao.json"
```

### Python テスト実行

```bash
# pytest でテスト実行（Issue #4 で構築予定）
pytest tests/ -v
```

## フォルダ構造（完成形）

```
MochiFitter-BlenderAddon-kai/
├── MochiFitter-BlenderAddon/          # Blender アドオン本体
├── MochFitter-unity-addon/
│   ├── BlenderScripts/                # リターゲットスクリプト（公開）
│   │   ├── README.md
│   │   └── retarget_script2_12.py
│   └── OutfitRetargetingSystem/       # Unity アドオン（非公開）
│       ├── README.md                  # セットアップ説明（公開）
│       └── Editor/                    # 有料 DLL 等
├── profile_data/                      # テストデータ（非公開）
│   ├── README.md                      # ファイル説明（公開）
│   ├── *.json                         # 設定・ポーズデータ
│   ├── *.npz                          # 変形フィールド
│   └── fbx/                           # FBX ファイル
└── docs/
    └── setup/
        └── test-environment.md        # このファイル
```

## トラブルシューティング

### Blender が見つからない

```bash
# Blender のパスを確認
which blender  # Linux/Mac
where blender  # Windows

# パスを指定して実行
/path/to/blender --background --python ...
```

### SciPy が見つからない

```bash
# Blender 内蔵 Python に SciPy をインストール
/path/to/blender/4.0/python/bin/pip install scipy
```

### FBX インポートエラー

- Blender のバージョンを確認（4.0+ 必須）
- FBX ファイルが破損していないか確認
- 正しいパスを指定しているか確認

## 関連ドキュメント

- [profile_data README](../../profile_data/README.md)
- [OutfitRetargetingSystem README](../../MochFitter-unity-addon/OutfitRetargetingSystem/README.md)
- [BlenderScripts README](../../MochFitter-unity-addon/BlenderScripts/README.md)
- [Issue #4: CLI テスト環境構築](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/issues/4)
- [Issue #15: チェーン処理時の Hips 位置・スケール調整](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/issues/15)
