# OutfitRetargetingSystem

Unity 用の衣装リターゲットシステムです。

> **Note**: このフォルダの内容（README.md を除く）は `.gitignore` で除外されています。
> 有料アドオンの DLL やデータファイルを含むためです。

## 必要なファイル

このフォルダには以下のファイルを配置する必要があります：

### 必須ファイル

| ファイル | 説明 | 入手方法 |
|---------|------|---------|
| `Editor/OutfitRetargetingSystem.dll` | メイン DLL | MochiFitter Unity アドオン購入 |
| `Editor/Template.fbx` | Template アバター | MochiFitter Unity アドオンに付属 |
| `Editor/avatar_data_template.json` | Template アバター情報 | MochiFitter Unity アドオンに付属 |
| `Editor/pose_basis_template.json` | Template ベースポーズ | MochiFitter Unity アドオンに付属 |

### オプションファイル

| ファイル | 説明 |
|---------|------|
| `Editor/HumanoidBoneNamePatterns.json` | ボーン名パターン定義 |
| `Editor/smoothing_processor.py` | スムージング処理スクリプト |
| `Editor/vertex_group_weights_*.json` | 頂点グループウェイト設定 |
| `Editor/config_*.json` | リターゲット設定ファイル |
| `Editor/posediff_*.json` | ポーズ差分データ |
| `Editor/deformation_*.npz` | 変形フィールドデータ |

## フォルダ構造

```
OutfitRetargetingSystem/
├── README.md                          # このファイル（Git追跡対象）
├── Editor/
│   ├── OutfitRetargetingSystem.dll    # メイン DLL
│   ├── Template.fbx                   # Template アバター
│   ├── avatar_data_template.json      # Template 情報
│   ├── pose_basis_template.json       # Template ベースポーズ
│   ├── config_*.json                  # 各種設定
│   ├── posediff_*.json                # ポーズ差分
│   └── deformation_*.npz              # 変形フィールド
└── Outputs/                           # 出力フォルダ
    └── retargeted_*.fbx               # リターゲット済み FBX
```

## セットアップ手順

1. **MochiFitter Unity アドオンを購入**
   - BOOTH 等で購入

2. **ファイルを配置**
   - 購入したアドオンの `OutfitRetargetingSystem/` フォルダを
     このディレクトリにコピー

3. **Unity プロジェクトで使用**
   - Unity で Assets フォルダに配置
   - Outfit Retargeting System ウィンドウを開く

## CLI テスト環境（Unity を使わずにテスト）

Unity を介さずに `retarget_script2_14.py` を直接実行してテストできます。

### 環境変数の設定

以下の環境変数を設定してください：

| 環境変数 | 説明 | 例 |
|---------|------|-----|
| `BLENDER_PATH` | Blender 実行ファイルのパス | `C:\Program Files\Blender Foundation\Blender 4.0\blender.exe` |
| `RETARGET_SCRIPT_PATH` | retarget_script2_14.py のパス | `D:\BlenderTools\dev\retarget_script2_14.py` |

**PowerShell での設定例:**
```powershell
$env:BLENDER_PATH = "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"
$env:RETARGET_SCRIPT_PATH = "D:\path\to\retarget_script2_14.py"
```

**コマンドプロンプト での設定例:**
```batch
set BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 4.0\blender.exe
set RETARGET_SCRIPT_PATH=D:\path\to\retarget_script2_14.py
```

### CLI ラッパースクリプト（推奨）

`run_retarget.py` を使用すると簡単にテストできます：

```batch
python run_retarget.py --preset beryl_to_mao --benchmark
```

オプション:
- `--preset`: プリセット設定を使用（`beryl_to_mao`）
- `--benchmark`: ベンチマーク結果を記録
- `--list-presets`: 利用可能なプリセットを表示

### 必要なファイル配置

```
OutfitRetargetingSystem/
├── Editor/
│   ├── base_project.blend              # ベース Blender プロジェクト
│   ├── Template.fbx                    # Template アバター
│   ├── avatar_data_template.json       # Template アバター情報
│   ├── avatar_data_<avatar>.json       # 対象アバター情報
│   ├── config_<src>2<dst>.json         # リターゲット設定
│   ├── deformation_*.npz               # 変形フィールドデータ
│   ├── posediff_*.json                 # ポーズ差分データ
│   ├── pose_basis_*.json               # ベースポーズデータ
│   └── TestingDatasets/
│       └── <clothing>.fbx              # テスト用衣装 FBX
└── Outputs/
    └── empty_pose.json                 # 空ポーズファイル（{"bones": []}）
```

### CLI 実行例（Beryl → Template → mao）

```batch
"<BlenderPath>\blender.exe" --background ^
  --python "<BlenderPath>\dev\retarget_script2_14.py" ^
  -- ^
  --input="Editor\TestingDatasets\Beryl_Costumes.fbx" ^
  --output="Outputs\cli_test_output.fbx" ^
  --base="Editor\base_project.blend" ^
  --base-fbx="Editor\Template.fbx;<mao.fbx path>" ^
  --config="Editor\config_beryl2template.json;Editor\config_template2mao.json" ^
  --init-pose="Outputs\empty_pose.json" ^
  --hips-position=0.00000000,0.00955725,0.93028500 ^
  --target-meshes="Costume_Body;Costume_Frill_Arm;..." ^
  --blend-shapes=Highheel ^
  --blend-shape-values=1.000 ^
  --no-subdivision
```

### 注意事項

- `empty_pose.json` は `{"bones": []}` の内容で作成
- `<BlenderPath>` は Blender 4.0+ のインストールパス
- チェーン処理（例: Beryl → Template → mao）は `--base-fbx` と `--config` をセミコロンで区切る

## 関連ドキュメント

- [BlenderTools/dev/](../BlenderTools/blender-4.0.2-windows-x64/dev/) - Blender スクリプト
- [テスト環境構築ガイド](../../docs/setup/test-environment.md)
- [Unity アドオン処理フロー](../../docs/unity-addon/overview.md)

## ライセンス

- **OutfitRetargetingSystem.dll**: 有料ライセンス（再配布禁止）
- **Template.fbx**: MochiFitter アドオン付属（再配布禁止）
- **その他スクリプト**: 各ファイルのライセンスに従う
