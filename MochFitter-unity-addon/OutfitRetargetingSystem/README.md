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

## 関連ドキュメント

- [BlenderScripts](../BlenderScripts/) - Blender 側スクリプト
- [テスト環境構築ガイド](../../docs/setup/test-environment.md)
- [Unity アドオン処理フロー](../../docs/unity-addon/overview.md)

## ライセンス

- **OutfitRetargetingSystem.dll**: 有料ライセンス（再配布禁止）
- **Template.fbx**: MochiFitter アドオン付属（再配布禁止）
- **その他スクリプト**: 各ファイルのライセンスに従う
