# Unity アドオン処理フロー概要

> **注意**: Unity アドオン (OutfitRetargetingSystem) は有料コンテンツのため、
> 本リポジトリの修正範囲外です。このドキュメントは Blender アドオンユーザー向けの
> 参考情報として提供されています。

## 概要

MochiFitter Unity アドオンは、Blender で生成されたリターゲットデータを使用して、
Unity 上でアバターの衣装を異なる体型に適合させます。

## 処理フロー

```
┌─────────────────────────────────────────────────────────────────┐
│                    MochiFitter 全体フロー                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│  1. Blender         │
│  (MochiFitter-Kai)  │
└──────────┬──────────┘
           │
           │ 生成するファイル:
           │ ├── posediff_*.json     (ポーズ差分)
           │ ├── deformation_*.npz   (変形フィールド)
           │ └── avatar_data_*.json  (ボーン階層)
           │
           ▼
┌─────────────────────┐
│  2. Unity           │
│  (設定・実行)        │
└──────────┬──────────┘
           │
           │ Unity が生成:
           │ └── config_*.json       (リターゲット設定)
           │
           │ Unity が呼び出し:
           │ └── Blender subprocess
           │     (retarget_script)
           │
           ▼
┌─────────────────────┐
│  3. Blender         │
│  (バックグラウンド)  │
│  retarget_script    │
└──────────┬──────────┘
           │
           │ 出力:
           │ └── リターゲット済み FBX
           │
           ▼
┌─────────────────────┐
│  4. Unity           │
│  (結果読み込み)      │
│  Prefab 生成        │
└─────────────────────┘
```

## 各フェーズの詳細

### Phase 1: Blender でのデータ生成

MochiFitter-Kai Blender アドオンを使用して、以下のデータを生成します：

| ファイル | 説明 |
|---------|------|
| `posediff_*.json` | ソース→ターゲット間のポーズ差分（delta_matrix 含む） |
| `deformation_*.npz` | RBF 補間による変形フィールド |
| `avatar_data_*.json` | ボーン階層と Humanoid マッピング |

詳細は [Blender アドオン - データフォーマット](../blender-addon/data_formats.md) を参照。

### Phase 2: Unity での設定

Unity アドオンの UI で以下を設定：

- ソース/ターゲットアバターの指定
- 使用するデータファイルの選択
- ブレンドシェイプの設定
- 出力オプション

設定は `config_*.json` として保存されます。

### Phase 3: リターゲット実行

Unity が Blender をバックグラウンドで起動し、リターゲット処理を実行：

```
Unity
  └── subprocess.Popen()
        └── blender --background --python retarget_script2_XX.py -- \
              --input ... --output ... --base ... --base-fbx ... \
              --config ... --init-pose ... [その他オプション]
```

> **Note**: スクリプト名はバージョンにより異なります（例: `retarget_script2_12.py`）。
> スクリプトは多数の必須引数を要求します。
> 詳細は以下を参照してください：
> - [execute_retargeting.md](execute_retargeting.md) - 実際のコマンド例とパラメータ詳細
> - [blender_script.md](blender_script.md) - スクリプト内部処理の解説

リターゲットスクリプトが行う処理：

1. FBX ファイルの読み込み
2. posediff JSON からポーズ適用（`delta_matrix` 使用）
3. deformation NPZ から変形フィールド適用
4. リターゲット済み FBX + .blend ファイルの出力

### Phase 4: 結果の読み込み

Unity が出力された FBX を読み込み、Prefab を生成します。

## データの流れ

```
Blender アドオン                    Unity アドオン
================                   ================

[Avatar A] ──┐
             ├─→ posediff_A_to_B.json ───┐
[Avatar B] ──┘                           │
                                         ▼
[Source Mesh] ─→ deformation_A_to_B.npz ─┼─→ [config_*.json]
                                         │         │
[Avatar Data] ─→ avatar_data_*.json ─────┘         │
                                                   │
                                                   ▼
                                         [Blender subprocess]
                                                   │
                                                   ▼
                                         [Retargeted FBX]
                                                   │
                                                   ▼
                                         [Unity Prefab]
```

## 関連ドキュメント

- [Execute Retargeting コマンド詳細](execute_retargeting.md) - 実際の呼び出しとパラメータ
- [config JSON 仕様](config_format.md)
- [リターゲットスクリプト解説](blender_script.md)
- [Blender アドオン - データフォーマット](../blender-addon/data_formats.md)
- [UPSTREAM_CHANGELOG](UPSTREAM_CHANGELOG.md) - 本家アドオンの変更履歴
