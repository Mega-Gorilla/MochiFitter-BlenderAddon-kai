# MochiFitter-BlenderAddon-kai

MochiFitter Blenderアドオンの最適化・リファクタリングプロジェクト

## 概要

このリポジトリは、[MochiFitter](https://booth.pm/ja/items/7657840)のBlenderアドオン部分について、コードの最適化とリファクタリングを行うプロジェクトです。

MochiFitterは、VRChat/VRMアバター向けの衣装リターゲティングシステムです。RBF（Radial Basis Function）補間と変形フィールドを使用して、異なるアバター間でメッシュ変形を転送します。

## このリポジトリの目的

- コードの可読性向上
- パフォーマンス最適化
- バグ修正
- 機能改善の検討・実装

## 含まれるもの

- `MochiFitter-BlenderAddon/` - Blenderアドオン本体（GPLv3ライセンス）

## 含まれないもの

- Unity アドオン（有料コンテンツのため非公開）

## 動作要件

- Blender 4.0以上
- NumPy
- SciPy（アドオン内の再インストールボタンで導入可能）

## 対応プラットフォーム

| プラットフォーム | サポート状況 | 備考 |
|-----------------|-------------|------|
| Windows | ✅ 完全対応 | Microsoft Store版Blenderにも対応 |
| Linux | ✅ 対応 | psutilは手動インストール推奨 |
| macOS | ⚠️ 未検証 | 動作する可能性あり |

### Linux ユーザー向け注意事項

Linux環境では、メモリ監視機能に使用する`psutil`モジュールがバンドル版では動作しません。
メモリ監視機能を有効にするには、Blender内蔵のPythonでpsutilをインストールしてください：

```bash
# Blender内蔵Pythonのパスを確認（例）
/path/to/blender/4.x/python/bin/python3 -m pip install psutil
```

psutilがなくても基本機能は動作しますが、メモリ監視とCPU親和性設定が無効になります。

## 開発環境セットアップ

ローカル開発・E2Eテストを行うには、以下の手順でBlender環境をセットアップしてください。

### 1. Blenderのインストール

```bash
python scripts/setup_blender.py
```

これにより以下がセットアップされます：
- Blender 4.0.2（もちふぃった～公式推奨バージョン）
- scipy, numpy（Blender内蔵Pythonにインストール）

### 2. robust-weight-transfer アドオンのコピー（必須）

E2Eテスト（`run_retarget.py`）を実行するには、もちふぃった～ Unityパッケージに同梱されている`robust-weight-transfer`アドオンを手動でコピーする必要があります。

```
コピー元: <MochiFitter Unity Project>/BlenderTools/blender-4.0.2-windows-x64/4.0/scripts/addons/robust-weight-transfer/
コピー先: MochFitter-unity-addon/BlenderTools/blender-4.0.2-windows-x64/4.0/scripts/addons/robust-weight-transfer/
```

> **Note**: このアドオンはGitHubリポジトリには含まれていません。もちふぃった～を購入後、Unityプロジェクトからコピーしてください。

### 3. E2Eテストの実行

```bash
cd MochFitter-unity-addon/OutfitRetargetingSystem
python run_retarget.py --preset beryl_to_mao
```

## ライセンス

このプロジェクトはGNU General Public License v3.0の下で公開されています。詳細は[LICENSE.txt](MochiFitter-BlenderAddon/LICENSE.txt)を参照してください。

## 関連リンク

- [MochiFitter（BOOTH）](https://booth.pm/ja/items/7657840) - オリジナル製品ページ

## 注意事項

このリポジトリは非公式のコミュニティプロジェクトです。オリジナルのMochiFitterに関するサポートや問い合わせは、BOOTHの製品ページをご確認ください。
