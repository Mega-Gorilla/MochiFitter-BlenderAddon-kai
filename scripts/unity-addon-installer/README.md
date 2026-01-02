# MochiFitter-Kai Optimization Installer

Unity アドオン（MochiFitter）に最適化パッチを適用するインストーラーです。

## 概要

このインストーラーは以下のファイルを最適化します：

| ファイル | 配布方法 | 効果 |
|---------|---------|------|
| `retarget_script2_14.py` | 完全置換（GPL） | 47% 高速化 |
| `smoothing_processor.py` | パッチ適用（DLL経由） | 2544x 高速化 |

## 使用方法

### インストール

1. `install.bat` をダブルクリック
2. MochiFitter がインストールされたプロジェクトを選択
3. インストール確認で `y` を入力

### アンインストール

```cmd
install.bat -Uninstall
```

## ファイル構成

```
scripts/unity-addon-installer/
├── install.bat              # Windows バッチランチャー
├── install.ps1              # メインインストーラーロジック
├── generate_patcher.py      # C# パッチャー DLL 生成スクリプト
├── test_patcher.ps1         # テストスイート
├── files/
│   └── retarget_script2_14.py  # 最適化版スクリプト（GPL）
├── build/
│   └── MochiFitterPatcher.cs   # 生成された C# ソース
└── dist/
    └── MochiFitterPatcher.dll  # コンパイル済みパッチャー
```

## DLL ビルド手順

リポジトリに含まれる `dist/MochiFitterPatcher.dll` は以下の手順で再生成できます：

### 前提条件

- Python 3.8+
- .NET Framework 4.0+ (Windows 標準搭載)

### ビルド

```powershell
cd scripts/unity-addon-installer
python generate_patcher.py --use-manual-patches
```

### DLL ハッシュ検証

ビルド後、以下のコマンドでハッシュを確認できます：

```powershell
certutil -hashfile dist\MochiFitterPatcher.dll SHA256
```

**現在のDLLハッシュ (SHA256):**
```
835ee95c7d51ae2ed45233f501af3b6cbf4148c9ca825142c92a28ce2ad5a6e2
```

## テスト実行

### 前提条件

テストを実行するには、以下のファイルが必要です：

```
original_v34r/Editor/smoothing_processor.py
```

このファイルは MochiFitter v34r のオリジナルファイルです（有料コンテンツのため非公開）。

### テスト実行

```powershell
cd scripts/unity-addon-installer
.\test_patcher.ps1 -Verbose
```

テストディレクトリを保持する場合：

```powershell
.\test_patcher.ps1 -KeepTestDir -Verbose
```

## 技術詳細

### パッチ適用ロジック

1. **行末空白の正規化**: 行末の空白を削除してマッチング
2. **部分適用の検出**: ヘッダーに `Patches Applied: X/Y` 形式で記録
3. **再適用サポート**: 部分適用状態からの完全適用が可能

### パッチヘッダー形式

```python
# ============================================
# MochiFitter-Kai Optimized
# Version: 1.0.0
# Patches Applied: 2/2
# DO NOT REMOVE THIS HEADER
# ============================================
```

### 状態判定

| ヘッダー状態 | 動作 |
|-------------|------|
| なし | 新規パッチ適用 |
| `Patches Applied: X/Y` (X < Y) | 警告後、再適用を試行 |
| `Patches Applied: Y/Y` | スキップ（完全適用済み） |

## ライセンス

- `retarget_script2_14.py`: GPL v3
- その他のファイル: MIT License
