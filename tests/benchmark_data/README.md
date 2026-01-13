# Benchmark Data

このフォルダーには、RBF処理のベンチマークに必要なデータファイルを配置します。

**注意**: このフォルダー内の大きなファイル（*.npz等）はgitで追跡されません。

## 必要なファイル

### 1. temp_rbf_*.npz（RBF処理入力ファイル）

Blenderアドオンで「変形フィールドを生成」を実行した際に生成される一時ファイル。

**入手方法**:
- Blenderでソース/ターゲットアバターを設定し、変形フィールドを生成
- 作業フォルダーから既存のファイルをコピー

**推奨ファイル**:
```
temp_rbf_template_to_ririka.npz  (~43MB)
temp_rbf_template_to_mao.npz    (~65MB)
```

**配置元**:
- MochiFitter作業フォルダー内の `temp_rbf_*.npz` ファイル
- 変形フィールド生成時に自動的に作成される

### 2. MochiFitter-Original/（純正版、比較用）

純正MochiFitterアドオン（v1.0.4）を展開したフォルダー。ベンチマーク比較に使用。

**入手方法**:
1. BOOTHから購入・ダウンロード: https://booth.pm/ja/items/6051400
2. ZIPを展開し、`MochiFitter-Original/` にリネーム

## ベンチマーク実行方法

### 準備

```bash
# NPZファイルをこのフォルダーにコピー
cp /path/to/temp_rbf_template_to_ririka.npz tests/benchmark_data/
```

### 実行

```bash
# MochiFitter-Kai (Numbaなし)
NUMBA_DISABLE_JIT=1 python MochiFitter-BlenderAddon/rbf_multithread_processor.py tests/benchmark_data/temp_rbf_template_to_ririka.npz

# MochiFitter-Kai (Numbaあり) - 要: pip install numba
python MochiFitter-BlenderAddon/rbf_multithread_processor.py tests/benchmark_data/temp_rbf_template_to_ririka.npz

# 純正 v1.0.4
python tests/benchmark_data/MochiFitter-Original/rbf_multithread_processor.py tests/benchmark_data/temp_rbf_template_to_ririka.npz
```

## ファイル構造

```
benchmark_data/
├── README.md                              # このファイル
├── .gitkeep                               # フォルダー追跡用
├── temp_rbf_template_to_ririka.npz        # (ユーザー配置)
├── temp_rbf_template_to_mao.npz           # (ユーザー配置)
└── MochiFitter-Original/                  # (ユーザー配置)
    └── rbf_multithread_processor.py
```

## 期待される結果

| バージョン | 処理時間（目安） | 備考 |
|------------|------------------|------|
| 純正 v1.0.4 | ~52秒 | ベースライン |
| Kai (Numbaなし) | ~20秒 | float32 + 動的バッチサイズ |
| Kai (Numbaあり) | ~15秒 | + Numba JIT距離計算 |

※ 処理時間は環境（CPU、メモリ）により大きく異なります。
