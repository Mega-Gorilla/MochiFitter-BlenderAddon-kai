# もちふぃった～ Blenderアドオン改

もちふぃった～向け Blenderアドオンのバグ修正・およびnpz出力の高速化をした アドオンです。

本アドオンは以下のURLにてGPL-3.0で公開されています
https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai.git

※Booth版はサポートの関係で有料にて配布しております

## ✨ 主な改善点

### 🚀 RBF処理の高速化・最適化
NPZ生成時のRBF補間処理を大幅に高速化しました。

| 最適化 | 効果 |
|--------|------|
| float32精度 | メモリ使用量 ~50%削減、速度 10-20%向上 |
| 動的バッチサイズ | 利用可能メモリに応じて最適化（1,000〜20,000） |
| Numba JIT 距離計算 | 3-5倍高速化（オプション） |
| ハイブリッド並列化 | ThreadPoolExecutor対応でCPU効率向上 |

結果: 140万頂点のメッシュ処理が 52秒 → 約20秒 に短縮（環境による）

### 🔧 NumPy・SciPy再インストールのバグ修正
Microsoft Store版Blenderで発生していたインストール失敗問題を完全に解決しました。

修正した問題:
- [WinError 183] - AppContainer サンドボックスでの makedirs 失敗
- [WinError 17] - pip のクロスドライブ移動エラー
- UnicodeDecodeError - Windows コンソールのエンコーディング問題
- UI フリーズ - 長時間処理での「応答なし」状態

新しいインストール方式:
- pip download + wheel手動展開方式を採用
- 安全なディレクトリ置換で既存ファイルを保護
- Modal Operator化でUIフリーズを解消

### 🖥️ UI応答性の改善
NPZ生成中もBlender UIが応答し続けるようになりました。

- Modal Operator化: バックグラウンドスレッドで処理実行
- キャンセル機能: ESCキーで処理中断可能
- 進捗表示: フェーズ表示＋プログレスバー
- BLASスレッド制限: システム全体への負荷を軽減

## インストール方法

1. MochiFitter-BlenderAddon.zip をダウンロード（最新リリース）
2. Blender の Edit → Preferences → Add-ons よりすでにインストールしているMochiFitterアドオンをアンインストール
3. ダウンロードしたZipファイルをBlenderウインドウにドラッグアンドドロップ
4. インストール後Add-onsウインドウに、MochiFitter Kaiが表示されることを確認する

## 【サポート】

Discordサーバーにてサポートを行っております！
私１人で開発を行っているので、デバッグが不十分な点が多々あります！
「動かない！」「要望」等のご意見はお気軽に！
https://discord.gg/raPrPwwW9z

## アップデートログ

アップデートログは以下を参照してください
https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/blob/master/CHANGELOG.md
