# MochiFitter-Kai（もちふぃった～改）

**もちふぃった～ Blenderアドオンの高速化・安定性強化版**

オリジナル版の処理時間を **最大70%短縮**、Microsoft Store版Blenderでも安定動作するように改良しました。

本アドオンはGPL-3.0ライセンスで公開されています：
https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai

※BOOTH版はサポート付きのため有料で配布しております

---

## 🎯 こんな方におすすめ

- NPZ生成が遅くて困っている方
- Microsoft Store版Blenderで再インストールが失敗する方
- 処理中にBlenderがフリーズして困っている方

---

## ✨ 主な改善点

### 🚀 RBF処理の大幅な高速化

NPZ生成時間を劇的に短縮しました。

**ベンチマーク結果（実測値）：**
| バージョン | 処理時間 | 高速化率 |
|-----------|---------|---------|
| オリジナル v1.0.4 | 196秒 | - |
| MochiFitter-Kai | 180秒 | 8.5%高速 |
| MochiFitter-Kai + Numba | **140秒** | **28.6%高速** |

**最適化技術：**
| 技術 | 効果 |
|------|------|
| float32精度 | メモリ使用量 ~50%削減、速度 10-20%向上 |
| 動的バッチサイズ | 利用可能メモリに応じて自動最適化 |
| Numba JIT | 距離計算を3-5倍高速化 |
| ハイブリッド並列化 | マルチコアCPUを効率的に活用 |

### ⚡ Numba JIT 最適化がワンクリックで有効に

v0.2.19から、**依存パッケージ再インストールボタン**でNumbaも自動インストール！

1. サイドバー「MochiFitter-Kai」→「依存パッケージ管理」を開く
2. 「依存パッケージ 再インストール」ボタンをクリック
3. Blenderを再起動

これだけでNumba JIT最適化が有効になり、さらなる高速化を実現します。

### 🔧 Microsoft Store版Blender完全対応

再インストール機能のバグを完全修正しました。

**修正した問題：**
- `[WinError 183]` - AppContainerサンドボックスでの失敗
- `[WinError 17]` - pipのクロスドライブ移動エラー
- `UnicodeDecodeError` - 日本語環境でのエンコーディング問題
- UIフリーズ - 長時間処理での「応答なし」状態

### 🖥️ 快適なUI体験

処理中もBlenderが応答し続けます。

- **バックグラウンド処理**: NPZ生成中も他の作業が可能
- **キャンセル機能**: ESCキーでいつでも中断
- **進捗表示**: 現在のフェーズとプログレスバーで状況確認
- **ステータス表示**: NumPy/SciPy/Numbaのバージョンを一目で確認

---

## 📦 インストール方法

1. `MochiFitter-BlenderAddon.zip` をダウンロード
2. Blenderの `Edit → Preferences → Add-ons` で既存のMochiFitterをアンインストール
3. ZIPファイルをBlenderウィンドウにドラッグ＆ドロップ
4. Add-onsで「**MochiFitter-Kai**」が表示されることを確認
5. サイドバーに「**MochiFitter-Kai**」タブが追加されます

---

## 💬 サポート

Discordサーバーでサポートを行っています！

「動かない！」「こんな機能がほしい！」などお気軽にどうぞ：
https://discord.gg/raPrPwwW9z

---

## 📋 アップデートログ

詳細な変更履歴はGitHubをご覧ください：
https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/blob/master/CHANGELOG.md
