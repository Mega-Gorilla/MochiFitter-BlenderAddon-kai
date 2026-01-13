# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.20] - 2026-01-13

### Fixed
- Microsoft Store 版 Blender で NPZ 生成がハングする問題を修正 (Issue #70, PR #71)
  - Numba JIT の `cache=True` が AppContainer サンドボックス環境でキャッシュディレクトリへのアクセス時にハング
  - `cache=False` に変更し、毎回 JIT コンパイルを実行（数秒の遅延）
  - 対象関数: `_cdist_sqeuclidean_numba()`, `_cdist_euclidean_numba()`

## [0.2.19] - 2026-01-13

### Added
- 依存パッケージ再インストールボタンに Numba を追加 (Issue #68, PR #69)
  - Numba JIT 最適化が再インストールボタンから自動インストール可能に
  - UI に Numba バージョン/ステータス表示を追加
  - セクションタイトルを「依存パッケージ管理」に変更
  - ボタンテキストを「依存パッケージ 再インストール」に変更

### Changed
- UI表示を「MochiFitter」から「MochiFitter-Kai」に統一 (PR #69)
  - サイドバータブ名、パネルラベルを変更
  - コンソールメッセージを更新

### Fixed
- Numba インストール時の安全性強化 (PR #69)
  - Numba を別ステップで試行し、失敗時も NumPy/SciPy/psutil は正常にインストール
  - Numba wheel 展開時に numpy/scipy をスキップし、ピン留めバージョンを保持

## [0.2.18] - 2026-01-07

### Added
- 出力ファイル名を小文字に統一 (Issue #64, PR #66)
  - `normalize_avatar_name_for_filename()` ヘルパー関数を追加
  - Unity拡張側との互換性のため、ファイル名のアバター名部分を小文字に変換
  - Linuxでの大文字/小文字による別ファイル認識問題を解消
  - 対象: `pose_basis_*.json`, `posediff_*_to_*.json`, `deformation_*.npz`, `temp_rbf_*.npz`

### Fixed
- 後方互換性のため旧ファイル名へのフォールバックを追加 (PR #66)
  - `find_field_data_file()` ヘルパー関数を追加
  - 小文字ファイル名が見つからない場合、旧来の大文字混在ファイル名にフォールバック
  - フォールバック時に「Note: Using legacy filename...」を表示

### Removed
- Unity アドオン最適化の開発を終了
  - 本リポジトリでのUnity アドオン開発は中止
  - Blender アドオンの開発に注力

## [0.2.17] - 2025-12-29

### Added
- Linux/macOS プラットフォームサポート (Issue #32, PR #33)
  - ディレクトリ作成処理のクロスプラットフォーム化
  - psutil 未インストール時の graceful degradation
  - CPU親和性設定の psutil 依存チェック追加
  - README にプラットフォーム対応表と Linux ユーザー向け手順を追加

## [0.2.16] - 2025-12-29

### Added
- リリースビルドスクリプト `scripts/build_release.py` を追加
  - Blender アドオンとして正しくインストール可能なZIPを生成
  - psutil をバンドル、numpy/scipy は除外
- GitHub Actions ワークフロー `.github/workflows/release.yml` を追加
  - 手動トリガーでリリースを自動生成
  - ZIPファイルを自動添付
- GPL-3.0 ライセンスファイルをリポジトリルートに追加
  - GitHub がライセンスを正しく検出するように

### Changed
- psutil を deps/ にバンドル (BSD-3-Clause ライセンス)
  - メモリ監視機能が即座に利用可能に
  - 再インストール時も psutil を含めるよう修正

### Fixed
- リリースZIPがBlenderアドオンとして認識されない問題を修正 (Issue #31)
  - `build_release.py` による正しいZIP構造で解決
  - GitHub自動生成ZIPの二重フォルダ問題を回避

## [0.2.15] - 2025-12-26

### Changed
- コンソール・ステータス出力を英語に統一 (PR #28)
  - 文字化け問題の解消
  - 国際的な利用環境への対応

## [0.2.14] - 2025-12-17

### Added
- RBF処理高速化 Phase 2 実装 (PR #27)
  - P2-1: Numba JIT 高速化 - 距離計算が 3-5倍高速化（オプション機能）
  - P2-2: GMRES 反復ソルバー - 実験的機能として追加
  - P2-3: ハイブリッド並列化 - ThreadPoolExecutor 対応でCPU効率向上

### Fixed
- GMRES の安全性強化・GIL 解放対応

## [0.2.13] - 2025-12-17

### Added
- RBF処理高速化 Phase 0-1 実装 (PR #26)
  - P0-1: float32 精度への変更 - メモリ使用量 ~50%削減、速度 10-20%向上
  - P1-2: バッチサイズ動的最適化 - 利用可能メモリに応じて 1,000〜20,000 に最適化

### Fixed
- lstsq フォールバックを float64 で実行（最大安定性確保）

## [0.2.12] - 2025-12-17

### Added
- RBF処理 高速化・最適化 実装計画ドキュメント (PR #25)
  - `docs/blender-addon/rbf-optimization-plan.md`

## [0.2.11] - 2025-12-17

### Fixed
- 線形システム求解のパフォーマンス改善 (PR #23)
  - 行列ソルバーの最適化

## [0.2.10] - 2025-12-15

### Added
- Phase 3: 進捗UI強化 (PR #22)
  - フェーズ表示（ステージ名）をステータスバーに表示
  - プログレスバー表示

## [0.2.9] - 2025-12-15

### Added
- Phase 2: キャンセル機能強化 (PR #21)
  - ESCキーで処理中断可能
  - 子プロセスの確実な終了
  - 一時ファイルのクリーンアップ

## [0.2.8] - 2025-12-15

### Fixed
- deformation_*.npz 生成時の UI ブロック問題を解消 (Issue #19, PR #20)
  - Modal Operator 化でバックグラウンド処理に移行
  - Blender が「応答なし」にならない
  - Queue の競合問題を修正

## [0.2.7] - 2025-12-15

### Added
- Unity アドオン リターゲット処理のドキュメント (PR #18)
  - Execute Retargeting コマンドの詳細説明
  - pose_basis ファイル形式の解説

## [0.2.6] - 2025-12-14

### Added
- Phase 3 統合ドキュメント (PR #14)
  - BlenderとUnityの連携フロー説明

## [0.2.5] - 2025-12-14

### Added
- Phase 2 Unity アドオンドキュメント (PR #13)
  - Blender スクリプトのドキュメント

## [0.2.4] - 2025-12-14

### Changed
- data_formats.md を delta_matrix 復活に合わせて更新 (PR #12)

## [0.2.3] - 2025-12-14

### Removed
- Unity Blender スクリプトをリポジトリから削除 (PR #11)
  - 有料コンテンツのため公開リポジトリから除外

## [0.2.2] - 2025-12-14

### Changed
- `delta_matrix` の出力を復活 (Issue #1, PR #10)
  - Unity パッケージ同梱スクリプトとの互換性を維持
  - `save_armature_pose()` で `delta_matrix` を JSON に保存
  - `location/rotation/scale` も同時に保存（参考値として）
- `add_pose_from_json()` は `delta_matrix` を最優先で使用
  - `delta_matrix` が存在する場合はそれを直接使用（最も正確）
  - `delta_matrix` がない場合のみ `location`, `rotation`, `scale` から再構築

## [0.2.1] - 2025-12-14

### Added
- NumPy/SciPy 再インストール機能の大幅な改善 (Issue #7, PR #8)
  - Microsoft Store 版 Blender (AppContainer サンドボックス) に完全対応
  - `pip download` + 手動 wheel 展開方式を採用（`pip install --target` の問題を回避）
  - `cmd /c mkdir` によるディレクトリ作成（`os.makedirs()` の `[WinError 183]` を回避）
  - 安全なディレクトリ置換（`deps_new` → `deps` 方式で pip 失敗時も既存 deps を保持）
- Modal Operator 化による非同期インストール
  - インストール中に Blender が「応答なし」にならない
  - ステータスバーにアニメーション付きメッセージを継続表示
  - 完了時にポップアップ通知を表示
- `safe_decode()` / `run_subprocess_safe()` ヘルパー関数
  - Windows コンソールの UnicodeDecodeError を完全に回避
  - UTF-8 → CP932 → 置換モードでフォールバック
- GitHub Actions ワークフロー (`auto-version-bump.yml`)
  - PR マージ時にパッチバージョンを自動インクリメント

### Changed
- バージョン取得を `importlib.metadata.version()` に変更
  - モジュールロードを回避し、DLL ロック問題を防止
- `__init__.py` から scipy の eager import/reload を削除
  - ファイルロック防止のため、scipy は必要時にのみ遅延ロード
- スレッド安全性の改善
  - `bpy` 依存の値をメインスレッドで事前取得
  - ワーカースレッドには純 Python データのみ渡す

### Fixed
- `[WinError 183]` - AppContainer での `os.makedirs()` 失敗を修正
- `[WinError 17]` - pip のクロスドライブ移動エラーを修正
- `UnicodeDecodeError` - subprocess の `_readerthread` エラーを修正
- scipy DLL ロック問題 - eager import 削除により解消
- `BrokenProcessPool` エラーを修正 - Windows での安定性向上
  - `max_workers` を最大8に制限 (`min(8, os.cpu_count())`)
  - 32プロセス同時起動によるメモリ不足を回避

## [0.1.0] - 2025-12-13

### Added
- GitHub リポジトリを公開
- CLAUDE.md - Claude Code 向けガイドドキュメント
- README.md - プロジェクト説明ドキュメント
- `matrix_to_list()` 関数を追加（JSON保存用）

### Changed
- アドオン名を「MochiFitter」から「MochiFitter-Kai」に変更
- 作者情報をコミュニティフォークとして更新
- バージョンを 0.1.0 にリセット（フォーク版として独自バージョニング開始）

### Notes
- **これは非公式のコミュニティフォークです**
- 本家 MochiFitter は [BOOTH](https://booth.pm/ja/items/7657840) で入手可能

---

## Base Version

This project is forked from [MochiFitter](https://booth.pm/ja/items/7657840) version 2.5.0.

[Unreleased]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.20...HEAD
[0.2.20]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.19...v0.2.20
[0.2.19]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.18...v0.2.19
[0.2.18]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.17...v0.2.18
[0.2.17]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.16...v0.2.17
[0.2.16]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.15...v0.2.16
[0.2.15]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.14...v0.2.15
[0.2.14]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.13...v0.2.14
[0.2.13]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.11...v0.2.12
[0.2.11]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.10...v0.2.11
[0.2.10]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.9...v0.2.10
[0.2.9]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/releases/tag/v0.1.0
