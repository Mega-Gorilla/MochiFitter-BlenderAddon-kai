# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unity側Blenderスクリプト (`retarget_script2_10.py`) をリポジトリに追加
  - GPL v3 ライセンスのため公開可能
  - パス: `MochFitter-unity-addon/BlenderTools/blender-4.0.2-windows-x64/dev/`

### Changed
- `delta_matrix` を最優先に変更 - 旧形式JSONとの完全互換性を実現
  - `delta_matrix` が存在する場合はそれを直接使用（最も正確）
  - `delta_matrix` がない新形式JSONでのみ `location`, `rotation`, `scale` から再構築
  - 旧形式JSONでの rotation 単位問題（ラジアン vs 度）を回避

### Fixed
- `BrokenProcessPool` エラーを修正 - Windows での安定性向上
  - `max_workers` を最大8に制限 (`min(8, os.cpu_count())`)
  - 32プロセス同時起動によるメモリ不足を回避
- 旧形式posediff JSONでの座標破綻を修正
  - 旧形式はrotationをラジアンで保存、新形式は度で保存
  - `delta_matrix` 優先により旧形式JSONでの読み込み精度が向上

### Removed
- 未使用の `matrix_to_list()` 関数を削除

## [0.2.1] - 2025-12-14

### Added
- NumPy/SciPy 再インストール機能の大幅な改善 (Issue #7)
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

## [0.1.0] - 2024-12-14

### Added
- GitHub リポジトリを公開
- CLAUDE.md - Claude Code 向けガイドドキュメント
- README.md - プロジェクト説明ドキュメント

### Changed
- アドオン名を「MochiFitter」から「MochiFitter-Kai」に変更
- 作者情報をコミュニティフォークとして更新
- バージョンを 0.1.0 にリセット（フォーク版として独自バージョニング開始）
- `delta_matrix` フィールドを廃止 - posediff JSON から削除
  - ユーザーが `location`, `rotation`, `scale` を直接編集可能に
  - Unity 側は `hasDeltaMatrix` フォールバックで互換性維持

### Removed
- `save_armature_pose()` から `delta_matrix` の保存処理を削除
- `add_pose_from_json()` から `delta_matrix` 優先ロジックを削除

### Fixed
- ユーザーが posediff JSON の scale 値を編集しても反映されなかった問題を修正

### Notes
- **精度に関する注意**: 以前は保存された `delta_matrix` をそのまま適用していましたが、
  現在は `location`, `rotation`, `scale` から行列を再構築しています。
  非等方スケールや特殊な回転を含む場合、わずかな数値誤差が生じる可能性があります。

---

## Base Version

This project is forked from [MochiFitter](https://booth.pm/ja/items/7657840) version 2.5.0.

[Unreleased]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/releases/tag/v0.1.0
