# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unity側Blenderスクリプト (`retarget_script2_10.py`) をリポジトリに追加
  - GPL v3 ライセンスのため公開可能
  - `location`, `rotation`, `scale` 優先 + `delta_matrix` フォールバック対応済み
  - パス: `MochFitter-unity-addon/BlenderTools/blender-4.0.2-windows-x64/dev/`

### Changed
- `delta_matrix` をフォールバックとして再実装 - 後方互換性を強化
  - `location`, `rotation`, `scale` フィールドを優先的に使用
  - これらが存在しない古いJSONでは `delta_matrix` にフォールバック
  - KeyError による読み込み失敗を防止

### Removed
- 未使用の `matrix_to_list()` 関数を削除

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

[Unreleased]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/releases/tag/v0.1.0
