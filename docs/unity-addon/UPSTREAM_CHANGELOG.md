# OutfitRetargetingSystem アップストリーム変更ログ

本家 [MochiFitter](https://booth.pm/ja/items/7657840) Unity アドオンの変更履歴です。
公式アップデートノートが提供されていないため、ファイル差分から推測した内容を記録しています。

---

## [32r] - 2024-12-14

**対象ファイル:**
- `OutfitRetargetingSystem.dll` (243,200 → 263,168 bytes, +20KB)
- `smoothing_processor.py` (30,259 → 33,958 bytes, +3.7KB)

### Changed (smoothing_processor.py)

#### メモリ最適化
- 共有 `neighbors_cache` を廃止
- 各ワーカープロセスで個別に `cKDTree` を構築するように変更
- 大規模データでのメモリ消費を抑制

```python
# 変更前: 共有キャッシュをプロセス間で受け渡し
neighbors_cache = build_neighbors_cache(vertex_coords, smoothing_radius)

# 変更後: 各プロセス内でcKDTreeを構築
kdtree = cKDTree(vertex_coords)
neighbor_indices = kdtree.query_ball_point(vertex_coords[i], smoothing_radius)
```

#### タイムアウト処理の追加
- `DEFAULT_WORKER_TIMEOUT = 600` (10分) を追加
- `FuturesTimeoutError` のハンドリングを追加
- タイムアウト時のフォールバック処理を実装

```python
for future in as_completed(future_to_group, timeout=DEFAULT_WORKER_TIMEOUT):
    try:
        result = future.result(timeout=DEFAULT_WORKER_TIMEOUT)
    except FuturesTimeoutError:
        # フォールバック処理
        failed_groups.append(worker_arg)
```

#### フォールバック処理
- 並列処理で失敗したグループをシングルスレッドで再処理
- 最終手段としてオリジナルウェイトを使用
- 処理の堅牢性が向上

#### Windows 互換性強化
- `freeze_support()` を追加（Windows マルチプロセス対応）
- `max_workers = os.cpu_count() - 1` に変更（1コア確保）

#### エラーハンドリング改善
- 例外発生時の traceback 出力を追加
- グループ単位でのエラーリカバリ

### Changed (OutfitRetargetingSystem.dll)

DLL のサイズが約 20KB 増加。バイナリのため詳細な変更内容は不明ですが、
smoothing_processor.py の API 変更に対応した更新と推測されます。

---

## [30r] - 2024-12-08 (推定)

リポジトリ初期取り込み時のバージョン。

### Known Issues
- `neighbors_cache` 共有方式によるメモリ消費
- タイムアウト処理なし
- フォールバック処理なし

---

## バージョン表記について

- `r` は BOOTH でのリリース番号を示します
- 正確なバージョン番号は DLL 内に埋め込まれている可能性がありますが、確認できていません
- 日付はファイルの更新日時から推測しています

## 関連リンク

- [MochiFitter (BOOTH)](https://booth.pm/ja/items/7657840)
- [MochiFitter-Kai リポジトリ](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai)
