# retarget_script2_14.py 問題点分析

## 1. 概要

本ドキュメントは、`retarget_script2_14.py` のコードレビューにより特定された問題点を整理したものです。

### 1.1 対象ファイル情報

| 項目 | 値 |
|------|-----|
| パス | `MochFitter-unity-addon/BlenderTools/blender-4.0.2-windows-x64/dev/retarget_script2_14.py` |
| 総行数 | 20,993行 |
| 関数数 | 231 |
| クラス数 | 1 (TransitionCache) |

### 1.2 関連Issue

- [Issue #36: retarget_script2_14.py: コード品質改善](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/issues/36)
- [Issue #34: チェーン処理時のメモリリーク](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/issues/34) (Closed - PR #35 で解決)

---

## 2. 問題一覧

### 2.1 サマリー

| 優先度 | 問題 | 件数 | 影響度 |
|--------|------|------|--------|
| 🔴 高 | 関数の重複定義 | 1 | 意図しない動作 |
| 🔴 高 | ハードコードされたオブジェクト名 | 4 | クラッシュ |
| 🔴 高 | 裸の `except:` 句 | 5 | デバッグ困難 |
| 🔴 高 | BMesh メモリリーク | 3関数 | メモリ不足 |
| 🟡 中 | グローバル変数の多用 | 9個 | テスト困難 |
| 🟡 中 | 巨大な単一ファイル | 21,000行 | 保守性低下 |
| 🟢 低 | マジックナンバー | 複数 | 可読性低下 |
| 🟢 低 | コメントアウトコード | 複数 | 可読性低下 |

---

## 3. 重大な問題（🔴 高優先度）

### 3.1 関数の重複定義

同一の関数が2箇所で定義されています。Python では後の定義が前を上書きするため、意図しない動作の原因となります。

| 関数名 | 1回目定義 | 2回目定義 |
|--------|-----------|-----------|
| `calculate_weight_pattern_similarity` | line 6334 | line 14175 |

**問題の詳細**:
```python
# line 6334
def calculate_weight_pattern_similarity(weights1: Dict[str, float], weights2: Dict[str, float]) -> float:
    # 実装 A
    ...

# line 14175 (この定義が実際に使用される)
def calculate_weight_pattern_similarity(weights1: Dict[str, float], weights2: Dict[str, float]) -> float:
    # 実装 B (実装 A を上書き)
    ...
```

**影響**:
- line 6334 の実装が呼び出されることはない
- 2つの実装に差異がある場合、予期しない動作を引き起こす
- コードの理解を妨げる

---

### 3.2 ハードコードされたオブジェクト名

lines 20344-20347 でオブジェクト名がハードコードされています。

```python
# lines 20344-20347
bpy.data.objects.remove(bpy.data.objects["Body.Template"], do_unlink=True)
bpy.data.objects.remove(bpy.data.objects["Body.Template.Eyes"], do_unlink=True)
bpy.data.objects.remove(bpy.data.objects["Body.Template.Head"], do_unlink=True)
bpy.data.objects.remove(bpy.data.objects["Armature.Template"], do_unlink=True)
```

**問題点**:
1. オブジェクトが存在しない場合 `KeyError` でクラッシュ
2. Template 以外のアバターでは動作しない
3. オブジェクト名が変更されると動作しなくなる

**影響**:
- 特定条件でプログラムがクラッシュ
- 汎用性の欠如

---

### 3.3 裸の `except:` 句

Python の裸の `except:` は `BaseException` を含むすべての例外をキャッチします。これには `KeyboardInterrupt`（Ctrl+C）や `SystemExit`（`sys.exit()`）も含まれるため、問題があります。

#### 検出された5箇所

| # | 行番号 | 関数 | コンテキスト | 現在のコード |
|---|--------|------|-------------|-------------|
| 1 | 1981 | `parse_args` | hips_position パース時 | `except: print("Error..."); sys.exit(1)` |
| 2 | 3202 | `triangulate_mesh` | エラー後のモード復元 | `except: pass` |
| 3 | 6501 | `inverse_bone_deform_all_vertices` | 逆行列計算失敗時 | `except: inverse_matrix = Matrix.Identity(4)` |
| 4 | 19764 | `export_armature_bone_data_to_json` | finally でのモード復元 | `except: pass` |
| 5 | 20964 | `main` | エラー時のシーン保存 | `except: pass` |

#### 裸の `except:` が問題となる理由

```python
# 問題のあるコード
try:
    long_running_process()
except:
    pass  # Ctrl+C を押してもプログラムが終了しない

# 継承階層
BaseException
├── SystemExit          # sys.exit() - 裸の except でキャッチされる
├── KeyboardInterrupt   # Ctrl+C - 裸の except でキャッチされる
├── GeneratorExit       # ジェネレータ終了 - 裸の except でキャッチされる
└── Exception           # 通常の例外 ← except Exception: はここ以下のみ
    ├── ValueError
    ├── TypeError
    └── ...
```

**影響**:
- Ctrl+C でプログラムを中断できない場合がある
- `sys.exit()` が正常に動作しない場合がある
- エラーの詳細が隠蔽されデバッグが困難になる

---

### 3.4 BMesh メモリリーク

`bmesh.from_mesh()` で生成した BMesh オブジェクトが解放されていない箇所があります。

| 関数 | 未解放の BMesh | 行番号 |
|------|---------------|--------|
| `create_hinge_bone_group` | `cloth_bm` | 4800 |
| `transfer_weights_from_nearest_vertex` | `body_bm`, `cloth_bm` | 11475, 11490 |
| `transfer_weights_x_projection` | `template_bm`, `target_bm` | 11821, 11838 |

**問題のあるパターン**:
```python
# 問題のあるコード
def some_function():
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # ... 処理 ...
    # bm.free() がない！
    return result  # BMesh がメモリに残る
```

**正しいパターン**:
```python
# 修正後
def some_function():
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        # ... 処理 ...
        return result
    finally:
        bm.free()  # 必ず解放
```

**影響**:
- チェーン処理時にメモリが累積
- 大規模処理でメモリ不足（OOM）発生
- Issue #34 で報告された問題の一部

---

## 4. 中程度の問題（🟡 中優先度）

### 4.1 グローバル変数の多用

9個のグローバル変数が複数の関数で共有されています。

| 変数 | 用途 | 使用関数数 |
|------|------|-----------|
| `_mesh_cache` | メッシュキャッシュ | 4 |
| `_deformation_field_cache` | NPZ/KDTree キャッシュ | 3 |
| `_saved_pose_state` | ポーズ状態保存 | 4 |
| `_previous_pose_state` | 前回ポーズ状態 | 2 |
| `_armature_record_data` | アーマチュア記録 | 3 |
| `_unity_script_directory` | Unity ディレクトリ | 2 |
| `_is_A_pose` | A ポーズフラグ | 3 |
| `_numpy_checked` | numpy チェック済フラグ | 1 |
| `_scipy_checked` | scipy チェック済フラグ | 1 |

**問題点**:
1. **テストが困難**: グローバル状態のリセットが必要
2. **状態管理の複雑化**: どの関数がどの状態を変更するか追跡困難
3. **マルチスレッド非対応**: 並列実行時に競合状態が発生
4. **モジュール分割の障壁**: 分割時にグローバル状態の管理が複雑になる

---

### 4.2 巨大な単一ファイル

21,000行の単一ファイルは以下の問題を引き起こします：

**保守性の問題**:
- 特定の機能を見つけるのに時間がかかる
- 変更の影響範囲が把握しづらい
- コードレビューが困難

**開発効率の問題**:
- IDE のパフォーマンス低下
- Git の差分が見づらい
- 複数人での同時編集が困難

**関数カテゴリ分析**:

| カテゴリ | 関数数 | 代表的な関数 |
|---------|--------|-------------|
| I/O (ロード/エクスポート) | 14 | `load_avatar_data`, `import_fbx`, `export_fbx` |
| 処理 (process_/apply_) | 35 | `process_single_config`, `apply_blendshape_values` |
| 取得/計算 (get_/find_/calculate_) | 48 | `get_deformation_field`, `calculate_obb` |
| ユーティリティ | ~82 | BMesh操作、ウェイト処理、ボーン操作 |

---

## 5. 軽微な問題（🟢 低優先度）

### 5.1 マジックナンバー

意味のない数値がコード中に直接記述されています。

```python
# 問題のあるコード
if len(loose_verts) >= 1000:      # line 20221 - なぜ1000？
if g.weight < 0.0005:             # line 20406 - なぜ0.0005？
```

**推奨される修正**:
```python
# 定数として定義
LOOSE_VERTEX_THRESHOLD = 1000  # 分離頂点の警告閾値
MIN_WEIGHT_THRESHOLD = 0.0005   # 最小ウェイト閾値

if len(loose_verts) >= LOOSE_VERTEX_THRESHOLD:
if g.weight < MIN_WEIGHT_THRESHOLD:
```

### 5.2 コメントアウトされたコード

lines 14346-14500 等に未使用のコメントアウトコードが残存しています。

**問題点**:
- 可読性の低下
- 保守性の低下
- 不要なコードとアクティブなコードの区別が困難

### 5.3 型アノテーションの不一致

一部の関数には型アノテーションがあり、一部にはありません。

```python
# 型アノテーションあり
def calculate_weight_pattern_similarity(weights1: Dict[str, float], weights2: Dict[str, float]) -> float:

# 型アノテーションなし
def get_bone_world_matrix(armature_obj, bone_name):
```

---

## 6. 分割可能性評価

### 6.1 現在の呼び出し方法

```bash
blender --background --python retarget_script2_14.py -- [引数]
```

- Unity から subprocess として呼び出される
- エントリーポイント: `main()` → `process_single_config()`
- 単一ファイルとして実行（相対インポートなし）

### 6.2 分割の障壁

#### グローバル変数による密結合

```
main()
 └── process_single_config()
      ├── process_base_avatar()
      ├── process_clothing_avatar()
      ├── apply_blendshape_deformation_fields()
      │    └── get_deformation_field()  ← _deformation_field_cache 使用
      ├── process_weight_transfer()
      │    └── apply_distance_normal_based_smoothing()
      │         └── get_cached_mesh_data()  ← _mesh_cache 使用
      └── clear_all_caches()  ← 全キャッシュクリア
```

#### Blender 環境の制約

- `bpy` モジュールは Blender Python 環境でのみ利用可能
- 分割したモジュールも同一 Blender プロセス内で実行必須
- 外部 Python からのテストが困難

### 6.3 分割可能性の結論

| 評価項目 | 結果 |
|---------|------|
| **技術的に分割可能か** | ⚠️ 可能だが大規模リファクタリングが必要 |
| **現状での分割推奨度** | 🟡 中（リスクと工数のバランス考慮） |
| **優先すべきアクション** | グローバル変数のクラス化から段階的に実施 |

---

## 7. まとめ

### 7.1 対応優先度マトリックス

| 優先度 | 問題 | 工数 | リスク |
|--------|------|------|--------|
| 🔴 高 | 重複関数の削除 | 小 | 低 |
| 🔴 高 | ハードコード箇所に存在チェック追加 | 小 | 低 |
| 🔴 高 | 裸の `except:` を `except Exception:` に変更 | 小 | 低 |
| 🔴 高 | BMesh メモリリーク修正 | 小 | 低 |
| 🟡 中 | グローバル変数をクラスにカプセル化 | 中 | 中 |
| 🟡 中 | ファイル分割 | 大 | 高 |
| 🟢 低 | マジックナンバーを定数化 | 小 | 低 |
| 🟢 低 | コメントアウトコード削除 | 小 | 低 |

### 7.2 次のステップ

詳細な修正計画については [retarget_script_refactor_plan.md](./retarget_script_refactor_plan.md) を参照してください。

---

## 更新履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-12-29 | 1.0 | 初版作成 |
