# retarget_script2_14.py リファクタリング計画

## 1. 概要

本ドキュメントは、`retarget_script2_14.py` の問題点を修正するための段階的なリファクタリング計画を記述します。

### 1.1 目標

| 目標 | 説明 |
|------|------|
| **安定性向上** | クラッシュやメモリリークの解消 |
| **保守性向上** | コードの理解と変更を容易にする |
| **テスト容易性** | ユニットテストの導入を可能にする |
| **段階的改善** | 低リスクから順に実施し、破壊的変更を避ける |

### 1.2 関連ドキュメント

- [問題点分析](./retarget_script_issues.md)
- [Issue #36](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/issues/36)

---

## 2. フェーズ構成

```
Phase 1: 緊急修正（低リスク・高効果）
├── 1.1 重複関数の削除
├── 1.2 ハードコードオブジェクト名の修正
├── 1.3 裸の except: の修正
└── 1.4 BMesh メモリリークの修正

Phase 2: コード品質改善（中リスク・中効果）
├── 2.1 マジックナンバーの定数化
├── 2.2 コメントアウトコードの削除
└── 2.3 論理的リージョン分割（コメント整理）

Phase 3: アーキテクチャ改善（高リスク・高効果）
├── 3.1 グローバル変数のクラス化
├── 3.2 関数の責務分離
└── 3.3 サブモジュール分割（オプション）
```

---

## 3. Phase 1: 緊急修正

### 3.1 重複関数の削除

**対象**: `calculate_weight_pattern_similarity`（line 6334, line 14175）

**手順**:

1. 両方の実装を比較して差異を確認
2. 正しい実装を特定（または統合）
3. 1つの実装を削除
4. 全ての呼び出し箇所をテスト

**実装例**:
```python
# 削除対象: line 6334 の実装
# 残す: line 14175 の実装（現在実際に使用されている）

# 確認コマンド
# grep -n "calculate_weight_pattern_similarity" retarget_script2_14.py
```

**検証方法**:
- 既存の衣装リターゲット処理が正常に動作することを確認
- ウェイト転送結果に差異がないことを確認

---

### 3.2 ハードコードオブジェクト名の修正

**対象**: lines 20344-20347

**現在のコード**:
```python
bpy.data.objects.remove(bpy.data.objects["Body.Template"], do_unlink=True)
bpy.data.objects.remove(bpy.data.objects["Body.Template.Eyes"], do_unlink=True)
bpy.data.objects.remove(bpy.data.objects["Body.Template.Head"], do_unlink=True)
bpy.data.objects.remove(bpy.data.objects["Armature.Template"], do_unlink=True)
```

**修正後のコード**:
```python
def safe_remove_object(obj_name: str) -> None:
    """オブジェクトが存在する場合のみ削除する"""
    if obj_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
        print(f"Removed object: {obj_name}")
    else:
        print(f"Object not found (skipped): {obj_name}")

# 使用例
template_objects = [
    "Body.Template",
    "Body.Template.Eyes",
    "Body.Template.Head",
    "Armature.Template"
]

for obj_name in template_objects:
    safe_remove_object(obj_name)
```

**検証方法**:
- Template オブジェクトが存在する場合のテスト
- Template オブジェクトが存在しない場合のテスト
- 部分的に存在する場合のテスト

---

### 3.3 裸の `except:` の修正

**対象**: 5箇所

| # | 行番号 | 関数 | 修正方針 |
|---|--------|------|----------|
| 1 | 1981 | `parse_args` | `except ValueError:` に変更 |
| 2 | 3202 | `triangulate_mesh` | `except Exception:` に変更 |
| 3 | 6501 | `inverse_bone_deform_all_vertices` | `except Exception:` に変更 |
| 4 | 19764 | `export_armature_bone_data_to_json` | `except Exception:` に変更 |
| 5 | 20964 | `main` | `except Exception:` に変更 |

**修正例**:

```python
# 修正前 (line 1981)
try:
    parts = args.hips_position.split(',')
    hips_pos = tuple(float(p.strip()) for p in parts)
except:
    print("Error: Invalid hips position format. Use x,y,z")
    sys.exit(1)

# 修正後
try:
    parts = args.hips_position.split(',')
    hips_pos = tuple(float(p.strip()) for p in parts)
except ValueError as e:
    print(f"Error: Invalid hips position format. Use x,y,z: {e}")
    sys.exit(1)
```

```python
# 修正前 (line 3202)
try:
    bpy.ops.object.mode_set(mode='OBJECT')
except:
    pass

# 修正後
try:
    bpy.ops.object.mode_set(mode='OBJECT')
except Exception:
    pass  # モード変更失敗は無視可能
```

**検証方法**:
- Ctrl+C でプログラムが正常に中断されることを確認
- 各例外発生時に適切なエラーメッセージが表示されることを確認

---

### 3.4 BMesh メモリリークの修正

**対象**: 3関数、5つの BMesh オブジェクト

| 関数 | BMesh 変数 | 行番号 |
|------|-----------|--------|
| `create_hinge_bone_group` | `cloth_bm` | 4800 |
| `transfer_weights_from_nearest_vertex` | `body_bm`, `cloth_bm` | 11475, 11490 |
| `transfer_weights_x_projection` | `template_bm`, `target_bm` | 11821, 11838 |

**修正パターン**:

```python
# 修正前
def create_hinge_bone_group(...):
    cloth_bm = bmesh.new()
    cloth_bm.from_mesh(cloth_obj.data)
    # ... 処理 ...
    return result  # cloth_bm.free() がない

# 修正後
def create_hinge_bone_group(...):
    cloth_bm = bmesh.new()
    try:
        cloth_bm.from_mesh(cloth_obj.data)
        # ... 処理 ...
        return result
    finally:
        cloth_bm.free()
```

**複数 BMesh の場合**:

```python
# 修正前
def transfer_weights_from_nearest_vertex(...):
    body_bm = bmesh.new()
    body_bm.from_mesh(body_mesh)
    cloth_bm = bmesh.new()
    cloth_bm.from_mesh(cloth_mesh)
    # ... 処理 ...
    return result

# 修正後
def transfer_weights_from_nearest_vertex(...):
    body_bm = bmesh.new()
    cloth_bm = bmesh.new()
    try:
        body_bm.from_mesh(body_mesh)
        cloth_bm.from_mesh(cloth_mesh)
        # ... 処理 ...
        return result
    finally:
        body_bm.free()
        cloth_bm.free()
```

**検証方法**:
- チェーン処理（複数 config）でメモリ使用量が累積しないことを確認
- 処理前後でメモリ使用量を計測

---

## 4. Phase 2: コード品質改善

### 4.1 マジックナンバーの定数化

**ファイル先頭に定数セクションを追加**:

```python
# =============================================================================
# Constants
# =============================================================================

# Thresholds
LOOSE_VERTEX_THRESHOLD = 1000       # 分離頂点の警告閾値
MIN_WEIGHT_THRESHOLD = 0.0005       # 最小ウェイト閾値
WEIGHT_SMOOTHING_FACTOR = 0.5       # ウェイトスムージング係数

# Limits
MAX_BONE_INFLUENCE = 4              # 最大ボーン影響数
DEFAULT_BATCH_SIZE = 1000           # デフォルトバッチサイズ

# Tolerances
POSITION_EPSILON = 0.0001           # 位置比較の許容誤差
ROTATION_EPSILON = 0.001            # 回転比較の許容誤差
```

**修正対象の特定**:
```bash
# マジックナンバーの検索
grep -n "if.*[0-9]\{3,\}" retarget_script2_14.py | head -20
grep -n "< 0\.[0-9]" retarget_script2_14.py | head -20
```

---

### 4.2 コメントアウトコードの削除

**手順**:

1. コメントアウトされたコードブロックを特定
2. Git 履歴で元のコードの目的を確認
3. 不要と判断したコードを削除
4. 必要な場合は TODO コメントに置換

**検索コマンド**:
```bash
# 大きなコメントブロックの検索
grep -n "^#.*def\|^#.*class\|^# *for\|^# *if" retarget_script2_14.py
```

---

### 4.3 論理的リージョン分割

**ファイル構造を明確化**:

```python
# =============================================================================
# IMPORTS
# =============================================================================
import bpy
import numpy as np
...

# =============================================================================
# CONSTANTS
# =============================================================================
LOOSE_VERTEX_THRESHOLD = 1000
...

# =============================================================================
# GLOBAL STATE (TODO: Phase 3 でクラス化)
# =============================================================================
_mesh_cache = {}
_deformation_field_cache = {}
...

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def safe_remove_object(obj_name: str) -> None:
    ...

# =============================================================================
# MESH PROCESSING
# =============================================================================
def triangulate_mesh(obj):
    ...

# =============================================================================
# WEIGHT TRANSFER
# =============================================================================
def transfer_weights_from_nearest_vertex(...):
    ...

# =============================================================================
# DEFORMATION FIELD
# =============================================================================
def get_deformation_field(...):
    ...

# =============================================================================
# MAIN PROCESSING
# =============================================================================
def process_single_config(...):
    ...

def main():
    ...

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
```

---

## 5. Phase 3: アーキテクチャ改善

### 5.1 グローバル変数のクラス化

**RetargetContext クラスの導入**:

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import bmesh

@dataclass
class RetargetContext:
    """
    リターゲット処理のコンテキスト（状態管理）

    グローバル変数を置き換え、テスト容易性を向上させる。
    """

    # Caches
    mesh_cache: Dict[str, Any] = field(default_factory=dict)
    deformation_field_cache: Dict[str, Any] = field(default_factory=dict)

    # Pose State
    saved_pose_state: Optional[Dict] = None
    previous_pose_state: Optional[Dict] = None

    # Armature Data
    armature_record_data: Dict[str, Any] = field(default_factory=dict)

    # Configuration
    unity_script_directory: Optional[str] = None
    is_A_pose: bool = False

    # Initialization Flags
    numpy_checked: bool = False
    scipy_checked: bool = False

    def clear_mesh_cache(self) -> None:
        """メッシュキャッシュをクリアし、BMesh を解放"""
        for cache_data in self.mesh_cache.values():
            if isinstance(cache_data, dict) and 'bmesh' in cache_data:
                try:
                    cache_data['bmesh'].free()
                except Exception:
                    pass
        self.mesh_cache.clear()
        print("Mesh cache cleared")

    def clear_deformation_cache(self) -> None:
        """変形フィールドキャッシュをクリア"""
        self.deformation_field_cache.clear()
        print("Deformation field cache cleared")

    def clear_all_caches(self) -> None:
        """全キャッシュをクリア"""
        self.clear_mesh_cache()
        self.clear_deformation_cache()
        self.saved_pose_state = None
        self.previous_pose_state = None
        self.armature_record_data.clear()

        # Blender のオーファンデータを削除
        import bpy
        bpy.data.orphans_purge(do_recursive=True)

        # ガベージコレクション
        import gc
        gc.collect()

        print("All caches cleared")


# グローバルインスタンス（互換性維持）
_context = RetargetContext()

# 旧グローバル変数からのマイグレーション
# TODO: 段階的に関数内のグローバル変数参照を _context に置換
_mesh_cache = _context.mesh_cache
_deformation_field_cache = _context.deformation_field_cache
```

**移行手順**:

1. `RetargetContext` クラスを追加
2. グローバルインスタンス `_context` を作成
3. 各グローバル変数を `_context` のプロパティへのエイリアスに変更
4. 関数を段階的に `ctx: RetargetContext` 引数を受け取るように変更
5. 最終的にエイリアスを削除

---

### 5.2 関数の責務分離

**大きな関数の分割例**:

```python
# 修正前: 1つの巨大関数
def process_single_config(ctx, config, input_fbx, output_fbx, ...):
    # 500行以上の処理
    ...

# 修正後: 責務ごとに分割
def process_single_config(ctx: RetargetContext, config: Dict, ...):
    """メイン処理のオーケストレーション"""

    # 1. 準備フェーズ
    prepare_processing(ctx, config)

    # 2. アバター処理
    process_avatars(ctx, config)

    # 3. 変形適用
    apply_deformations(ctx, config)

    # 4. ウェイト転送
    process_weights(ctx, config)

    # 5. エクスポート
    export_result(ctx, output_fbx)

    # 6. クリーンアップ
    ctx.clear_all_caches()

def prepare_processing(ctx: RetargetContext, config: Dict) -> None:
    """処理の準備（インポート、初期化）"""
    ...

def process_avatars(ctx: RetargetContext, config: Dict) -> None:
    """アバター処理（ベース、衣装）"""
    ...
```

---

### 5.3 サブモジュール分割（オプション）

**最終的な構造案**:

```
MochFitter-unity-addon/BlenderTools/blender-4.0.2-windows-x64/dev/
├── retarget_script2_14.py      # エントリーポイント（main のみ）
└── retarget/
    ├── __init__.py             # パッケージ初期化
    ├── context.py              # RetargetContext クラス
    ├── constants.py            # 定数定義
    ├── io/
    │   ├── __init__.py
    │   ├── fbx.py              # FBX インポート/エクスポート
    │   └── avatar_data.py      # Avatar データ I/O
    ├── deformation/
    │   ├── __init__.py
    │   ├── field.py            # Deformation Field 処理
    │   └── blendshape.py       # BlendShape 処理
    ├── weight/
    │   ├── __init__.py
    │   ├── transfer.py         # ウェイト転送
    │   └── smoothing.py        # スムージング処理
    └── mesh/
        ├── __init__.py
        └── component.py        # メッシュコンポーネント処理
```

**注意点**:
- Blender `--python` 実行時のモジュール読み込みパスの設定が必要
- 単一ファイル実行との互換性を維持する wrapper が必要

```python
# retarget_script2_14.py (エントリーポイント)
import sys
import os

# モジュールパスを追加
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from retarget import main

if __name__ == "__main__":
    main()
```

---

## 6. 実装ロードマップ

### 6.1 Phase 1（緊急修正）✅ 完了

> **ステータス**: PR #38 および PR #39 で実装完了

```
[x] 1.1 重複関数の削除
    [x] 両実装の差分確認
    [x] 1つの実装を削除（line 14175 の重複を削除）
    [x] 構文チェック（python -m py_compile）
    [ ] E2Eテスト（Blenderでの動作確認）

[x] 1.2 ハードコードオブジェクト名の修正
    [x] safe_remove_object 関数の追加
    [x] 呼び出し箇所の修正
    [x] 構文チェック
    [ ] E2Eテスト

[x] 1.3 裸の except: の修正
    [x] 5箇所すべてを修正
    [x] except Exception: または except ValueError: に変更

[x] 1.4 BMesh メモリリークの修正
    [x] 3関数の .free() 追加（create_hinge_bone_group は try/finally 済み）
    [x] 2関数の try/finally 追加（transfer_weights_* - PR #39）
    [ ] チェーン処理でのメモリテスト
```

### 6.2 Phase 2（コード品質改善）✅ 完了

```
[x] 2.1 マジックナンバーの定数化 (PR #38)
    [x] 定数セクションの追加（Constants セクション）
    [x] 主要なマジックナンバーの置換

[x] 2.2 コメントアウトコードの削除 (PR #40)
    [x] 不要コードの特定（225行）
    [x] 削除

[x] 2.3 論理的リージョン分割 (PR #40)
    [x] Global State セクションの追加
    [x] リージョンコメントの追加（Weight Transfer, Main Processing, Entry Point）
    [ ] 関数の並べ替え（Phase 3で対応）
```

### 6.3 Phase 3（アーキテクチャ改善）

```
[x] 3.1 グローバル変数のクラス化 (PR #43)
    [x] RetargetContext クラスの追加
    [x] エイリアスによる互換性維持
    [x] clear_all_caches() の _context への委譲
    [ ] 段階的な関数の移行（今後のPRで対応）

[x] 3.2 関数の責務分離 (PR #45)
    [x] print_config_details() - 設定ファイル内容出力
    [x] clean_mesh_invalid_vertices() - 独立・非有限頂点削除
    [x] apply_sub_bone_overrides() / restore_bone_overrides() - サブボーン上書き
    [x] process_mesh_in_cycle1() - Cycle1メッシュ処理
    [x] preprocess_for_export() - FBXエクスポート前処理
    [x] process_single_config をオーケストレーション関数に変換

[ ] 3.3 サブモジュール分割（オプション）
    [ ] ディレクトリ構造の作成
    [ ] モジュールの分割
    [ ] エントリーポイントの更新
```

---

## 7. テスト計画

### 7.1 各フェーズのテスト

| フェーズ | テスト内容 |
|---------|-----------|
| Phase 1 | 既存の Unity → Blender パイプラインが動作すること |
| Phase 2 | リグレッションテスト（出力が変わらないこと） |
| Phase 3 | ユニットテストの追加、統合テスト |

### 7.2 テストケース

```python
# テストシナリオ
test_scenarios = [
    {
        "name": "単一 config 処理",
        "input": "clothing.fbx",
        "configs": ["config_a.json"],
        "expected": "正常終了"
    },
    {
        "name": "チェーン処理（2段階）",
        "input": "clothing.fbx",
        "configs": ["config_a.json", "config_b.json"],
        "expected": "正常終了、メモリリークなし"
    },
    {
        "name": "存在しないオブジェクト",
        "input": "clothing_no_template.fbx",
        "configs": ["config_a.json"],
        "expected": "警告出力、クラッシュなし"
    },
    {
        "name": "Ctrl+C による中断",
        "input": "clothing.fbx",
        "configs": ["config_a.json"],
        "action": "処理中に Ctrl+C",
        "expected": "正常終了"
    }
]
```

---

## 8. リスク管理

### 8.1 リスク一覧

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|---------|------|
| 修正によるリグレッション | 高 | 中 | 各フェーズ後にテスト実行 |
| Unity 側との互換性問題 | 高 | 低 | API（引数、出力形式）を変更しない |
| Phase 3 での大規模変更失敗 | 中 | 中 | Phase 1, 2 を先に完了させる |
| テスト不足による問題見逃し | 中 | 中 | テストケースを事前に定義 |

### 8.2 ロールバック計画

- 各フェーズは別ブランチで作業
- 問題発生時は前のバージョンにロールバック可能
- Git タグでマイルストーンを記録

```bash
# フェーズ完了時のタグ付け
git tag -a v0.2.18-phase1 -m "Phase 1: 緊急修正完了"
git tag -a v0.2.18-phase2 -m "Phase 2: コード品質改善完了"
```

---

## 9. 成功基準

| フェーズ | 成功基準 |
|---------|---------|
| Phase 1 | 全ての重大問題（🔴）が解消されている |
| Phase 2 | コードの可読性が向上し、新規開発者が理解しやすくなっている |
| Phase 3 | ユニットテストのカバレッジが 50% 以上になっている |

---

## 更新履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-12-29 | 1.0 | 初版作成 |
| 2025-12-29 | 1.1 | Phase 1 完了、Phase 2.1 完了を反映（PR #38） |
| 2025-12-30 | 1.2 | Phase 3.1 完了（RetargetContext クラス追加） |
| 2025-12-30 | 1.3 | Phase 3.2 完了（関数の責務分離） |
