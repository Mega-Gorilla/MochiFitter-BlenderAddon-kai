"""
FBXファイルのモディファイア状態を確認するスクリプト
apply_modifiers_keep_shapekeys_with_temp() のスキップ可能性を評価
"""

import bpy
import sys
import json
import os

def check_modifiers(fbx_path: str) -> dict:
    """FBXファイルを読み込み、各メッシュのモディファイア情報を取得"""

    # シーンをクリア
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # FBXをインポート
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    result = {
        "fbx_file": os.path.basename(fbx_path),
        "meshes": {}
    }

    # 全メッシュオブジェクトのモディファイアをチェック
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        mesh_info = {
            "num_vertices": len(obj.data.vertices),
            "num_shape_keys": 0,
            "shape_key_names": [],
            "modifiers": [],
            "armature_modifiers": [],
            "non_armature_modifiers": []
        }

        # シェイプキー情報
        if obj.data.shape_keys:
            mesh_info["num_shape_keys"] = len(obj.data.shape_keys.key_blocks)
            mesh_info["shape_key_names"] = [sk.name for sk in obj.data.shape_keys.key_blocks]

        # モディファイア情報
        for mod in obj.modifiers:
            mod_info = {
                "name": mod.name,
                "type": mod.type
            }
            mesh_info["modifiers"].append(mod_info)

            if mod.type == 'ARMATURE':
                mesh_info["armature_modifiers"].append(mod_info)
            else:
                mesh_info["non_armature_modifiers"].append(mod_info)

        result["meshes"][obj.name] = mesh_info

    return result

def main():
    # コマンドライン引数を取得
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    if len(argv) < 1:
        print("Usage: blender --background --python check_modifiers.py -- <fbx_path>")
        sys.exit(1)

    fbx_path = argv[0]

    print(f"\n{'='*60}")
    print(f"Checking modifiers in: {fbx_path}")
    print(f"{'='*60}\n")

    result = check_modifiers(fbx_path)

    # サマリーを表示
    total_meshes = len(result["meshes"])
    total_shape_keys = sum(m["num_shape_keys"] for m in result["meshes"].values())
    total_armature_mods = sum(len(m["armature_modifiers"]) for m in result["meshes"].values())
    total_non_armature_mods = sum(len(m["non_armature_modifiers"]) for m in result["meshes"].values())

    print(f"Summary:")
    print(f"  Total meshes: {total_meshes}")
    print(f"  Total shape keys: {total_shape_keys}")
    print(f"  Total Armature modifiers: {total_armature_mods}")
    print(f"  Total non-Armature modifiers: {total_non_armature_mods}")
    print()

    # 詳細を表示
    print("Details per mesh:")
    for mesh_name, info in result["meshes"].items():
        print(f"\n  {mesh_name}:")
        print(f"    Vertices: {info['num_vertices']}")
        print(f"    Shape keys: {info['num_shape_keys']}")
        print(f"    Armature modifiers: {len(info['armature_modifiers'])}")
        print(f"    Non-Armature modifiers: {len(info['non_armature_modifiers'])}")

        if info["non_armature_modifiers"]:
            print(f"    Non-Armature modifier details:")
            for mod in info["non_armature_modifiers"]:
                print(f"      - {mod['name']} ({mod['type']})")

    # スキップ可能性の評価
    print(f"\n{'='*60}")
    print("Optimization Assessment:")
    print(f"{'='*60}")

    if total_non_armature_mods == 0:
        print("✓ No non-Armature modifiers found!")
        print("→ apply_modifiers_keep_shapekeys_with_temp() can be SKIPPED")
        print(f"→ Potential savings: ~{total_shape_keys} object duplications per mesh pair")
    else:
        print("✗ Non-Armature modifiers exist")
        print("→ apply_modifiers_keep_shapekeys_with_temp() is REQUIRED")
        print("→ Consider: check modifiers at runtime and skip if none")

if __name__ == "__main__":
    main()
