#!/usr/bin/env python3
"""
FBX ファイルの Armature 位置と Hips ボーン位置を比較するスクリプト

使用方法:
    blender --background --python compare_fbx_hips.py -- <fbx1> <fbx2>

例:
    blender --background --python compare_fbx_hips.py -- \
        "path/to/before.fbx" \
        "path/to/after.fbx"
"""

import bpy
import sys
from mathutils import Vector


def clear_scene():
    """シーンをクリア"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def import_fbx(filepath):
    """FBX をインポートして Armature を返す"""
    # インポート前の Armature 名を記録
    existing_armatures = {obj.name for obj in bpy.data.objects if obj.type == 'ARMATURE'}

    bpy.ops.import_scene.fbx(filepath=filepath)

    # インポート後に追加された Armature を探す（より堅牢な方法）
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj.name not in existing_armatures:
            return obj

    # フォールバック: 選択状態から探す
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def get_hips_world_position(armature_obj):
    """Hips ボーンのワールド位置を取得"""
    if not armature_obj:
        return None

    # Hips ボーンを探す（様々な名前パターンに対応）
    hips_names = ['Hips', 'hips', 'J_Bip_C_Hips', 'mixamorig:Hips', 'Bip001 Pelvis']

    for bone_name in hips_names:
        if bone_name in armature_obj.pose.bones:
            pose_bone = armature_obj.pose.bones[bone_name]
            return armature_obj.matrix_world @ pose_bone.head

    # 見つからない場合は最初のボーンを使用
    if armature_obj.pose.bones:
        first_bone = armature_obj.pose.bones[0]
        print(f"  Warning: Hips not found, using first bone: {first_bone.name}")
        return armature_obj.matrix_world @ first_bone.head

    return None


def get_mesh_bounding_box_center():
    """シーン内のすべてのメッシュのバウンディングボックス中心を取得"""
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

    mesh_count = 0
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh_count += 1
            # ワールド座標でのバウンディングボックスを計算
            for vert in obj.data.vertices:
                world_coord = obj.matrix_world @ vert.co
                min_coords.x = min(min_coords.x, world_coord.x)
                min_coords.y = min(min_coords.y, world_coord.y)
                min_coords.z = min(min_coords.z, world_coord.z)
                max_coords.x = max(max_coords.x, world_coord.x)
                max_coords.y = max(max_coords.y, world_coord.y)
                max_coords.z = max(max_coords.z, world_coord.z)

    if mesh_count == 0:
        return None

    center = (min_coords + max_coords) / 2
    return center, min_coords, max_coords


def get_first_mesh_vertex_average():
    """最初のメッシュの頂点平均位置を取得"""
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and len(obj.data.vertices) > 0:
            total = Vector((0, 0, 0))
            for vert in obj.data.vertices:
                world_coord = obj.matrix_world @ vert.co
                total += world_coord
            avg = total / len(obj.data.vertices)
            return avg, obj.name, len(obj.data.vertices)
    return None, None, 0


def analyze_fbx(filepath, label):
    """FBX を分析して情報を返す"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {label}")
    print(f"File: {filepath}")
    print('='*60)

    clear_scene()
    armature = import_fbx(filepath)

    if not armature:
        print("  ERROR: No armature found!")
        return None

    # メッシュ情報を取得
    bbox_result = get_mesh_bounding_box_center()
    vertex_avg, mesh_name, vertex_count = get_first_mesh_vertex_average()

    result = {
        'label': label,
        'filepath': filepath,
        'armature_name': armature.name,
        'armature_location': armature.location.copy(),
        'hips_world_position': get_hips_world_position(armature),
        'mesh_bbox_center': bbox_result[0] if bbox_result else None,
        'mesh_bbox_min': bbox_result[1] if bbox_result else None,
        'mesh_bbox_max': bbox_result[2] if bbox_result else None,
        'vertex_average': vertex_avg,
        'mesh_name': mesh_name,
        'vertex_count': vertex_count,
    }

    print(f"  Armature name: {result['armature_name']}")
    print(f"  Armature location: {result['armature_location']}")
    print(f"  Hips world position: {result['hips_world_position']}")
    print(f"  Mesh bbox center: {result['mesh_bbox_center']}")
    print(f"  Mesh bbox min: {result['mesh_bbox_min']}")
    print(f"  Mesh bbox max: {result['mesh_bbox_max']}")
    print(f"  Vertex average ({mesh_name}, {vertex_count} verts): {vertex_avg}")

    return result


def compare_results(result1, result2):
    """2つの結果を比較"""
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print('='*60)

    if not result1 or not result2:
        print("ERROR: Cannot compare - missing data")
        return

    # Armature location の差
    loc_diff = result2['armature_location'] - result1['armature_location']
    print(f"\nArmature Location Difference:")
    print(f"  X: {loc_diff.x:+.6f} m ({loc_diff.x * 100:+.2f} cm)")
    print(f"  Y: {loc_diff.y:+.6f} m ({loc_diff.y * 100:+.2f} cm)")
    print(f"  Z: {loc_diff.z:+.6f} m ({loc_diff.z * 100:+.2f} cm)")
    print(f"  Total: {loc_diff.length:.6f} m ({loc_diff.length * 100:.2f} cm)")

    # Hips world position の差
    if result1['hips_world_position'] and result2['hips_world_position']:
        hips_diff = result2['hips_world_position'] - result1['hips_world_position']
        print(f"\nHips World Position Difference:")
        print(f"  X: {hips_diff.x:+.6f} m ({hips_diff.x * 100:+.2f} cm)")
        print(f"  Y: {hips_diff.y:+.6f} m ({hips_diff.y * 100:+.2f} cm)")
        print(f"  Z: {hips_diff.z:+.6f} m ({hips_diff.z * 100:+.2f} cm)")
        print(f"  Total: {hips_diff.length:.6f} m ({hips_diff.length * 100:.2f} cm)")

    # Mesh Bounding Box Center の差（重要！）
    if result1['mesh_bbox_center'] and result2['mesh_bbox_center']:
        bbox_diff = result2['mesh_bbox_center'] - result1['mesh_bbox_center']
        print(f"\n*** MESH Bounding Box Center Difference (KEY METRIC): ***")
        print(f"  X: {bbox_diff.x:+.6f} m ({bbox_diff.x * 100:+.2f} cm)")
        print(f"  Y: {bbox_diff.y:+.6f} m ({bbox_diff.y * 100:+.2f} cm)")
        print(f"  Z: {bbox_diff.z:+.6f} m ({bbox_diff.z * 100:+.2f} cm)")
        print(f"  Total: {bbox_diff.length:.6f} m ({bbox_diff.length * 100:.2f} cm)")

    # Vertex Average の差
    if result1['vertex_average'] and result2['vertex_average']:
        vert_diff = result2['vertex_average'] - result1['vertex_average']
        print(f"\nVertex Average Position Difference:")
        print(f"  X: {vert_diff.x:+.6f} m ({vert_diff.x * 100:+.2f} cm)")
        print(f"  Y: {vert_diff.y:+.6f} m ({vert_diff.y * 100:+.2f} cm)")
        print(f"  Z: {vert_diff.z:+.6f} m ({vert_diff.z * 100:+.2f} cm)")
        print(f"  Total: {vert_diff.length:.6f} m ({vert_diff.length * 100:.2f} cm)")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"  Before ({result1['label']}): BBox Center at {result1['mesh_bbox_center']}")
    print(f"  After  ({result2['label']}): BBox Center at {result2['mesh_bbox_center']}")
    if result1['mesh_bbox_center'] and result2['mesh_bbox_center']:
        bbox_diff = result2['mesh_bbox_center'] - result1['mesh_bbox_center']
        if bbox_diff.length > 0.001:
            print(f"  -> MESH POSITION CHANGED BY {bbox_diff.length * 100:.2f} cm")
        else:
            print(f"  -> Mesh positions are essentially identical")


def main():
    # コマンドライン引数を取得
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("Usage: blender --background --python compare_fbx_hips.py -- <fbx1> <fbx2>")
        sys.exit(1)

    if len(argv) < 2:
        print("ERROR: Please provide two FBX file paths")
        sys.exit(1)

    fbx1_path = argv[0]
    fbx2_path = argv[1]

    # FBX を分析
    result1 = analyze_fbx(fbx1_path, "Before (12/14)")
    result2 = analyze_fbx(fbx2_path, "After (12/15)")

    # 比較
    compare_results(result1, result2)


if __name__ == "__main__":
    main()
