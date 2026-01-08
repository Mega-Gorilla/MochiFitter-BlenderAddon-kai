import bpy
import numpy as np
from mathutils import Vector, Matrix, Euler
import os
import sys
import subprocess
import platform
import math
import bmesh
from mathutils.bvhtree import BVHTree
import time
from math import ceil, sqrt
import json
import traceback
import shutil
from typing import Dict, Optional, Tuple, Set
from bpy_extras.io_utils import ExportHelper

# scipyのインポートを条件付きで行う
try:
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist, pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not found. Some features will be limited.")
    print("Please use the NumPy/SciPy reinstall button to install.")

print(f"SciPy available: {SCIPY_AVAILABLE}")

def get_scene_folder():
    """
    現在のBlenderシーンファイルのフォルダパスを取得する
    未保存の場合はカレントディレクトリを使用
    
    Returns:
        str: フォルダパス
    """
    blend_filepath = bpy.data.filepath
    if blend_filepath:
        return os.path.dirname(blend_filepath)
    else:
        print("Warning: Blend file is not saved. Using current directory.")
        return os.getcwd()

def load_avatar_data(filename="avatar_data.json"):
    """
    アバターデータを読み込む
    
    Parameters:
        filename (str): アバターデータのJSONファイル名
    
    Returns:
        dict: アバターデータの辞書
    """
    scene_folder = get_scene_folder()
    filepath = os.path.join(scene_folder, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Avatar data file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        avatar_data = json.load(f)
    
    return avatar_data

def normalize_avatar_name_for_filename(name: str) -> str:
    """
    アバター名をファイル名用に正規化（小文字変換）

    Unity拡張側との互換性のため、出力ファイル名を小文字に統一する。
    Linuxでは大文字小文字が区別されるため、統一しないと別ファイルとして扱われる。

    Parameters:
        name (str): アバター名

    Returns:
        str: 小文字に変換されたアバター名

    See: GitHub Issue #64
    """
    return name.lower() if name else ""

def find_field_data_file(scene_folder: str, source_avatar_name: str, target_avatar_name: str = None,
                         source_shape_key_name: str = None, inverse_suffix: str = "") -> Optional[str]:
    """
    変形フィールドデータファイルのパスを探索する（後方互換性対応）

    新しい小文字ファイル名を優先し、存在しない場合は元の大文字混在ファイル名にフォールバック。
    これにより、旧バージョンで作成されたファイルも引き続き使用可能。

    Parameters:
        scene_folder (str): 検索対象のフォルダパス
        source_avatar_name (str): ソースアバター名
        target_avatar_name (str, optional): ターゲットアバター名（アバター間変形の場合）
        source_shape_key_name (str, optional): シェイプキー名（シェイプキーモードの場合）
        inverse_suffix (str): 逆変換サフィックス（"_inv" or ""）

    Returns:
        Optional[str]: 見つかったファイルパス、見つからない場合はNone

    See: GitHub Issue #64, PR #66 review feedback
    """
    # 新しい小文字ファイル名（優先）
    if source_shape_key_name:
        new_filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_shape_{source_shape_key_name}{inverse_suffix}.npz"
        old_filename = f"deformation_{source_avatar_name}_shape_{source_shape_key_name}{inverse_suffix}.npz"
    else:
        new_filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}{inverse_suffix}.npz"
        old_filename = f"deformation_{source_avatar_name}_to_{target_avatar_name}{inverse_suffix}.npz"

    new_path = os.path.join(scene_folder, new_filename)
    old_path = os.path.join(scene_folder, old_filename)

    # 小文字ファイルを優先して探索
    if os.path.exists(new_path):
        return new_path
    # 旧来の大文字混在ファイルにフォールバック
    if os.path.exists(old_path):
        print(f"Note: Using legacy filename '{old_filename}' (consider renaming to '{new_filename}')")
        return old_path

    return None

def build_bone_hierarchy(bone_node: dict, bone_parents: Dict[str, str], current_path: list):
    """
    ボーン階層から親子関係のマッピングを再帰的に構築する

    Parameters:
        bone_node (dict): 現在のボーンノード
        bone_parents (Dict[str, str]): ボーン名から親ボーン名へのマッピング
        current_path (list): 現在のパス上のボーン名のリスト
    """
    bone_name = bone_node['name']
    if current_path:
        bone_parents[bone_name] = current_path[-1]
    
    current_path.append(bone_name)
    for child in bone_node.get('children', []):
        build_bone_hierarchy(child, bone_parents, current_path)
    current_path.pop()

def get_humanoid_bone_hierarchy(avatar_data: dict) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    アバターデータからHumanoidボーンの階層関係を抽出する

    Parameters:
        avatar_data (dict): アバターデータ

    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]: 
            (ボーン名から親への辞書, Humanoidボーン名からボーン名への辞書, ボーン名からHumanoidボーン名への辞書)
    """
    # ボーンの親子関係を構築
    bone_parents = {}
    build_bone_hierarchy(avatar_data['boneHierarchy'], bone_parents, [])

    # Humanoidボーン名とボーン名の対応マップを作成
    humanoid_to_bone = {bone_map['humanoidBoneName']: bone_map['boneName'] 
                       for bone_map in avatar_data['humanoidBones']}
    bone_to_humanoid = {bone_map['boneName']: bone_map['humanoidBoneName'] 
                       for bone_map in avatar_data['humanoidBones']}
    
    return bone_parents, humanoid_to_bone, bone_to_humanoid

def find_nearest_parent_with_pose(bone_name: str, 
                                bone_parents: Dict[str, str], 
                                bone_to_humanoid: Dict[str, str],
                                pose_data: dict) -> Optional[str]:
    """
    指定されたボーンの親を辿り、ポーズデータを持つ最も近い親のHumanoidボーン名を返す

    Parameters:
        bone_name (str): 開始ボーン名
        bone_parents (Dict[str, str]): ボーンの親子関係辞書
        bone_to_humanoid (Dict[str, str]): ボーン名からHumanoidボーン名への変換辞書
        pose_data (dict): ポーズデータ

    Returns:
        Optional[str]: 見つかった親のHumanoidボーン名、見つからない場合はNone
    """
    current_bone = bone_name
    while current_bone in bone_parents:
        parent_bone = bone_parents[current_bone]
        if parent_bone in bone_to_humanoid:
            parent_humanoid = bone_to_humanoid[parent_bone]
            if parent_humanoid in pose_data:
                return parent_humanoid
        current_bone = parent_bone
    return None

def save_armature_pose(armature_obj, filename="pose_data.json", avatar_data_file="avatar_data.json"):
    """
    アクティブなArmatureのHumanoidボーンのポーズをワールド座標系でJSONファイルに保存する

    Parameters:
        filename (str): 保存するJSONファイルの名前
        avatar_data_file (str): アバターデータのJSONファイル名
    """
    if not armature_obj:
        raise ValueError("No armature object found")
    
    if armature_obj.type != 'ARMATURE':
        raise ValueError(f"Active object '{armature_obj.name}' is not an armature")
    
    # アバターデータを読み込む
    avatar_data = load_avatar_data(avatar_data_file)
    
    # ボーン名からHumanoidボーン名へのマッピングを作成
    _, _, bone_to_humanoid = get_humanoid_bone_hierarchy(avatar_data)
    
    # 保存先のフルパスを作成
    scene_folder = get_scene_folder()
    filepath = os.path.join(scene_folder, filename)
    
    # ポーズボーンの情報を格納する辞書
    pose_data = {}
    
    for bone in armature_obj.pose.bones:
        # Humanoidボーンでない場合はスキップ
        if bone.name not in bone_to_humanoid:
            continue
            
        humanoid_name = bone_to_humanoid[bone.name]
        base_matrix = armature_obj.data.bones[bone.name].matrix_local
        
        # ワールド空間での行列を計算
        world_matrix = armature_obj.matrix_world @ bone.matrix
        base_world_matrix = armature_obj.matrix_world @ base_matrix
        
        delta_matrix = world_matrix @ base_world_matrix.inverted()
        
        # ボーンのHeadのワールド座標を計算
        head_local = armature_obj.data.bones[bone.name].head_local
        head_world = armature_obj.matrix_world @ head_local
        head_world_transformed = armature_obj.matrix_world @ bone.head
        
        # 位置を取得
        location = head_world_transformed - head_world
        
        # 回転を取得（オイラー角に変換）
        rotation = delta_matrix.to_euler('XYZ')
        
        # スケールを取得
        scale = delta_matrix.to_scale()

        # データを辞書に格納（Humanoidボーン名をキーとして使用）
        # delta_matrix: Unity側スクリプトとの互換性のため必須
        # location/rotation/scale: 参考値（デバッグ用）
        pose_data[humanoid_name] = {
            'delta_matrix': matrix_to_list(delta_matrix),
            'location': [location.x, location.y, location.z],
            'rotation': [math.degrees(rotation.x),
                        math.degrees(rotation.y),
                        math.degrees(rotation.z)],
            'scale': [scale.x, scale.y, scale.z],
            'head_world': [head_world.x, head_world.y, head_world.z],
            'head_world_transformed': [head_world_transformed.x, head_world_transformed.y, head_world_transformed.z]
        }
    
    # JSONファイルに保存
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(pose_data, f, indent=4)
        
    print(f"Pose data saved for humanoid bones to {filepath}")
    return filepath

def clear_humanoid_bone_relations_preserve_pose(armature_obj, avatar_data_file="avatar_data.json"):
    """
    Humanoidボーンの親子関係を解除しながらワールド空間でのポーズを保持する
    
    Args:
        armature_obj: bpy.types.Object - アーマチュアオブジェクト
        avatar_data_file (str): アバターデータのJSONファイル名
    """
    if armature_obj.type != 'ARMATURE':
        raise ValueError("Selected object must be an armature")
    
    # アバターデータを読み込む
    avatar_data = load_avatar_data(avatar_data_file)
    
    # Humanoidボーンのリストを作成
    humanoid_bones = {bone_map['boneName'] for bone_map in avatar_data['humanoidBones']}
    
    # Get the armature data
    armature = armature_obj.data
    
    # Store original world space matrices for humanoid bones
    original_matrices = {}
    for bone in armature.bones:
        if bone.name in humanoid_bones:
            pose_bone = armature_obj.pose.bones[bone.name]
            original_matrices[bone.name] = armature_obj.matrix_world @ pose_bone.matrix
    
    # Switch to edit mode to modify bone relations
    bpy.context.view_layer.objects.active = armature_obj
    original_mode = bpy.context.object.mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Clear parent relationships for humanoid bones only
    for edit_bone in armature.edit_bones:
        if edit_bone.name in humanoid_bones:
            edit_bone.parent = None
    
    # Return to pose mode
    bpy.ops.object.mode_set(mode='POSE')
    
    # Restore original world space positions for humanoid bones
    for bone_name, original_matrix in original_matrices.items():
        pose_bone = armature_obj.pose.bones[bone_name]
        pose_bone.matrix = armature_obj.matrix_world.inverted() @ original_matrix
    
    # Return to original mode
    bpy.ops.object.mode_set(mode=original_mode)

def is_finger_bone(humanoid_bone: str) -> bool:
    """
    指のボーンかどうかを判定する
    
    Parameters:
        humanoid_bone (str): Humanoidボーン名
        
    Returns:
        bool: 指のボーンの場合True
    """
    finger_keywords = [
        "Thumb", "Index", "Middle", "Ring", "Little",
        "Toe"
    ]
    return any(keyword in humanoid_bone for keyword in finger_keywords)

def get_next_joint_bone(humanoid_bone: str) -> Optional[str]:
    """
    指の次の関節のボーン名を取得する
    
    Parameters:
        humanoid_bone (str): Humanoidボーン名
        
    Returns:
        Optional[str]: 次の関節のボーン名、存在しない場合None
    """
    joint_mapping = {
        "Proximal": "Intermediate",
        "Intermediate": "Distal",
    }
    
    # 現在の関節タイプを特定
    current_joint = None
    for joint_type in joint_mapping.keys():
        if joint_type in humanoid_bone:
            current_joint = joint_type
            break
            
    if not current_joint:
        return None
        
    # 次の関節のボーン名を生成
    next_joint = joint_mapping[current_joint]
    return humanoid_bone.replace(current_joint, next_joint)

def apply_finger_bone_adjustments(
    armature_obj: bpy.types.Object,
    humanoid_to_bone: Dict[str, str],
    bone_to_humanoid: Dict[str, str]
) -> None:
    """
    指のボーンの位置を調整する
    各ボーンのTailが次の関節のHeadと一致するように調整
    
    Parameters:
        armature_obj: アーマチュアオブジェクト
        humanoid_to_bone: Humanoidボーン名からボーン名への変換辞書
        bone_to_humanoid: ボーン名からHumanoidボーン名への変換辞書
    """
    # すべての指ボーンについて処理
    for bone_name, pose_bone in armature_obj.pose.bones.items():
        if bone_name not in bone_to_humanoid:
            continue
            
        humanoid_bone = bone_to_humanoid[bone_name]
        if not is_finger_bone(humanoid_bone):
            continue
            
        # 次の関節を取得
        next_humanoid_bone = get_next_joint_bone(humanoid_bone)
        if not next_humanoid_bone or next_humanoid_bone not in humanoid_to_bone:
            continue
            
        next_bone_name = humanoid_to_bone[next_humanoid_bone]
        if next_bone_name not in armature_obj.pose.bones:
            continue
            
        next_bone = armature_obj.pose.bones[next_bone_name]
        
        # 現在のボーンの方向ベクトルを取得
        current_dir = ((armature_obj.matrix_world @ pose_bone.tail) - (armature_obj.matrix_world @ pose_bone.head)).normalized()
        
        # 世界空間での位置を計算
        head_world = armature_obj.matrix_world @ pose_bone.head
        next_head_world = armature_obj.matrix_world @ next_bone.head
        
        # 新しい方向ベクトルを計算
        new_dir = (next_head_world - head_world).normalized()
        
        # 回転の差分を計算
        #rot_diff = new_dir.rotation_difference(current_dir)
        rot_diff = current_dir.rotation_difference(new_dir)
        
        # 現在の行列を取得
        current_matrix = pose_bone.matrix.copy()
        
        translation, rotation, scale = current_matrix.decompose()
        trans_mat = Matrix.Translation(translation)

        # 回転を適用した新しい行列を作成
        rot_matrix = rot_diff.to_matrix().to_4x4()
        new_matrix = trans_mat @ rot_matrix @ trans_mat.inverted() @ current_matrix
        
        print(f"{bone_name} {next_bone_name} \n {head_world} \n {next_head_world} \n {rot_diff.to_euler('XYZ')}")
        
        # 新しい行列を適用
        pose_bone.matrix = new_matrix

def matrix_to_list(matrix):
    """
    Matrix型からリストに変換する（JSON保存用）

    Parameters:
        matrix: Matrix - Blenderの行列オブジェクト

    Returns:
        list: 行列を2次元リストに変換したもの
    """
    return [list(row) for row in matrix]

def list_to_matrix(matrix_list):
    """
    リストからMatrix型に変換する（JSON読み込み用）

    Parameters:
        matrix_list: list - 行列のデータを含む2次元リスト

    Returns:
        Matrix: 変換された行列
    """
    return Matrix(matrix_list)

def add_pose_from_json(filename="pose_data.json", avatar_data_file="avatar_data.json", invert=False):
    """
    JSONファイルから読み込んだポーズデータをアクティブなArmatureの現在のポーズに加算する
    
    Parameters:
        filename (str): 読み込むJSONファイルの名前
        avatar_data_file (str): アバターデータのJSONファイル名
        invert (bool): 逆変換を適用するかどうか
    """
    # アクティブオブジェクトを取得
    active_obj = bpy.context.active_object
    
    if not active_obj:
        raise ValueError("No active object found")
    
    if active_obj.type != 'ARMATURE':
        raise ValueError(f"Active object '{active_obj.name}' is not an armature")
    
    # アバターデータを読み込む
    avatar_data = load_avatar_data(avatar_data_file)
    
    # 階層関係と変換マップを取得
    bone_parents, humanoid_to_bone, bone_to_humanoid = get_humanoid_bone_hierarchy(avatar_data)
    
    # ファイルの完全パスを取得
    scene_folder = get_scene_folder()
    filepath = os.path.join(scene_folder, filename)
    
    # ファイルの存在確認
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pose data file not found: {filepath}")
    
    # JSONファイルを読み込む
    with open(filepath, 'r', encoding='utf-8') as f:
        pose_data = json.load(f)
    
    # アンドゥ用にステップを作成
    bpy.ops.ed.undo_push(message="Add Pose from JSON")

    # エディットモードに切り替え
    bpy.ops.object.mode_set(mode='EDIT')
    
    # すべての編集ボーンのConnectedを解除
    for bone in active_obj.data.edit_bones:
        bone.use_connect = False
    
    # オブジェクトモードに戻る
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 親子関係を維持したまま処理するため、階層順序でボーンを取得
    def get_bone_hierarchy_order():
        """親から子への順序でHumanoidボーンを取得"""
        order = []
        visited = set()
        
        def add_bone_and_children(humanoid_bone):
            if humanoid_bone in visited:
                return
            visited.add(humanoid_bone)
            order.append(humanoid_bone)
            
            # 子ボーンを検索
            for child_bone, parent_bone in bone_parents.items():
                if parent_bone == humanoid_bone and child_bone not in visited:
                    add_bone_and_children(child_bone)
        
        # ルートボーン（Hips）から開始
        root_bones = []
        root_bones.append(humanoid_to_bone['Hips'])
        
        for root_bone in root_bones:
            add_bone_and_children(root_bone)
        
        return order
    
    bone_order = get_bone_hierarchy_order()
    
    # 処理済みのHumanoidボーンを記録する辞書
    processed_bones = {}
    
    # 事前にすべてのボーンの変形前の状態を保存
    original_bone_data = {}
    for humanoid_bone in humanoid_to_bone.keys():
        bone_name = humanoid_to_bone.get(humanoid_bone)
        if bone_name and bone_name in active_obj.pose.bones:
            bone = active_obj.pose.bones[bone_name]
            original_bone_data[humanoid_bone] = {
                'matrix': bone.matrix.copy(),
                'head': bone.head.copy(),
                'tail': bone.tail.copy(),
                'bone_name': bone_name
            }
    
    # 階層順序でポーズデータの計算を実行
    for bone_name in bone_order:
        if not bone_name or bone_name not in active_obj.pose.bones:
            continue
   
        humanoid_bone = bone_to_humanoid.get(bone_name)
        if not humanoid_bone:
            continue
        
        # 既に処理済みの場合はスキップ
        if humanoid_bone in processed_bones:
            continue

        # ポーズデータを直接持っているか、親から継承するかを決定
        source_humanoid_bone = humanoid_bone
        if humanoid_bone not in pose_data:
            parent_with_pose = find_nearest_parent_with_pose(
                bone_name, bone_parents, bone_to_humanoid, pose_data)
            if not parent_with_pose:
                continue
            source_humanoid_bone = parent_with_pose
            print(f"Using pose data from parent bone {source_humanoid_bone} for {humanoid_bone}")
        
        # 保存されたオリジナルデータを使用して計算
        if humanoid_bone not in original_bone_data:
            continue
            
        bone = active_obj.pose.bones[bone_name]
        
        original_data = original_bone_data[humanoid_bone]
        
        # 現在のワールド空間での行列を取得（オリジナルデータを使用）
        current_world_matrix = active_obj.matrix_world @ original_data['matrix']

        # 変換行列を構築
        bone_pose = pose_data[source_humanoid_bone]

        # delta_matrixが存在する場合は優先的に使用（旧形式JSONとの互換性）
        # delta_matrixがない場合のみlocation/rotation/scaleから再構築（新形式JSON）
        if 'delta_matrix' in bone_pose:
            # 旧形式: delta_matrixを直接使用（最も正確）
            delta_matrix = list_to_matrix(bone_pose['delta_matrix'])
        elif 'location' in bone_pose and 'rotation' in bone_pose and 'scale' in bone_pose:
            # 新形式: location/rotation/scaleから行列を再構築
            # rotation値は度で保存されているのでラジアンに変換
            delta_loc = Vector(bone_pose['location'])
            delta_rot = Euler([math.radians(x) for x in bone_pose['rotation']], 'XYZ')
            delta_scale = Vector(bone_pose['scale'])

            delta_matrix = Matrix.Translation(delta_loc) @ \
                        delta_rot.to_matrix().to_4x4() @ \
                        Matrix.Scale(delta_scale.x, 4, (1, 0, 0)) @ \
                        Matrix.Scale(delta_scale.y, 4, (0, 1, 0)) @ \
                        Matrix.Scale(delta_scale.z, 4, (0, 0, 1))
        else:
            print(f"Warning: No valid pose data for {source_humanoid_bone}, skipping")
            continue
        
        if invert:
            delta_matrix = delta_matrix.inverted()
            
        # 現在の行列に加算
        combined_matrix = delta_matrix @ current_world_matrix
        
        # ローカル空間に変換して適用
        bone.matrix = active_obj.matrix_world.inverted() @ combined_matrix
        
        print(bone_name)
        print(bone.matrix)
        
        # 変更を即座に反映（子ボーンの計算に影響するため）
        bpy.context.view_layer.update()
        
        # 処理済みとしてマーク
        processed_bones[humanoid_bone] = True
    
    # 最終的なポーズの更新を強制
    bpy.context.view_layer.update()
    print(f"Pose data added to armature '{active_obj.name}' from {filepath}")
    
     # Invert時の指ボーンの調整
    if invert:
        #apply_finger_bone_adjustments(active_obj, humanoid_to_bone, bone_to_humanoid)
        # 最終的なポーズの更新を強制
        bpy.context.view_layer.update()
    
    for bone_name in active_obj.pose.bones.keys():
        if bone_name in bone_to_humanoid:
            humanoid_name = bone_to_humanoid[bone_name]
            if humanoid_name in processed_bones:
                mat = active_obj.pose.bones[bone_name].matrix
                print(f"'{humanoid_name}' ({bone_name}) bone.matrix_final {mat}")

def get_vertices_in_scaled_bbox(source_obj, scale_factor=1.2):
    """
    選択された頂点から計算されるBounding Boxをスケールし、
    そのBounding Box内に含まれる全ての頂点のインデックスを取得する
    
    Parameters:
    source_obj: ソースオブジェクト
    scale_factor: スケール倍率
    
    Returns:
    list: Bounding Box内に含まれる頂点のインデックスのリスト
    """
    # 選択された頂点のBounding Boxを計算
    bounds_min, bounds_max = calculate_target_bounding_box(source_obj, scale_factor, use_selected_vertices=True)
    
    # 評価されたメッシュを取得
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = source_obj.evaluated_get(depsgraph)
    
    # 全ての頂点をワールド座標に変換してBounding Box内かチェック
    matrix_world = source_obj.matrix_world
    vertices_in_bbox = []
    
    for i, vertex in enumerate(eval_obj.data.vertices):
        world_pos = matrix_world @ Vector(vertex.co)
        
        # Bounding Box内かチェック
        if (bounds_min.x <= world_pos.x <= bounds_max.x and
            bounds_min.y <= world_pos.y <= bounds_max.y and
            bounds_min.z <= world_pos.z <= bounds_max.z):
            vertices_in_bbox.append(i)
    
    print(f"Number of vertices in scaled Bounding Box: {len(vertices_in_bbox)}")
    return vertices_in_bbox


def calculate_target_bounding_box(target_obj, scale_factor=1.2, use_selected_vertices=False):
    """
    ターゲットメッシュのBounding Boxを計算し、スケールして正方形にする
    
    Parameters:
    target_obj: ターゲットメッシュオブジェクト
    scale_factor: スケール倍率（デフォルト1.2倍）
    use_selected_vertices: Trueの場合、選択された頂点のみを使用
    
    Returns:
    bounds_min, bounds_max: 正方形のBounding Boxの最小・最大座標
    """
    # 編集モードかどうかを確認し、選択頂点情報を取得
    vertices_world = []
    matrix_world = target_obj.matrix_world
    
    if use_selected_vertices:
        # 現在のモードを保存
        current_mode = bpy.context.object.mode if bpy.context.object else 'OBJECT'
        was_in_edit_mode = current_mode == 'EDIT'
        
        try:
            # 編集モードでない場合は編集モードに切り替え
            if not was_in_edit_mode:
                bpy.context.view_layer.objects.active = target_obj
                bpy.ops.object.mode_set(mode='EDIT')
            
            # bmeshを使用して選択された頂点を取得
            bm = bmesh.from_edit_mesh(target_obj.data)
            
            # 選択された頂点のみを取得
            selected_vertices = [v for v in bm.verts if v.select]
            
            if not selected_vertices:
                print("Warning: No vertices selected. Using all vertices.")
                # 選択された頂点がない場合は全ての頂点を使用
                vertices_world = [matrix_world @ Vector(v.co) for v in bm.verts]
            else:
                vertices_world = [matrix_world @ Vector(v.co) for v in selected_vertices]
                print(f"Number of selected vertices: {len(selected_vertices)}")
            
            # bmeshの更新（必須ではないが推奨）
            bmesh.update_edit_mesh(target_obj.data)
            
        finally:
            # 元のモードに戻す
            if not was_in_edit_mode and current_mode == 'OBJECT':
                bpy.context.view_layer.objects.active = target_obj
                bpy.ops.object.mode_set(mode='OBJECT')
    
    else:
        # 全ての頂点を使用
        depsgraph = bpy.context.evaluated_depsgraph_get()
        target_eval = target_obj.evaluated_get(depsgraph)
        vertices_world = [matrix_world @ Vector(v.co) for v in target_eval.data.vertices]
        print(f"Using all vertices: {len(vertices_world)}")
    
    if not vertices_world:
        raise ValueError("ターゲットメッシュに有効な頂点がありません")
    
    # numpy配列に変換して一括処理
    vertices_array = np.array([[v.x, v.y, v.z] for v in vertices_world])
    bounds_min_orig = Vector(vertices_array.min(axis=0))
    bounds_max_orig = Vector(vertices_array.max(axis=0))
    
    # 元のBounding Boxの中心と寸法を計算
    center = (bounds_min_orig + bounds_max_orig) * 0.5
    dimensions = bounds_max_orig - bounds_min_orig
    
    # 最も長い辺の長さを取得
    max_dimension = max(dimensions.x, dimensions.y, dimensions.z)
    
    # スケールを適用
    scaled_half_size = (max_dimension * scale_factor) * 0.5
    
    # 正方形のBounding Boxを生成（X軸対称を考慮）
    x_extent = max(abs(center.x - scaled_half_size), abs(center.x + scaled_half_size))
    
    bounds_min = Vector((
        -x_extent,
        center.y - scaled_half_size,
        center.z - scaled_half_size
    ))
    bounds_max = Vector((
        x_extent,
        center.y + scaled_half_size,
        center.z + scaled_half_size
    ))
    
    vertex_type = "selected vertices" if use_selected_vertices else "all vertices"
    print(f"Vertices used: {vertex_type} ({len(vertices_world)})")
    print(f"Original dimensions of target mesh: {dimensions}")
    print(f"Maximum dimension: {max_dimension:.4f}")
    print(f"Scaled square size: {max_dimension * scale_factor:.4f}")
    print(f"Generated Bounding Box: Min{bounds_min}, Max{bounds_max}")
    
    return bounds_min, bounds_max


def create_adaptive_deformation_field(target_obj, base_grid_spacing=0.005, surface_distance=2.1, max_distance=2.1, min_distance=0.0036, density_falloff=3.0, bbox_scale_factor=1.2, use_selected_vertices=False):
    """
    ターゲットメッシュから自動生成されたBounding Boxを使用して、
    距離に応じて密度が変化するDeformation Fieldを生成する
    
    Parameters:
    target_obj: Surface Deformのターゲットメッシュ
    base_grid_spacing: 基本グリッドの間隔（メートル単位）
    surface_distance: ターゲットメッシュ表面からの最大距離
    max_distance: 最大ウェイト距離
    min_distance: 最小ウェイト距離
    density_falloff: 密度の減衰率（大きいほど段階的な密度の変化が急速に起こる）
    bbox_scale_factor: Bounding Boxのスケール倍率
    use_selected_vertices: Trueの場合、選択された頂点のみでBounding Boxを計算
    """
    start_time = time.time()
    
    # ターゲットメッシュから自動的にBounding Boxを計算
    bounds_min, bounds_max = calculate_target_bounding_box(target_obj, bbox_scale_factor, use_selected_vertices)
    
    # 各軸の長さを計算
    dimensions = bounds_max - bounds_min
    
    # ターゲットメッシュのBVHツリーを作成
    depsgraph = bpy.context.evaluated_depsgraph_get()
    target_eval = target_obj.evaluated_get(depsgraph)
    target_mesh = target_eval.data
    
    bm = bmesh.new()
    bm.from_mesh(target_mesh)
    bm.transform(target_obj.matrix_world)
    
    # "ignore"頂点グループをチェックし、ウェイト0.5以上の頂点を含む面を除外
    ignore_group = None
    for vg in target_obj.vertex_groups:
        if vg.name == "ignore":
            ignore_group = vg
            break
    
    if ignore_group:
        print(f"Found 'ignore' vertex group. Filtering faces...")
        # ウェイト0.5以上の頂点を特定
        ignore_vertices = set()
        for vert in target_mesh.vertices:
            for group in vert.groups:
                if group.group == ignore_group.index and group.weight >= 0.5:
                    ignore_vertices.add(vert.index)
        
        # 除外対象の頂点を含む面を削除
        faces_to_remove = []
        for face in bm.faces:
            for vert in face.verts:
                if vert.index in ignore_vertices:
                    faces_to_remove.append(face)
                    break
        
        for face in faces_to_remove:
            bm.faces.remove(face)
        
        print(f"Removed {len(faces_to_remove)} faces containing ignore vertices (total ignore vertices: {len(ignore_vertices)})")
    
    bvh = BVHTree.FromBMesh(bm)
    
    # グリッドポイントの生成
    vertices = []
    
    # 事前計算とキャッシュ
    inv_max_min_diff = 1.0 / (max_distance - min_distance)
    
    # 適応的なグリッド生成のためのヘルパー関数（最適化）
    def get_adaptive_spacing(distance):
        if distance <= min_distance:
            return 0
        elif distance > surface_distance:
            return float('inf')  # 範囲外のポイントは生成しない
        else:
            # 正規化した距離を計算（0～1の間の値）
            normalized_distance = (distance - min_distance) * inv_max_min_diff
            normalized_distance = min(1.0, max(0.0, normalized_distance))
            
            # 距離に応じて2のべき乗で間隔を増加させる
            power = sqrt(normalized_distance) * density_falloff
            level = int(power + 1)  # 整数部分を取得して段階化
            
            # 2^levelの値を計算（ビットシフトで最適化）
            return 1 << level
    
    # X軸の対称性を考慮したグリッド生成
    steps_x_positive = int(ceil(bounds_max.x / base_grid_spacing)) + 1
    steps_y = int(ceil(dimensions.y / base_grid_spacing)) + 1
    steps_z = int(ceil(dimensions.z / base_grid_spacing)) + 1
    
    # 進捗表示用の変数
    total_points = steps_x_positive * steps_y * steps_z
    processed_points = 0
    last_update = time.time()
    update_interval = 2.0  # 2秒ごとに進捗を更新
    
    # バッチ処理用のバッファ
    batch_size = 1000
    batch_positions = []
    batch_mirror_positions = []
    
    # 処理済みの適応的間隔をキャッシュ
    spacing_cache = {}
    
    # 段階的グリッド走査のための関数
    def process_cell(x_start, y_start, z_start, cell_size, level=0, max_level=3):
        nonlocal batch_positions, batch_mirror_positions, processed_points, last_update
        
        # セルの8つの頂点の座標を計算
        cell_vertices = []
        min_distance_in_cell = float('inf')
        
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    x_pos = x_start + dx * cell_size * base_grid_spacing
                    y_pos = y_start + dy * cell_size * base_grid_spacing
                    z_pos = z_start + dz * cell_size * base_grid_spacing
                    
                    vertex_pos = Vector((x_pos, y_pos, z_pos))
                    location, normal, index, distance = bvh.find_nearest(vertex_pos)
                    
                    if location and distance <= surface_distance:
                        cell_vertices.append((vertex_pos, distance))
                        min_distance_in_cell = min(min_distance_in_cell, distance)
        
        # セル内に有効な頂点がない場合は処理しない
        if not cell_vertices:
            return
        
        # セル内の最小距離に基づく適応的間隔を取得
        if min_distance_in_cell in spacing_cache:
            min_adaptive_spacing = spacing_cache[min_distance_in_cell]
        else:
            min_adaptive_spacing = get_adaptive_spacing(min_distance_in_cell)
            spacing_cache[min_distance_in_cell] = min_adaptive_spacing
        
        # セルサイズが最小適応的間隔より大きく、最大レベルに達していない場合は分割
        if cell_size > min_adaptive_spacing and level < max_level:
            half_size = cell_size // 2
            if half_size > 0:
                # セルを8つのサブセルに分割
                for dx in [0, 1]:
                    for dy in [0, 1]:
                        for dz in [0, 1]:
                            new_x = x_start + dx * half_size * base_grid_spacing
                            new_y = y_start + dy * half_size * base_grid_spacing
                            new_z = z_start + dz * half_size * base_grid_spacing
                            process_cell(new_x, new_y, new_z, half_size, level + 1, max_level)
        else:
            # セルの中心点を追加
            x_center = x_start + (cell_size * base_grid_spacing) / 2
            y_center = y_start + (cell_size * base_grid_spacing) / 2
            z_center = z_start + (cell_size * base_grid_spacing) / 2
            
            center_pos = Vector((x_center, y_center, z_center))
            location, normal, index, distance = bvh.find_nearest(center_pos)
            
            if location and distance <= surface_distance:
                # バッチに追加
                batch_positions.append(center_pos)
                batch_mirror_positions.append(Vector((-x_center, y_center, z_center)))
                
                # バッチサイズに達したら処理
                if len(batch_positions) >= batch_size:
                    process_batch(batch_positions, batch_mirror_positions, bvh, vertices)
                    batch_positions = []
                    batch_mirror_positions = []
            
            processed_points += 1
            current_time = time.time()
            if current_time - last_update >= update_interval:
                print(f"Processing: {processed_points} points")
                last_update = current_time
    
    # 初期の粗いグリッドを生成し、段階的に細分化
    initial_cell_size = 2 ** int(density_falloff+1)  # 初期セルサイズ（2のべき乗が効率的）
    
    # X>0の領域のグリッドポイントを生成
    for z in range(0, steps_z, initial_cell_size):
        z_pos = bounds_min.z + z * base_grid_spacing
        
        for y in range(0, steps_y, initial_cell_size):
            y_pos = bounds_min.y + y * base_grid_spacing
            
            for x in range(0, steps_x_positive, initial_cell_size):
                x_pos = x * base_grid_spacing  # 正のX軸方向
                
                # セルを処理
                process_cell(x_pos, y_pos, z_pos, initial_cell_size, 0, int(density_falloff+1))
    
    # 残りのバッチを処理
    if batch_positions:
        process_batch(batch_positions, batch_mirror_positions, bvh, vertices)
    
    if not vertices:
        bm.free()
        print("Warning: No valid points found in specified range")
        return None

    # クリーンアップ
    bm.free()

    end_time = time.time()
    print(f"Number of generated points: {len(vertices)}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    return vertices


def process_batch(positions, mirror_positions, bvh, vertices):
    """バッチでグリッドポイントを処理する"""
    for pos, mirror_pos in zip(positions, mirror_positions):
        # 正のX側のポイント
        location, normal, index, distance = bvh.find_nearest(pos)
        if location:
            vertices.append(pos)
        
        # 負のX側のポイント
        location, normal, index, distance = bvh.find_nearest(mirror_pos)
        if location:
            vertices.append(mirror_pos)


def compute_distances_to_source_mesh(target_vertices, source_obj):
    """
    ターゲットメッシュの各頂点からソースメッシュの最近接面までの距離を計算
    BVHTreeを使用して高速に距離を計算
    
    Parameters:
    - target_vertices: ターゲットメッシュの頂点座標（ワールド座標）
    - source_obj: ソースメッシュオブジェクト
    
    Returns:
    - 各ターゲット頂点からソースメッシュまでの距離の配列
    """
    num_vertices = len(target_vertices)
    distances = np.zeros(num_vertices)
    
    print("Building BVH tree for source mesh...")
    
    # ソースメッシュからBVHツリーを構築
    bm_source = bmesh.new()
    bm_source.from_mesh(source_obj.data)
    bm_source.faces.ensure_lookup_table()
    
    # ソースメッシュをワールド座標に変換
    for v in bm_source.verts:
        v.co = source_obj.matrix_world @ v.co
    
    # BVHツリーを構築
    source_bvh = BVHTree.FromBMesh(bm_source)
    
    print("Calculating distance to nearest face for each vertex...")
    for i, vertex in enumerate(target_vertices):
        # BVHツリーを使用して最近接点と距離を計算
        closest_point, closest_normal, closest_face_idx, distance = source_bvh.find_nearest(vertex)
        
        if closest_point is not None:
            distances[i] = distance
        else:
            # 最近接点が見つからない場合は大きな値を設定
            distances[i] = 9999.0
    
    # bmeshを解放
    bm_source.free()
    
    print("Distance calculation complete")
    return distances


def smooth_step(x, edge0, edge1):
    """
    Performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
    
    Parameters:
    x: The input value to interpolate
    edge0: The lower edge of the interpolation range
    edge1: The upper edge of the interpolation range
    
    Returns:
    A value between 0 and 1, with smooth transitions at the edges
    """
    # Clamp x to the range [0, 1]
    x = np.maximum(0, np.minimum(1, (x - edge0) / (edge1 - edge0)))
    
    # Apply the smooth step formula: 3x^2 - 2x^3
    return x * x * (3 - 2 * x)


def create_partial_mesh_from_vertices(source_obj, vertex_indices):
    """
    指定された頂点インデックスから部分メッシュオブジェクトを作成する
    
    Parameters:
    - source_obj: ソースメッシュオブジェクト
    - vertex_indices: 含める頂点のインデックスリスト
    
    Returns:
    - 部分メッシュオブジェクト（一時的）
    """
    # bmeshを使用して部分メッシュを作成
    bm = bmesh.new()
    
    # ソースメッシュを読み込み
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = source_obj.evaluated_get(depsgraph)
    bm.from_mesh(eval_obj.data)
    
    # ワールド変換を適用
    bm.transform(source_obj.matrix_world)
    
    # 選択された頂点以外を削除
    vertex_indices_set = set(vertex_indices)
    verts_to_remove = [v for i, v in enumerate(bm.verts) if i not in vertex_indices_set]
    
    for v in verts_to_remove:
        bm.verts.remove(v)
    
    # 面を再計算
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    # 孤立した頂点や辺を削除
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    
    # 新しいメッシュを作成
    partial_mesh = bpy.data.meshes.new(name="PartialMesh_Temp")
    bm.to_mesh(partial_mesh)
    bm.free()
    
    # 新しいオブジェクトを作成（ワールド座標系に既に変換済みなので、単位行列を使用）
    partial_obj = bpy.data.objects.new("PartialMesh_Temp", partial_mesh)
    partial_obj.matrix_world = Matrix.Identity(4)
    
    # シーンに追加（BVHTree作成のため必要）
    bpy.context.scene.collection.objects.link(partial_obj)
    
    return partial_obj


def add_normal_control_points_func(source_obj, control_indices, control_positions_original, control_positions_deformed, normal_distance):
    """
    制御点の法線方向に追加の制御点を生成する
    
    Parameters:
    - source_obj: ソースメッシュオブジェクト
    - control_indices: 制御点として使用されている頂点のインデックス
    - control_positions_original: 元の制御点位置（ワールド座標）
    - control_positions_deformed: 変形後の制御点位置（ワールド座標）
    - normal_distance: 法線方向への距離（ワールド座標系）
    
    Returns:
    - 拡張された制御点の元の位置配列
    - 拡張された制御点の変形後位置配列
    """
    # ソースオブジェクトのワールド行列を取得
    source_world_matrix = source_obj.matrix_world
    
    # 評価されたメッシュを取得
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = source_obj.evaluated_get(depsgraph)
    
    # 拡張された制御点配列を初期化（元の制御点 + 法線方向の制御点）
    extended_original = []
    extended_deformed = []
    
    # 元の制御点を追加
    extended_original.extend(control_positions_original)
    extended_deformed.extend(control_positions_deformed)
    
    # 各制御点について法線方向に制御点を追加
    for i, vertex_index in enumerate(control_indices):
        # 頂点の法線を取得（ローカル座標）
        vertex_normal_local = eval_obj.data.vertices[vertex_index].normal.copy()
        
        # 法線をワールド座標に変換（回転のみ、位置は影響しない）
        normal_world = source_world_matrix.to_3x3() @ vertex_normal_local
        normal_world.normalize()
        
        # 元の制御点位置
        original_pos = Vector(control_positions_original[i])
        deformed_pos = Vector(control_positions_deformed[i])
        
        # 法線方向のオフセット（指定された距離、正負により方向が決まる）
        normal_offset = normal_world * normal_distance
        
        # 法線方向の制御点位置を計算
        normal_original = original_pos + normal_offset
        normal_deformed = deformed_pos + normal_offset
        
        extended_original.append(normal_original)
        extended_deformed.append(normal_deformed)
    
    # NumPy配列に変換
    extended_original = np.array([[p[0], p[1], p[2]] for p in extended_original])
    extended_deformed = np.array([[p[0], p[1], p[2]] for p in extended_deformed])
    
    direction_text = "inward" if normal_distance < 0 else "outward"
    print(f"Extended control points: {len(control_indices)} -> {len(extended_original)} (added control points {abs(normal_distance):.5f}m {direction_text} along normals)")
    
    return extended_original, extended_deformed


def falloff_displacements(target_vertices, target_displacements, source_obj):
    """
    距離に基づいて変位にフォールオフを適用
    """
    num_vertices = len(target_vertices)
    
    # 各頂点のソースメッシュの最近接面までの距離を計算
    print("Calculating distance to source mesh...")
    distances = compute_distances_to_source_mesh(target_vertices, source_obj)
    
    # 距離に基づく重み付け
    distances = np.maximum(distances - 0.015, 0.0)
    weights = np.minimum(1.0, smooth_step(distances * 4.0, 0.0, 1.0))

    final_displacements = []
    
    for i in range(num_vertices):
        if weights[i] > 0:
            # 距離に応じた重み付けを適用
            blend_factor = weights[i]
            next_displacement = (1.0 - blend_factor) * target_displacements[i]
        else:
            next_displacement = target_displacements[i]
        
        final_displacements.append(next_displacement)
    
    return final_displacements


def multi_quadratic_biharmonic(r, epsilon=1.0):
    """Multi-Quadratic Biharmonic RBFカーネル関数"""
    return np.sqrt(r**2 + epsilon**2)


def rbf_interpolation(source_control_points, source_control_points_deformed, target_vertices, source_obj, epsilon=1.0, batch_size=100000, falloff_source_obj=None):
    """
    RBFを使用してターゲットメッシュの新しい位置を計算（バッチ処理版）
    
    Parameters:
    - source_control_points: ソースメッシュの選択された制御点（基準位置）- ワールド座標
    - source_control_points_deformed: シェイプキーで変形後のソースメッシュの制御点 - ワールド座標
    - target_vertices: 変形するターゲットメッシュの頂点座標（ワールド座標）
    - source_obj: ソースメッシュオブジェクト（最近接面の距離計算に使用）
    - epsilon: RBFパラメータ
    - batch_size: 一度に処理するターゲット頂点の数
    - falloff_source_obj: フォールオフ計算用のソースオブジェクト（Noneの場合はsource_objを使用）
    
    Returns:
    - 変形後のターゲットメッシュの頂点位置（ローカル座標）
    - ターゲットメッシュの世界座標
    - 変位ベクトル
    """
    # 変位ベクトルを計算（変形後の位置 - 元の位置）
    displacements = source_control_points_deformed - source_control_points
    
    # SciPyの利用可能性をチェック
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPyが利用できません。NumPy・SciPy再インストールボタンを使用してインストールしてください。")
    
    # スケーリング係数を計算：距離の標準偏差に基づく値を使用
    if epsilon <= 0:
        # 平均距離に基づいて適切なepsilonを計算
        dists = cdist(source_control_points, source_control_points)
        mean_dist = np.mean(dists[dists > 0])
        epsilon = mean_dist  # 平均距離をepsilonとして使用
        print(f"Auto-calculated epsilon: {epsilon}")
    
    # 制御点間の距離行列を計算
    dist_matrix = cdist(source_control_points, source_control_points)
    
    # RBF行列を計算
    phi = multi_quadratic_biharmonic(dist_matrix, epsilon)
    
    num_pts, dim = source_control_points.shape
    P = np.ones((num_pts, dim + 1))
    P[:, 1:] = source_control_points  # 多項式項のための拡張行列
    
    # 完全な線形システムを構築
    A = np.zeros((num_pts + dim + 1, num_pts + dim + 1))
    A[:num_pts, :num_pts] = phi
    A[:num_pts, num_pts:] = P
    A[num_pts:, :num_pts] = P.T
    
    # 右辺を設定
    b = np.zeros((num_pts + dim + 1, dim))
    b[:num_pts] = displacements
    
    # 解を求める
    try:
        # 通常の解法を試みる
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 行列が特異な場合、正則化して疑似逆行列を使用
        print("Matrix is singular - applying regularization")
        reg = np.eye(A.shape[0]) * 1e-6
        x = np.linalg.lstsq(A + reg, b, rcond=None)[0]
    
    # 重みを抽出
    rbf_weights = x[:num_pts]
    poly_weights = x[num_pts:]
    
    # ターゲット頂点のローカル座標を取得
    total_vertices = len(target_vertices)
    
    # 結果を格納する配列を初期化
    target_deformed = np.zeros_like(target_vertices)
    target_world_vertices = np.zeros_like(target_vertices)
    target_displacements = np.zeros_like(target_vertices)
    
    # バッチごとに処理
    print(f"Processing target mesh vertices in batches of {batch_size} (total {total_vertices} vertices)")
    
    # 進捗表示用のカウンター
    processed_count = 0
    
    # バッチごとに処理
    for batch_start in range(0, total_vertices, batch_size):
        batch_end = min(batch_start + batch_size, total_vertices)
        current_batch_size = batch_end - batch_start
        
        print(f"Processing batch: {batch_start} to {batch_end-1} ({current_batch_size} vertices)")
        
        # 現在のバッチの座標
        batch_world_vertices = target_vertices[batch_start:batch_end]
        
        # ターゲット頂点と制御点の間の距離を計算
        batch_dists = cdist(batch_world_vertices, source_control_points)
        batch_phi = multi_quadratic_biharmonic(batch_dists, epsilon)
        
        # 多項式項の計算
        batch_P = np.ones((current_batch_size, dim + 1))
        batch_P[:, 1:] = batch_world_vertices
        
        # 各ターゲット頂点の変位を計算
        batch_displacements = np.dot(batch_phi, rbf_weights) + np.dot(batch_P, poly_weights)
        
        # フォールオフ処理を適用（一度に大量のメモリを消費するため、バッチ処理が有効）
        falloff_obj = falloff_source_obj if falloff_source_obj is not None else source_obj
        batch_final_displacements = falloff_displacements(
            batch_world_vertices, 
            batch_displacements, 
            falloff_obj
        )
        
        # 変位をターゲット頂点に適用（ワールド座標）
        batch_deformed_world = batch_world_vertices + batch_final_displacements
        
        for i in range(current_batch_size):
            target_deformed[batch_start + i] = batch_deformed_world[i]
            target_world_vertices[batch_start + i] = batch_world_vertices[i]
            target_displacements[batch_start + i] = batch_final_displacements[i]
        
        # 進捗を更新
        processed_count += current_batch_size
        progress_percent = (processed_count / total_vertices) * 100
        print(f"Progress: {processed_count}/{total_vertices} vertices processed ({progress_percent:.1f}%)")

    print("All batch processing complete")
    return target_deformed, target_world_vertices, target_displacements


def ensure_objects_visible(objects_to_check):
    """
    指定されたオブジェクトが非表示の場合は表示状態にし、元の状態を記録する
    
    Parameters:
        objects_to_check: チェックするオブジェクトのリスト
    
    Returns:
        dict: 元の表示状態を記録した辞書
    """
    original_states = {}
    
    for obj in objects_to_check:
        if obj is None:
            continue
        
        # 元の状態を記録
        original_states[obj.name] = {
            'hide_viewport': obj.hide_viewport,
            'hide_render': obj.hide_render,
            'hide_select': obj.hide_select
        }
        
        # 非表示の場合は表示状態にする
        if obj.hide_viewport:
            print(f"Made object '{obj.name}' visible")
            obj.hide_viewport = False
        
        if obj.hide_render:
            obj.hide_render = False
        
        if obj.hide_select:
            obj.hide_select = False
    
    bpy.context.view_layer.update()

    return original_states


def restore_objects_visibility(objects_to_restore, original_states):
    """
    オブジェクトの表示状態を元に戻す
    
    Parameters:
        objects_to_restore: 復元するオブジェクトのリスト
        original_states: 元の表示状態の辞書
    """
    for obj in objects_to_restore:
        if obj is None or obj.name not in original_states:
            continue
        
        state = original_states[obj.name]
        obj.hide_viewport = state['hide_viewport']
        obj.hide_render = state['hide_render']
        obj.hide_select = state['hide_select']
        
        if state['hide_viewport']:
            print(f"Restored visibility state of object '{obj.name}'")
    
    bpy.context.view_layer.update()

def remove_overlapping_vertices(vertices, tolerance=1e-6):
    """
    重なっている頂点を除外する
    
    Parameters:
    - vertices: 頂点座標の配列 (n, 3)
    - tolerance: 重複と見なす距離の閾値
    
    Returns:
    - unique_indices: 重複していない頂点のインデックス
    - duplicate_mask: 重複している頂点のマスク（True=重複）
    """
    if len(vertices) <= 1:
        return np.arange(len(vertices)), np.zeros(len(vertices), dtype=bool)
    
    from scipy.spatial import cKDTree
    kdtree = cKDTree(vertices)
    
    # 各頂点について、近傍の重複を検出
    pairs = kdtree.query_pairs(r=tolerance)
    
    # 重複している頂点のセットを作成
    duplicate_indices = set()
    for i, j in pairs:
        # より小さいインデックスを保持し、大きいインデックスを重複として扱う
        duplicate_indices.add(i)
        duplicate_indices.add(j)
    
    # 重複マスクを作成
    duplicate_mask = np.zeros(len(vertices), dtype=bool)
    duplicate_mask[list(duplicate_indices)] = True
    
    # 重複のない頂点のインデックスを取得
    unique_indices = np.where(~duplicate_mask)[0]
    
    print(f"Total control points: {len(vertices)}, after deduplication: {len(unique_indices)}, removed duplicates: {len(duplicate_indices)}")
    
    return unique_indices, duplicate_mask


def identify_overlapping_control_points_for_shape_keys(
    source_obj, 
    source_shape_key_name, 
    selected_indices, 
    source_world_matrix, 
    add_normal_control_points=False, 
    normal_distance=-0.0002, 
    tolerance=1e-6
):
    """
    シェイプキー値を0および1に設定した際の制御点位置をチェックし、
    いずれかで重複する制御点のインデックスを特定する
    
    Parameters:
    - source_obj: ソースオブジェクト
    - source_shape_key_name: シェイプキー名
    - selected_indices: 制御点として使用する頂点インデックス
    - source_world_matrix: ソースオブジェクトのワールド行列
    - add_normal_control_points: 法線方向制御点を追加するか
    - normal_distance: 法線方向距離
    - tolerance: 重複判定の閾値
    
    Returns:
    - overlapping_indices: 除外すべき制御点のインデックス（extended配列での位置）
    """
    
    # 元のシェイプキー値を保存
    original_shape_key_value = source_obj.data.shape_keys.key_blocks[source_shape_key_name].value
    
    overlapping_indices_set = set()
    
    try:
        # シェイプキー値0と1での制御点位置をチェック
        for shape_key_value in [0.0, 1.0]:
            print(f"Checking for duplicates at shape key value {shape_key_value}")
            
            # シェイプキー値を設定
            source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = shape_key_value
            bpy.context.view_layer.update()
            
            # 評価後のオブジェクトを取得
            depsgraph = bpy.context.evaluated_depsgraph_get()
            depsgraph.update()
            evaluated_source = source_obj.evaluated_get(depsgraph)
            
            # 制御点位置を取得（ローカル座標）
            control_points_local = np.array([evaluated_source.data.vertices[i].co.copy() for i in selected_indices])
            
            # ワールド座標に変換
            control_points_world = np.zeros_like(control_points_local)
            for i, local_co in enumerate(control_points_local):
                local_v = Vector((local_co[0], local_co[1], local_co[2], 1.0))
                world_v = source_world_matrix @ local_v
                control_points_world[i] = np.array([world_v[0], world_v[1], world_v[2]])
            
            # 法線方向制御点を追加する場合
            if add_normal_control_points:
                control_points_extended, _ = add_normal_control_points_func(
                    source_obj, 
                    selected_indices, 
                    control_points_world, 
                    control_points_world,  # 変形前後が同じ
                    normal_distance
                )
            else:
                control_points_extended = control_points_world
            
            # 重複する制御点を特定
            _, duplicate_mask = remove_overlapping_vertices(control_points_extended, tolerance)
            
            # 重複するインデックスを集合に追加
            duplicate_indices = np.where(duplicate_mask)[0]
            overlapping_indices_set.update(duplicate_indices)
            
            print(f"Detected {len(duplicate_indices)} duplicate control points at shape key value {shape_key_value}")
    
    finally:
        # 元のシェイプキー値に戻す
        source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = original_shape_key_value
        bpy.context.view_layer.update()
    
    overlapping_indices = np.array(sorted(list(overlapping_indices_set)))
    print(f"Total duplicate control points (to be excluded from all steps): {len(overlapping_indices)}")
    
    return overlapping_indices

def create_shape_key_from_rbf(source_obj, source_shape_key_name, selected_only=True, epsilon=0.0, num_steps=1, source_avatar_name="", target_avatar_name="", save_shape_key_mode=False, keep_first_field=False, add_normal_control_points=False, normal_distance=-0.0002, shape_key_start_value=0.0, shape_key_end_value=1.0):
    """
    ソースオブジェクトのシェイプキーに基づいてRBF補間を実行し、
    各ステップごとにフィールドを生成してDeformation Fieldデータを保存
    
    Parameters:
    - source_obj: ソースオブジェクト（シェイプキーを持つ）
    - source_shape_key_name: 使用するソースオブジェクトのシェイプキー名
    - selected_only: 選択された頂点のみを制御点として使用するか
    - epsilon: RBFパラメータ（0または負の値の場合は自動計算）
    - num_steps: 分割するステップ数
    - source_avatar_name: 変換元のアバター名
    - target_avatar_name: 変換先のアバター名
    - save_shape_key_mode: シェイプキー変形モード（通常と逆の両方向を保存）
    - keep_first_field: デバッグ用に最初の変形フィールドを残す
    - add_normal_control_points: 制御点の法線方向に追加制御点を配置するか
    - normal_distance: 法線方向への距離（ワールド座標系）
    - shape_key_start_value: シェイプキーの開始値
    - shape_key_end_value: シェイプキーの終了値
    """
    
    # 対象オブジェクトの表示状態を確認し、必要に応じて表示状態にする
    armature_obj = get_armature_from_source_object(source_obj)
    objects_to_check = [source_obj]
    if armature_obj:
        objects_to_check.append(armature_obj)
    
    original_visibility_states = ensure_objects_visible(objects_to_check)
    
    try:
        results = []
        
        # 保存する方向のリスト（save_shape_key_modeに基づいて決定）
        if save_shape_key_mode:
            directions = [False, True]  # 通常の変形と逆変形の両方
        else:
            directions = [False]  # 通常の変形のみ
        
        for invert in directions:
            direction_suffix = "_inv" if invert else ""
            print(f"\n=== Starting {'inverse' if invert else 'normal'} deformation processing ===")
            
            # フィールドデータの保存パスを自動生成
            scene_folder = get_scene_folder()
            
            if save_shape_key_mode:
                # シェイプキー変形モードの場合
                field_data_path = os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_shape_{source_shape_key_name}{direction_suffix}.npz")
            else:
                # 通常のアバター間変形の場合
                field_data_path = os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}{direction_suffix}.npz")
            
            # シェイプキーの値を保存
            original_values = {}
            for key in source_obj.data.shape_keys.key_blocks:
                original_values[key.name] = key.value
                key.value = 0.0
            
            # Invertオプションに応じて初期状態を設定
            if invert:
                # シェイプキーの終了値を基準にする
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = shape_key_end_value
            else:
                # シェイプキーの開始値を基準にする
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = shape_key_start_value
            
            # シーンを更新
            bpy.context.view_layer.update()
            
            # 評価後のデプスグラフを取得
            depsgraph = bpy.context.evaluated_depsgraph_get()
            
            # ソースオブジェクトの評価後のオブジェクトを取得
            evaluated_source = source_obj.evaluated_get(depsgraph)
            
            # ソースオブジェクトのシェイプキーを取得
            if source_obj.data.shape_keys is None or source_shape_key_name not in source_obj.data.shape_keys.key_blocks:
                raise ValueError(f"シェイプキー '{source_shape_key_name}' がソースオブジェクトに見つかりません")
            
            # ソースオブジェクトのワールド行列を取得
            source_world_matrix = source_obj.matrix_world
            
            # 制御点として使用する頂点のインデックスを取得
            original_selected_vertices = []  # 元の選択頂点を記録
            if selected_only:
                # 編集モードでの選択を反映するために、オブジェクトモードに切り替える
                was_edit_mode = False
                if bpy.context.object == source_obj and bpy.context.object.mode == 'EDIT':
                    was_edit_mode = True
                    bpy.ops.object.mode_set(mode='OBJECT')
                
                # 選択された頂点があるかチェック
                original_selected_vertices = [i for i, v in enumerate(source_obj.data.vertices) if v.select]
                
                # 編集モードに戻す
                if was_edit_mode:
                    bpy.ops.object.mode_set(mode='EDIT')
                
                if len(original_selected_vertices) == 0:
                    raise ValueError("選択された頂点がありません。少なくとも1つの頂点を選択してください。")
                
                # 選択された頂点から計算されるスケールされたBounding Box内の全ての頂点を制御点として使用
                selected_indices = get_vertices_in_scaled_bbox(source_obj, bpy.context.scene.rbf_bbox_scale_factor)
                
                if len(selected_indices) < 4:
                    print(f"Warning: Very few control points ({len(selected_indices)}). Consider selecting more control points.")
            else:
                # すべての頂点を使用
                selected_indices = list(range(len(source_obj.data.vertices)))
            
            # シェイプキー値0と1での重複制御点を事前に特定
            print("Pre-checking for duplicate control points at shape key values 0 and 1...")
            overlapping_indices = identify_overlapping_control_points_for_shape_keys(
                source_obj, 
                source_shape_key_name, 
                selected_indices, 
                source_world_matrix, 
                add_normal_control_points, 
                normal_distance
            )
            
            # 各ステップでの変形を計算
            all_displacements = []
            all_target_world_vertices = []
            
            for step in range(num_steps):
                print(f"\n=== Step {step+1}/{num_steps} ===")
                
                # 現在のステップの値を計算
                progress = (step + 1) / num_steps
                if invert:
                    # Invertモードでは終了値から開始値へ変化
                    step_value = shape_key_end_value - (shape_key_end_value - shape_key_start_value) * progress
                else:
                    # 通常モードでは開始値から終了値へ変化
                    step_value = shape_key_start_value + (shape_key_end_value - shape_key_start_value) * progress
                
                print(f"Shape key value: {step_value}")
                
                # 頂点グループに基づいて制御点をフィルタリング
                filtered_indices = filter_control_points_by_vertex_groups(source_obj, selected_indices, step_value)
                
                if len(filtered_indices) < 4:
                    print(f"Warning: Very few valid control points at step {step+1} ({len(filtered_indices)}).")
                    if len(filtered_indices) == 0:
                        print(f"Skipping step {step+1}: No valid control points.")
                        continue
                
                print(f"Control points: {len(selected_indices)} -> {len(filtered_indices)} (after vertex group filtering)")
                
                # 変形前の状態を取得（バウンディングボックス計算用）
                current_basis_local = np.array([evaluated_source.data.vertices[i].co.copy() for i in filtered_indices])
                
                # 変形前の状態をワールド座標に変換
                current_basis = np.zeros_like(current_basis_local)
                for i, basis_co in enumerate(current_basis_local):
                    basis_v = Vector((basis_co[0], basis_co[1], basis_co[2], 1.0))
                    world_basis = source_world_matrix @ basis_v
                    current_basis[i] = np.array([world_basis[0], world_basis[1], world_basis[2]])
                
                # 現在のステップでのフィールドを生成（変形前のソースオブジェクトを使用）
                print(f"Generating Deformation Field for step {step+1}...")
                field_vertices = create_adaptive_deformation_field(
                    target_obj=source_obj,
                    base_grid_spacing=bpy.context.scene.rbf_base_grid_spacing,
                    surface_distance=bpy.context.scene.rbf_surface_distance,
                    max_distance=bpy.context.scene.rbf_max_distance,
                    min_distance=bpy.context.scene.rbf_min_distance,
                    density_falloff=bpy.context.scene.rbf_density_falloff,
                    bbox_scale_factor=bpy.context.scene.rbf_bbox_scale_factor,
                    use_selected_vertices=selected_only
                )
                
                if field_vertices is None:
                    print(f"Failed to generate field at step {step+1}")
                    continue
                
                # シェイプキーの値を更新して変形後の状態を取得
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = step_value
                
                # シーンを更新
                bpy.context.view_layer.update()
                
                # 評価後のオブジェクトを再取得
                depsgraph.update()
                evaluated_source_deformed = source_obj.evaluated_get(depsgraph)
                
                # 変形後の頂点位置を取得
                current_deformed_local = np.array([evaluated_source_deformed.data.vertices[i].co.copy() for i in filtered_indices])
                
                # 変形後の位置をワールド座標に変換
                current_deformed = np.zeros_like(current_deformed_local)
                for i, deformed_co in enumerate(current_deformed_local):
                    deformed_v = Vector((deformed_co[0], deformed_co[1], deformed_co[2], 1.0))
                    world_deformed = source_world_matrix @ deformed_v
                    current_deformed[i] = np.array([world_deformed[0], world_deformed[1], world_deformed[2]])
                
                # 法線方向に制御点を追加する場合
                if add_normal_control_points:
                    current_basis_extended, current_deformed_extended = add_normal_control_points_func(
                        source_obj, 
                        filtered_indices, 
                        current_basis, 
                        current_deformed, 
                        normal_distance
                    )
                else:
                    current_basis_extended = current_basis
                    current_deformed_extended = current_deformed
                
                # 事前に特定された重複制御点を除外
                if len(overlapping_indices) > 0:
                    print(f"Excluding {len(overlapping_indices)} pre-identified duplicate control points")
                    # 重複していない制御点のインデックスを取得
                    all_indices = np.arange(len(current_basis_extended))
                    valid_indices = np.setdiff1d(all_indices, overlapping_indices)
                    
                    if len(valid_indices) < len(current_basis_extended):
                        current_basis_extended = current_basis_extended[valid_indices]
                        current_deformed_extended = current_deformed_extended[valid_indices]
                        print(f"Excluded duplicate control points: using {len(valid_indices)} control points")
                
                # 変位の最大値をチェック
                displacements = current_deformed_extended - current_basis_extended
                max_disp = np.max(np.linalg.norm(displacements, axis=1))
                print(f"Maximum control point displacement: {max_disp}")
                
                # selected_onlyの場合、フォールオフ用の部分メッシュを作成
                falloff_source_obj = None
                if selected_only and original_selected_vertices:
                    falloff_source_obj = create_partial_mesh_from_vertices(source_obj, original_selected_vertices)
                    print(f"Created partial mesh for falloff ({len(original_selected_vertices)} vertices)")
                
                # RBF補間を実行
                target_deformed, target_world_vertices, target_displacements = rbf_interpolation(
                    current_basis_extended, 
                    current_deformed_extended, 
                    field_vertices, 
                    source_obj, 
                    epsilon,
                    10000,  # batch_size
                    falloff_source_obj
                )
                
                # 部分メッシュのクリーンアップ
                if falloff_source_obj:
                    mesh_data = falloff_source_obj.data
                    bpy.data.objects.remove(falloff_source_obj, do_unlink=True)
                    bpy.data.meshes.remove(mesh_data)
                    print("Deleted partial mesh for falloff")
                
                # 結果を保存
                all_target_world_vertices.append(target_world_vertices)
                all_displacements.append(target_displacements)
                
                print(f"Step {step+1} displacement calculation complete")
            
            # シェイプキーの値を元に戻す
            for key_name, value in original_values.items():
                source_obj.data.shape_keys.key_blocks[key_name].value = value
            
            # シーンを更新
            bpy.context.view_layer.update()
            
            print(f"Used maximum of {len(current_basis_extended)} vertices as control points")
            
            # Deformation Fieldデータを保存
            # 最初のフィールドオブジェクトを基準として使用
            save_field_data_multi_step(
                field_data_path,
                all_target_world_vertices,  # 各ステップの座標をすべて保存
                all_displacements,
                num_steps,
                old_version=False,
                enable_x_mirror=bpy.context.scene.rbf_enable_x_mirror
            )
            print(f"Saved Deformation Field data: {field_data_path}")
            
            # 結果をリストに追加
            results.append({
                'target_world_vertices': all_target_world_vertices,
                'displacements': all_displacements,
                'filepath': field_data_path,
                'invert': invert
            })
    
    finally:
        # 処理完了後、オブジェクトの表示状態を元に戻す
        restore_objects_visibility(objects_to_check, original_visibility_states)
        
    # 生成されたフィールドオブジェクトをリストとして返す（下位互換性のため最初の結果を返す）
    # 注意：オブジェクトは既に削除されているため、空のリストを返す
    if results:
        return [], results[0]['target_world_vertices'], results[0]['displacements']
    else:
        return [], [], []


def save_field_data_multi_step(filepath, all_field_points, all_delta_positions, num_steps, old_version=False, enable_x_mirror=True):
    """
    複数ステップのDeformation Fieldの変形前後の差分をnumpy arrayとして直接保存
    各ステップの座標をそれぞれ保存
    enable_x_mirrorが有効な場合、X座標が0以上のデータのみを保存
    """
    
    # オブジェクトのワールド行列を保存
    world_matrix = np.identity(4)
    
    kdtree_query_k = 27
    
    # RBF補間のパラメータを追加
    rbf_epsilon = 0.00001  # 固定値
    rbf_smoothing = 0.0    # スムージングパラメータ
    
    # enable_x_mirrorが有効でold_versionではない場合、X座標が0以上のデータのみフィルタリング
    if not old_version and enable_x_mirror:
        filtered_field_points = []
        filtered_delta_positions = []
        
        for step in range(num_steps):
            field_points = all_field_points[step]
            delta_positions = all_delta_positions[step]
            
            if len(field_points) > 0:
                # X座標が0以上のインデックスを取得
                x_positive_mask = field_points[:, 0] >= 0.0
                filtered_field = field_points[x_positive_mask]
                filtered_delta = delta_positions[x_positive_mask]
                
                filtered_field_points.append(filtered_field.astype(np.float32))
                filtered_delta_positions.append(filtered_delta.astype(np.float32))
                
                print(f"Step {step+1}: original vertices {len(field_points)} -> after filter {len(filtered_field)}")
            else:
                filtered_field_points.append(np.array([]))
                filtered_delta_positions.append(np.array([]))
                print(f"Step {step+1}: field vertex count 0")
        
        # フィルタ後のデータを使用
        all_field_points = filtered_field_points
        all_delta_positions = filtered_delta_positions
    elif not old_version and not enable_x_mirror:
        # ミラーが無効の場合、float32にキャストのみ行う
        filtered_field_points = []
        filtered_delta_positions = []
        
        for step in range(num_steps):
            field_points = all_field_points[step]
            delta_positions = all_delta_positions[step]
            
            if len(field_points) > 0:
                filtered_field_points.append(field_points.astype(np.float32))
                filtered_delta_positions.append(delta_positions.astype(np.float32))
                print(f"Step {step+1}: vertex count {len(field_points)} (no mirror filter)")
            else:
                filtered_field_points.append(np.array([]))
                filtered_delta_positions.append(np.array([]))
                print(f"Step {step+1}: field vertex count 0")
        
        # キャスト後のデータを使用
        all_field_points = filtered_field_points
        all_delta_positions = filtered_delta_positions
   
    # データを保存
    np.savez(filepath,
             all_field_points=np.array(all_field_points, dtype=object),  # 各ステップの座標を保存
             all_delta_positions=np.array(all_delta_positions, dtype=object),
             num_steps=num_steps,
             world_matrix=world_matrix,
             kdtree_query_k=kdtree_query_k,
             rbf_epsilon=rbf_epsilon,
             rbf_smoothing=rbf_smoothing,
             enable_x_mirror=enable_x_mirror)
    
    print(f"Saved Deformation Field differential data: {filepath}")
    print(f"Number of steps: {num_steps}")
    for step in range(num_steps):
        print(f"Step {step+1}: vertex count {len(all_field_points[step])}")
    print(f"RBF function: multi_quadratic_biharmonic, epsilon: {rbf_epsilon}, smoothing: {rbf_smoothing}")


def get_vertex_groups_and_weights(mesh_obj, vertex_index):
    """指定された頂点のグループとウェイトを取得"""
    groups = {}
    for group in mesh_obj.vertex_groups:
        try:
            weight = group.weight(vertex_index)
            groups[group.name] = weight
        except RuntimeError:
            continue
    return groups


def filter_control_points_by_vertex_groups(mesh_obj, selected_indices, step_value):
    """
    頂点グループのウェイトに基づいて制御点をフィルタリングする
    
    Parameters:
    - mesh_obj: メッシュオブジェクト
    - selected_indices: 制御点候補の頂点インデックスリスト
    - step_value: 現在のステップ値
    
    Returns:
    - フィルタリング後の頂点インデックスリスト
    """
    filtered_indices = []
    
    # exclude_min と exclude_max 頂点グループを取得
    exclude_min_group = mesh_obj.vertex_groups.get("exclude_min")
    exclude_max_group = mesh_obj.vertex_groups.get("exclude_max")
    
    for vertex_index in selected_indices:
        should_exclude = False
        
        # exclude_min グループの処理
        if exclude_min_group and exclude_max_group:
            weight_min = 1.0
            weight_max = 0.0
            try:
                weight_min = exclude_min_group.weight(vertex_index)
                weight_max = exclude_max_group.weight(vertex_index)
                if weight_min < step_value and weight_max > step_value:
                    should_exclude = True
            except RuntimeError:
                # 頂点がグループに属していない場合は除外しない
                pass
        
        # 除外されない場合は制御点として使用
        if not should_exclude:
            filtered_indices.append(vertex_index)
    
    return filtered_indices


def get_armature_from_modifier(mesh_obj):
    """Armatureモディファイアからアーマチュアを取得"""
    for modifier in mesh_obj.modifiers:
        if modifier.type == 'ARMATURE':
            return modifier.object
    return None


def calculate_inverse_pose_matrix(mesh_obj, armature_obj, vertex_index):
    """指定された頂点のポーズ逆行列を計算"""

    # 頂点グループとウェイトの取得
    weights = get_vertex_groups_and_weights(mesh_obj, vertex_index)
    if not weights:
        raise ValueError(f"頂点 {vertex_index} にウェイトが割り当てられていません")

    # 最終的な変換行列の初期化
    final_matrix = Matrix.Identity(4)
    final_matrix.zero()
    total_weight = 0

    # 各ボーンの影響を計算
    for bone_name, weight in weights.items():
        if weight > 0 and bone_name in armature_obj.data.bones:
            bone = armature_obj.data.bones[bone_name]
            pose_bone = armature_obj.pose.bones.get(bone_name)
            if bone and pose_bone:
                # ボーンの最終的な行列を計算
                mat = armature_obj.matrix_world @ \
                      pose_bone.matrix @ \
                      bone.matrix_local.inverted() @ \
                      armature_obj.matrix_world.inverted()
                
                # ウェイトを考慮して行列を加算
                final_matrix += mat * weight
                total_weight += weight

    # ウェイトの合計で正規化
    if total_weight > 0:
        final_matrix = final_matrix * (1.0 / total_weight)

    # 逆行列を計算して返す
    return final_matrix.inverted()


def apply_field_data(target_obj, field_data_path, shape_key_name="RBFDeform"):
    """
    保存されたDeformation Field差分データを読み込んでメッシュに適用（RBF補間版）
    各ステップの座標を使用して変形を適用
    """
    # データの読み込み
    data = np.load(field_data_path, allow_pickle=True)
    
    # データ形式の確認と読み込み
    if 'all_field_points' in data:
        # 新形式：各ステップの座標が保存されている場合
        all_field_points = data['all_field_points']
        all_delta_positions = data['all_delta_positions']
        num_steps = int(data.get('num_steps', len(all_delta_positions)))
        print(f"Detected multi-step data (new format): {num_steps} steps")

        # ミラー設定を確認（データに含まれていない場合はそのまま使用）
        enable_x_mirror = data.get('enable_x_mirror', False)
        print(f"X-axis mirror setting: {'enabled' if enable_x_mirror else 'disabled'}")
        
        if enable_x_mirror:
            # X軸ミラーリング：X座標が0より大きいデータを負に反転してミラーデータを追加
            mirrored_field_points = []
            mirrored_delta_positions = []
            
            for step in range(num_steps):
                field_points = all_field_points[step].copy()
                delta_positions = all_delta_positions[step].copy()
                
                if len(field_points) > 0:
                    # X座標が0より大きいデータを検索
                    x_positive_mask = field_points[:, 0] > 0.0
                    if np.any(x_positive_mask):
                        # ミラーデータを作成
                        mirror_field_points = field_points[x_positive_mask].copy()
                        mirror_delta_positions = delta_positions[x_positive_mask].copy()
                        
                        # X座標とX成分の変位を反転
                        mirror_field_points[:, 0] *= -1.0
                        mirror_delta_positions[:, 0] *= -1.0
                        
                        # 元のデータとミラーデータを結合
                        combined_field_points = np.vstack([field_points, mirror_field_points])
                        combined_delta_positions = np.vstack([delta_positions, mirror_delta_positions])
                        
                        mirrored_field_points.append(combined_field_points)
                        mirrored_delta_positions.append(combined_delta_positions)
                        
                        print(f"Step {step+1}: original vertices {len(field_points)} -> after mirror {len(combined_field_points)}")
                    else:
                        mirrored_field_points.append(field_points)
                        mirrored_delta_positions.append(delta_positions)
                        print(f"Step {step+1}: field vertex count {len(field_points)} (no mirror targets)")
                else:
                    mirrored_field_points.append(field_points)
                    mirrored_delta_positions.append(delta_positions)
                    print(f"Step {step+1}: field vertex count 0")
            
            # ミラー適用後のデータを使用
            all_field_points = mirrored_field_points
            all_delta_positions = mirrored_delta_positions
        else:
            # ミラーが無効の場合、元のデータをそのまま使用
            print("X-axis mirroring is disabled, using original data")
            for step in range(num_steps):
                print(f"Step {step+1}: field vertex count {len(all_field_points[step])}")
        
    else:
        # 後方互換性のため、単一ステップのデータも処理
        field_points = data.get('field_points')
        delta_positions = data.get('delta_positions')
        all_field_points = [field_points]
        all_delta_positions = [delta_positions]
        num_steps = 1
        print("Detected single-step data")
    
    field_matrix = Matrix(data['world_matrix'])
    field_matrix_inv = field_matrix.inverted()
    
    # RBFパラメータの読み込み
    rbf_epsilon = float(data.get('rbf_epsilon', 0.00001))
    
    print(f"RBF interpolation parameters: function=multi_quadratic_biharmonic, epsilon={rbf_epsilon}")
    
    # 評価されたメッシュを取得（モディファイア適用後）
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = target_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.data
    
    # シェイプキーの準備
    if target_obj.data.shape_keys is None:
        target_obj.shape_key_add(name='Basis')
    
    # シェイプキーを作成
    shape_key = target_obj.shape_key_add(name=shape_key_name)
    shape_key.value = 1.0
    
    # 頂点データをNumPy配列として準備
    vertices = np.array([v.co for v in eval_mesh.vertices])
    num_vertices = len(vertices)
    
    # 累積変位を初期化
    cumulative_displacements = np.zeros((num_vertices, 3))
    # 現在の頂点位置（ワールド座標）を保存
    current_world_positions = np.array([target_obj.matrix_world @ Vector(v) for v in vertices])
    
    # 各ステップの変位を累積的に適用
    for step in range(num_steps):
        field_points = all_field_points[step]
        delta_positions = all_delta_positions[step]
        
        print(f"Applying deformation for step {step+1}/{num_steps}...")
        print(f"Number of field vertices to use: {len(field_points)}")
        
        # SciPyの利用可能性をチェック
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPyが利用できません。NumPy・SciPy再インストールボタンを使用してインストールしてください。")
        
        # KDTreeを使用して近傍点を検索（各ステップで新しいKDTreeを構築）
        kdtree = cKDTree(field_points)
        
        # カスタムRBF補間で新しい頂点位置を計算
        batch_size = 1000
        step_displacements = np.zeros((num_vertices, 3))
        
        for start_idx in range(0, num_vertices, batch_size):
            end_idx = min(start_idx + batch_size, num_vertices)
            batch_vertices = vertices[start_idx:end_idx]
            
            # バッチ内の全頂点をフィールド空間に変換（現在の累積変位を考慮）
            batch_world = current_world_positions[start_idx:end_idx].copy()
            batch_field = np.array([field_matrix_inv @ Vector(v) for v in batch_world])
            
            # 各頂点ごとに逆距離加重法で補間
            batch_displacements = np.zeros((len(batch_field), 3))
            
            for i, point in enumerate(batch_field):
                # 近傍点を検索（最大8点）
                k = min(8, len(field_points))
                distances, indices = kdtree.query(point, k=k)
                
                # 距離が0の場合（完全に一致する点がある場合）
                if distances[0] < 1e-10:
                    batch_displacements[i] = delta_positions[indices[0]]
                    continue
                
                # 逆距離の重みを計算
                weights = 1.0 / np.sqrt(distances**2 + rbf_epsilon**2)
                
                # 重みの正規化
                weights /= np.sum(weights)
                
                # 重み付き平均で変位を計算
                weighted_deltas = delta_positions[indices] * weights[:, np.newaxis]
                batch_displacements[i] = np.sum(weighted_deltas, axis=0)
            
            # ワールド空間での変位を計算
            for i, displacement in enumerate(batch_displacements):
                world_displacement = field_matrix.to_3x3() @ Vector(displacement)
                step_displacements[start_idx + i] = world_displacement
                
                # 現在のワールド位置を更新（次のステップのために）
                current_world_positions[start_idx + i] += world_displacement
        
        # このステップの変位を累積変位に追加
        cumulative_displacements += step_displacements
        
        print(f"Step {step+1} complete: max displacement {np.max(np.linalg.norm(step_displacements, axis=1)):.6f}")
    
    # アーマチュアの取得
    armature_obj = get_armature_from_modifier(target_obj)
    if not armature_obj:
        print("Armature modifier not found")
    
    # 累積変位を適用して最終的な頂点位置を計算
    results = np.zeros((num_vertices, 3))
    for i in range(num_vertices):
        # 元のワールド位置に累積変位を加えた位置をローカル座標に変換
        world_pos = target_obj.matrix_world @ Vector(vertices[i])
        final_world_pos = world_pos + Vector(cumulative_displacements[i])
        if armature_obj:
            matrix_armature_inv = calculate_inverse_pose_matrix(target_obj, armature_obj, i)
            undeformed_world_pos = matrix_armature_inv @ Vector(final_world_pos)
        else:
            undeformed_world_pos = Vector(final_world_pos)
        local_pos = target_obj.matrix_world.inverted() @ undeformed_world_pos
        results[i] = local_pos
    
    # 結果をシェイプキーに適用
    for i, local_pos in enumerate(results):
        shape_key.data[i].co = local_pos
    
    print(f"Applied cumulative deformation from all steps: {shape_key_name}")
    print(f"Final maximum cumulative displacement: {np.max(np.linalg.norm(cumulative_displacements, axis=1)):.6f}")


# プロパティの定義
def create_field_object_from_data(field_data_path, target_step=1, object_name="FieldVisualization"):
    """
    保存されたDeformation Field差分データを読み込んでフィールドをBlenderオブジェクトとして作成
    各ステップの変位をシェイプキーとして保存
    
    Parameters:
    - field_data_path: フィールドデータファイルのパス
    - target_step: 表示するステップ（1から始まる）
    - object_name: 作成するオブジェクトの名前
    
    Returns:
    - 作成されたBlenderオブジェクト
    """
    # データの読み込み
    data = np.load(field_data_path, allow_pickle=True)
    
    # データ形式の確認と読み込み
    if 'all_field_points' in data:
        # 新形式：各ステップの座標が保存されている場合
        all_field_points = data['all_field_points']
        all_delta_positions = data['all_delta_positions']
        num_steps = int(data.get('num_steps', len(all_delta_positions)))
        print(f"Detected multi-step data (new format): {num_steps} steps")
    elif 'field_points' in data and 'all_delta_positions' in data:
        # 旧形式：単一の座標セットが保存されている場合
        field_points = data['field_points']
        all_delta_positions = data['all_delta_positions']
        num_steps = int(data.get('num_steps', len(all_delta_positions)))
        
        # 旧形式の場合、すべてのステップで同じ座標を使用
        all_field_points = [field_points for _ in range(num_steps)]
        print(f"Detected multi-step data (old format): {num_steps} steps")
    else:
        # 後方互換性のため、単一ステップのデータも処理
        field_points = data.get('field_points', data.get('delta_positions', []))
        delta_positions = data.get('delta_positions', data.get('all_delta_positions', [[]])[0])
        all_field_points = [field_points]
        all_delta_positions = [delta_positions]
        num_steps = 1
        print("Detected single-step data")
    
    # ステップ数の検証
    if target_step < 1 or target_step > num_steps:
        raise ValueError(f"ステップ {target_step} は範囲外です（有効範囲: 1-{num_steps}）")
    
    # 指定されたステップのデータを取得（0ベースに変換）
    step_index = target_step - 1
    field_points = all_field_points[step_index]
    
    if len(field_points) == 0:
        raise ValueError("フィールドポイントが空です")
    
    print(f"Field point count for step {target_step}/{num_steps}: {len(field_points)}")
    
    # メッシュオブジェクトを作成
    mesh = bpy.data.meshes.new(object_name + "_mesh")
    obj = bpy.data.objects.new(object_name, mesh)
    
    # 頂点座標を設定
    vertices = []
    for point in field_points:
        if hasattr(point, '__len__') and len(point) >= 3:
            vertices.append([point[0], point[1], point[2]])
        else:
            print(f"Warning: Invalid point data: {point}")
    
    if not vertices:
        raise ValueError("有効な頂点が見つかりません")
    
    mesh.from_pydata(vertices, [], [])
    mesh.update()
    
    # シーンに追加
    bpy.context.scene.collection.objects.link(obj)
    
    # ベースシェイプキーを作成
    obj.shape_key_add(name='Basis')
    
    # 指定されたステップの変位をシェイプキーとして追加
    step_name = f"Step_{target_step:02d}_Displacement"
    shape_key = obj.shape_key_add(name=step_name)
    
    # 指定されたステップの変位を取得
    target_delta_positions = all_delta_positions[step_index]
    
    # フィールドポイントの数と変位の数が一致することを確認
    field_count = len(field_points)
    delta_count = len(target_delta_positions)
    
    if field_count != delta_count:
        print(f"Warning: Field point count ({field_count}) and displacement count ({delta_count}) mismatch at step {target_step}")
    else:
        # シェイプキーに変位を適用
        for i in range(min(len(vertices), len(target_delta_positions))):
            if i < len(shape_key.data):
                # 元の頂点位置に変位を加算
                original_pos = vertices[i]
                displacement = target_delta_positions[i]
                
                if hasattr(displacement, '__len__') and len(displacement) >= 3:
                    shape_key.data[i].co = [
                        original_pos[0] + displacement[0],
                        original_pos[1] + displacement[1],
                        original_pos[2] + displacement[2]
                    ]
                else:
                    print(f"Warning: Invalid displacement data at step {target_step}: index {i}")
        
        print(f"Created shape key '{step_name}': {len(target_delta_positions)} displacements")
    
    # オブジェクトを選択してアクティブにする
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    print(f"Created field object '{object_name}'")
    print(f"Vertex count: {len(vertices)}, target step: {target_step}/{num_steps}")
    
    return obj


def register_properties():
    bpy.types.Scene.rbf_source_obj = bpy.props.PointerProperty(
        name="Source Mesh",
        description="ソースメッシュオブジェクト",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH'
    )
    
    bpy.types.Scene.rbf_source_shape_key = bpy.props.StringProperty(
        name="Source Shape Key",
        description="ソースオブジェクトのシェイプキー"
    )
    
    # シェイプキーの値の範囲を指定するプロパティ
    bpy.types.Scene.rbf_shape_key_start_value = bpy.props.FloatProperty(
        name="Shape Key Start Value",
        description="シェイプキーの開始値",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=3
    )
    
    bpy.types.Scene.rbf_shape_key_end_value = bpy.props.FloatProperty(
        name="Shape Key End Value", 
        description="シェイプキーの終了値",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3
    )
    
    bpy.types.Scene.rbf_selected_only = bpy.props.BoolProperty(
        name="Selected Vertices Only",
        description="選択された頂点のみを制御点として使用",
        default=False
    )
    
    bpy.types.Scene.rbf_save_shape_key_mode = bpy.props.BoolProperty(
        name="Save Self Shape Key Transform",
        description="ソースアバター自身のシェイプキー変形を保存する（通常と逆の両方）",
        default=False
    )
    
    bpy.types.Scene.rbf_keep_first_field = bpy.props.BoolProperty(
        name="Keep First Field for Debug",
        description="デバッグ用に最初の変形フィールドを削除せずに残す",
        default=False
    )
    
    bpy.types.Scene.rbf_epsilon = bpy.props.FloatProperty(
        name="Epsilon",
        description="RBFパラメータ（0以下の場合は自動計算）",
        default=0.00001,
        precision=6
    )
    
    bpy.types.Scene.rbf_num_steps = bpy.props.IntProperty(
        name="Number of Steps",
        description="変形を分割するステップ数",
        min=1,
        default=1
    )
    
    # アバター名プロパティを追加
    bpy.types.Scene.rbf_source_avatar_name = bpy.props.StringProperty(
        name="Source Avatar Name",
        description="変換元のアバター名",
        default=""
    )
    
    bpy.types.Scene.rbf_target_avatar_name = bpy.props.StringProperty(
        name="Target Avatar Name",
        description="変換先のアバター名",
        default=""
    )
    
    # アバターデータファイルのプロパティを追加
    bpy.types.Scene.rbf_source_avatar_data_file = bpy.props.StringProperty(
        name="Source Avatar Data",
        description="変換元のアバターデータファイル",
        default="avatar_data_template.json",
        subtype='FILE_PATH'
    )
    
    bpy.types.Scene.rbf_target_avatar_data_file = bpy.props.StringProperty(
        name="Target Avatar Data",
        description="変換先のアバターデータファイル",
        default="avatar_data_target.json",
        subtype='FILE_PATH'
    )
    
    # 法線制御点のプロパティ
    bpy.types.Scene.rbf_add_normal_control_points = bpy.props.BoolProperty(
        name="Add Normal Control Points",
        description="制御点の法線方向に追加制御点を配置する",
        default=False
    )
    
    bpy.types.Scene.rbf_normal_distance = bpy.props.FloatProperty(
        name="Normal Distance",
        description="法線方向への距離（ワールド座標系、負の値で内側、正の値で外側）",
        default=-0.0002,
        min=-0.005,
        max=0.005,
        precision=5
    )
    
    # Xミラープロパティ
    bpy.types.Scene.rbf_enable_x_mirror = bpy.props.BoolProperty(
        name="Enable X Mirror",
        description="X軸ミラーリングを有効にする（X座標が0以上のデータのみ保存し、読み込み時に自動でミラー）",
        default=True
    )
    
    bpy.types.Scene.rbf_apply_shape_key_name = bpy.props.StringProperty(
        name="Apply Shape Key Name",
        description="適用するシェイプキーの名前",
        default="RBF_Deform"
    )
    
    # Deformation Fieldパラメータ
    bpy.types.Scene.rbf_base_grid_spacing = bpy.props.FloatProperty(
        name="Base Grid Spacing",
        description="基本グリッドの間隔（メートル単位）",
        default=0.00250,
        min=0.0001,
        max=0.1,
        precision=5
    )
    
    bpy.types.Scene.rbf_surface_distance = bpy.props.FloatProperty(
        name="Surface Distance",
        description="ターゲットメッシュ表面からの最大距離",
        default=2.0,
        min=0.1,
        max=10.0,
        precision=3
    )
    
    bpy.types.Scene.rbf_max_distance = bpy.props.FloatProperty(
        name="Max Distance",
        description="最大ウェイト距離",
        default=0.2,
        min=0.001,
        max=1.0,
        precision=4
    )
    
    bpy.types.Scene.rbf_min_distance = bpy.props.FloatProperty(
        name="Min Distance",
        description="最小ウェイト距離",
        default=0.005,
        min=0.0001,
        max=0.1,
        precision=5
    )
    
    bpy.types.Scene.rbf_density_falloff = bpy.props.FloatProperty(
        name="Density Falloff",
        description="密度減衰率（値を大きくすると段階が早く変化）",
        default=4.0,
        min=1.0,
        max=10.0,
        precision=2
    )
    
    bpy.types.Scene.rbf_bbox_scale_factor = bpy.props.FloatProperty(
        name="BBox Scale Factor",
        description="Bounding Boxのスケール倍率",
        default=1.5,
        min=1.0,
        max=5.0,
        precision=2
    )
    
    # ポーズ関連のプロパティ
    bpy.types.Scene.rbf_pose_invert = bpy.props.BoolProperty(
        name="Invert Pose",
        description="ポーズを逆変換で適用するかどうか",
        default=False
    )
    
    # デバッグ用プロパティ
    bpy.types.Scene.rbf_show_debug_info = bpy.props.BoolProperty(
        name="Show Debug Info",
        description="デバッグ情報を表示する",
        default=False
    )
    
    # フィールド可視化用プロパティ
    bpy.types.Scene.rbf_field_step = bpy.props.IntProperty(
        name="Field Step",
        description="可視化するフィールドのステップ数",
        min=1,
        default=1
    )
    
    bpy.types.Scene.rbf_field_use_inverse = bpy.props.BoolProperty(
        name="Use Inverse Data",
        description="逆変換データを使用する",
        default=False
    )
    
    bpy.types.Scene.rbf_field_object_name = bpy.props.StringProperty(
        name="Field Object Name",
        description="作成するフィールドオブジェクトの名前",
        default="FieldVisualization"
    )


def get_armature_from_source_object(source_obj):
    """
    ソースオブジェクトからArmatureモディファイアを検索し、対象のArmatureオブジェクトを取得
    
    Parameters:
        source_obj: ソースメッシュオブジェクト
        
    Returns:
        bpy.types.Object: Armatureオブジェクト、見つからない場合はNone
    """
    if not source_obj or source_obj.type != 'MESH':
        return None
    
    for modifier in source_obj.modifiers:
        if modifier.type == 'ARMATURE' and modifier.object:
            return modifier.object
    return None


# ツールパネルの設定
class RBF_PT_DeformationPanel(bpy.types.Panel):
    bl_label = "MochiFitter"
    bl_idname = "RBF_PT_DeformationPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MochiFitter'  # ここで「MochiFitter」タブに表示するよう設定
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # アバター名設定セクション
        box = layout.box()
        box.label(text="アバター設定", icon='ARMATURE_DATA')
        
        row = box.row()
        row.prop(scene, "rbf_source_avatar_name")
        
        row = box.row()
        row.prop(scene, "rbf_target_avatar_name")
        
        # アバターデータファイル設定
        col = box.column(align=True)
        col.label(text="アバターデータファイル:")
        col.prop(scene, "rbf_source_avatar_data_file", text="Source")
        col.prop(scene, "rbf_target_avatar_data_file", text="Target")
        
        # アバター設定入れ替えボタン
        row = box.row()
        row.operator("object.swap_avatar_settings", text="ソース⇄ターゲット入れ替え", icon='ARROW_LEFTRIGHT')

        # ソースオブジェクトの選択
        row = box.row()
        row.prop(scene, "rbf_source_obj")
        
        # Humanoidボーン Inherit Scale 設定ボタン
        row = box.row()
        humanoid_bone_ready = (context.active_object and 
                              context.active_object.type == 'ARMATURE' and 
                              scene.rbf_source_avatar_data_file)
        if humanoid_bone_ready:
            row.operator("object.set_humanoid_bone_inherit_scale", 
                        text="Humanoidボーン Inherit Scale → Average", 
                        icon='BONE_DATA')
        else:
            if not context.active_object or context.active_object.type != 'ARMATURE':
                row.label(text="Armatureオブジェクトを選択してください", icon='ERROR')
            elif not scene.rbf_source_avatar_data_file:
                row.label(text="ソースアバターデータファイルを設定してください", icon='ERROR')
        
        # ベースポーズ差分セクション
        box = layout.box()
        box.label(text="ベースポーズ差分", icon='ARMATURE_DATA')
        
        # ベースポーズ保存ボタン
        row = box.row()
        armature_available = False
        base_pose_save_ready = False
        if scene.rbf_source_avatar_name and context.active_object and context.active_object.type == 'ARMATURE':
            armature_available = True
            if scene.rbf_source_avatar_data_file:
                base_pose_save_ready = True
                base_pose_filename = f"pose_basis_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}.json"
                row.operator("object.save_base_pose_diff", text=f"ベースポーズを保存", icon='EXPORT')
                row = box.row()
                row.label(text=f"保存先: {base_pose_filename}", icon='FILE')
            else:
                row.label(text="ソースアバターデータファイルを指定", icon='ERROR')
        else:
            if not scene.rbf_source_avatar_name:
                row.label(text="ソースアバター名を設定", icon='ERROR')
            elif not context.active_object or context.active_object.type != 'ARMATURE':
                row.label(text="Armatureオブジェクトを選択してください", icon='ERROR')
            else:
                row.label(text="設定を完了してください", icon='ERROR')
        
        # ベースポーズ適用セクション
        row = box.row()
        row.prop(scene, "rbf_pose_invert")
        
        row = box.row()
        base_pose_apply_ready = armature_available and scene.rbf_source_avatar_data_file
        if base_pose_apply_ready:
            row.operator("object.apply_base_pose_diff", text="ベースポーズを適用", icon='IMPORT')
        else:
            if armature_available:
                row.label(text="アバターデータファイルを指定", icon='ERROR')
            else:
                row.label(text="設定を完了してください", icon='ERROR')

        # ポーズ差分セクション
        box = layout.box()
        box.label(text="ポーズ差分", icon='POSE_HLT')
        
        # ポーズ保存ボタン
        row = box.row()
        pose_armature_available = False
        pose_save_ready = False
        if scene.rbf_source_avatar_name and scene.rbf_target_avatar_name and context.active_object and context.active_object.type == 'ARMATURE':
            pose_armature_available = True
            if scene.rbf_source_avatar_data_file:
                pose_save_ready = True
                pose_filename = f"posediff_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}.json"
                row.operator("object.save_pose_diff", text=f"ポーズを保存", icon='EXPORT')
                row = box.row()
                row.label(text=f"保存先: {pose_filename}", icon='FILE')
            else:
                row.label(text="ソースアバターデータファイルを指定", icon='ERROR')
        else:
            if not scene.rbf_source_avatar_name or not scene.rbf_target_avatar_name:
                row.label(text="アバター名を設定してください", icon='ERROR')
            elif not context.active_object or context.active_object.type != 'ARMATURE':
                row.label(text="Armatureオブジェクトを選択してください", icon='ERROR')
            else:
                row.label(text="設定を完了してください", icon='ERROR')
        
        # ポーズ適用セクション
        row = box.row()
        row.prop(scene, "rbf_pose_invert")
        
        row = box.row()
        pose_apply_ready = pose_armature_available and scene.rbf_target_avatar_data_file
        if pose_apply_ready:
            row.operator("object.apply_pose_diff", text="ポーズを適用", icon='IMPORT')
        else:
            if pose_armature_available:
                row.label(text="ターゲットアバターデータファイルを指定", icon='ERROR')
            else:
                row.label(text="設定を完了してください", icon='ERROR')
        
        # 区切り線
        layout.separator()
        
        # UIの描画
        box = layout.box()
        box.label(text="変形フィールド設定", icon='MESH_DATA')
        
        # ソースオブジェクトが選択されていればシェイプキーのドロップダウンを表示
        if scene.rbf_source_obj and scene.rbf_source_obj.data.shape_keys:
            row = box.row()
            row.label(text="シェイプキー:")
            
            # シェイプキー選択用のドロップダウン
            row = box.row()
            shape_keys = [key.name for key in scene.rbf_source_obj.data.shape_keys.key_blocks if key.name != "Basis"]
            if shape_keys:
                op = row.operator("object.select_rbf_shape_key", text=scene.rbf_source_shape_key if scene.rbf_source_shape_key else "シェイプキーを選択")
            else:
                row.label(text="有効なシェイプキーがありません")
        elif scene.rbf_source_obj:
            box.label(text="ソースオブジェクトにシェイプキーがありません", icon='ERROR')
        
        # シェイプキーの値の範囲設定
        if scene.rbf_source_obj and scene.rbf_source_obj.data.shape_keys and scene.rbf_source_shape_key:
            col = box.column(align=True)
            col.label(text="シェイプキー値の範囲:")
            row = col.row(align=True)
            row.prop(scene, "rbf_shape_key_start_value", text="開始値")
            row.prop(scene, "rbf_shape_key_end_value", text="終了値")
            
            # 範囲の妥当性チェック
            if scene.rbf_shape_key_start_value == scene.rbf_shape_key_end_value:
                col.label(text="開始値と終了値は異なる値を設定してください", icon='ERROR')
        
        # シェイプキー変形保存オプション
        col = box.column(align=True)
        col.prop(scene, "rbf_save_shape_key_mode")
        # 選択された頂点のみを使用するかどうか
        col.prop(scene, "rbf_selected_only")
        # デバッグオプション
        col.prop(scene, "rbf_keep_first_field")
        
        col = box.column(align=True)
        col.label(text="RBF変形設定:")
        # Epsilonの設定
        col.prop(scene, "rbf_epsilon")
        # ステップ数の設定
        col.prop(scene, "rbf_num_steps")
        
        # 法線制御点設定
        col = box.column(align=True)
        col.label(text="法線制御点設定:")
        col.prop(scene, "rbf_add_normal_control_points")
        if scene.rbf_add_normal_control_points:
            col.prop(scene, "rbf_normal_distance")
        
        # Deformation Fieldパラメータセクション
        # 基本パラメータ
        col = box.column(align=True)
        col.label(text="グリッド設定:")
        col.prop(scene, "rbf_base_grid_spacing")
        col.prop(scene, "rbf_bbox_scale_factor")
        
        # 距離パラメータ
        col = box.column(align=True)
        col.label(text="距離減衰設定:")
        col.prop(scene, "rbf_surface_distance")
        col.prop(scene, "rbf_max_distance")
        col.prop(scene, "rbf_min_distance")
        
        # 密度設定
        col = box.column(align=True)
        col.prop(scene, "rbf_density_falloff")
        
        # 実行ボタン
        box = layout.box()
        
        # 警告メッセージがあれば表示
        warning_msg = ""
        if not SCIPY_AVAILABLE:
            warning_msg = "SciPyが利用できません。NumPy・SciPy再インストールボタンを使用してください"
        elif not scene.rbf_source_avatar_name or not scene.rbf_target_avatar_name:
            warning_msg = "アバター名を設定してください"
        elif not scene.rbf_source_obj:
            warning_msg = "ソースオブジェクトを選択してください"
        elif not scene.rbf_source_obj.data.shape_keys:
            warning_msg = "ソースオブジェクトにシェイプキーがありません"
        elif not scene.rbf_source_shape_key:
            warning_msg = "ソースシェイプキーを選択してください"
        
        if warning_msg:
            box.label(text=warning_msg, icon='ERROR')
        else:
            # 実行ボタン（従来の方法）
            row = box.row()
            row.scale_y = 1.2
            op = row.operator("object.create_rbf_deformation", text="変形を保存（シングルスレッド）", icon='MOD_MESHDEFORM')
            
            # 一時データエクスポートボタン（新しい方法）
            row = box.row()
            row.scale_y = 1.5
            op = row.operator("object.export_rbf_temp_data", text="一時データエクスポート＆マルチスレッド処理", icon='PLAY')
            
            # Xミラーチェックボックス
            row = box.row()
            row.prop(scene, "rbf_enable_x_mirror", icon='MOD_MIRROR')
            
            # 注意書きを追加
            row = box.row()
            row.label(text="※ rbf_multithread_processor.pyが同じフォルダに必要です", icon='INFO')
            
            # 保存先ファイル名を表示
            row = box.row()
            if scene.rbf_save_shape_key_mode:
                # シェイプキー変形モードの場合（通常と逆の両方を保存）
                base_filename = f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}.npz"
                row.label(text=f"デフォルト名: {base_filename} + _inv.npz", icon='FILE')
                row = box.row()
                temp_filename = f"temp_rbf_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}.npz"
                row.label(text=f"一時ファイル: {temp_filename} + _inv.npz", icon='TEMP')
            else:
                # 通常のアバター間変形の場合（通常のみ）
                field_filename = f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}.npz"
                row.label(text=f"デフォルト名: {field_filename}", icon='FILE')
                row = box.row()
                temp_filename = f"temp_rbf_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}.npz"
                row.label(text=f"一時ファイル: {temp_filename}", icon='TEMP')
        
        # 区切り線
        layout.separator()
        
        # 保存したフィールドデータの適用セクション
        box = layout.box()
        box.label(text="保存した変形データを適用", icon='IMPORT')
        
        row = box.row()
        row.prop(scene, "rbf_apply_shape_key_name", text="シェイプキー名")
        
        # デフォルトの変形データ適用（通常）
        apply_row = box.row(align=True)
        apply_row.operator("rbf.apply_field_data", text="変形データ適用")
        
        # 逆変形データ適用
        apply_row.operator("rbf.apply_inverse_field_data", text="逆変形データ適用")
        
        # 区切り線
        layout.separator()
        
        # フィールド可視化セクション
        box = layout.box()
        box.label(text="フィールド可視化", icon='MESH_ICOSPHERE')
        
        col = box.column(align=True)
        col.prop(scene, "rbf_field_step")
        col.prop(scene, "rbf_field_use_inverse")
        col.prop(scene, "rbf_field_object_name")
        
        # フィールド可視化ボタン
        field_row = box.row()
        field_row.scale_y = 1.2
        
        # 現在の設定で作成されるファイル名を表示
        warning_msg = ""
        if not scene.rbf_source_avatar_name:
            warning_msg = "ソースアバター名を設定してください"
        elif scene.rbf_save_shape_key_mode and not scene.rbf_source_shape_key:
            warning_msg = "シェイプキー変形モードではシェイプキー名を選択してください"
        elif not scene.rbf_save_shape_key_mode and not scene.rbf_target_avatar_name:
            warning_msg = "アバター間変形モードではターゲットアバター名を設定してください"
        
        if warning_msg:
            box.label(text=warning_msg, icon='ERROR')
        else:
            field_row.operator("rbf.create_field_visualization", text="フィールドを可視化", icon='MESH_ICOSPHERE')
            
            # 対象ファイル名を表示
            row = box.row()
            inverse_suffix = "_inv" if scene.rbf_field_use_inverse else ""
            if scene.rbf_save_shape_key_mode:
                target_filename = f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}{inverse_suffix}.npz"
            else:
                target_filename = f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}{inverse_suffix}.npz"
            row.label(text=f"対象: {target_filename}", icon='FILE')
            
            row = box.row()
            direction_text = "逆変換" if scene.rbf_field_use_inverse else "通常変換"
            row.label(text=f"設定: {direction_text}、ステップ{scene.rbf_field_step}", icon='INFO')
        
        # 区切り線
        layout.separator()
        
        # NumPy・SciPy再インストールセクション（常に表示）
        box = layout.box()
        box.label(text="NumPy・SciPy マルチスレッド対応", icon='LIBRARY_DATA_DIRECT')
        
        # 現在のnumpyとscipyのバージョンを表示
        col = box.column(align=True)
        try:
            import numpy as np
            numpy_version = np.__version__
            col.label(text=f"現在のNumPy: {numpy_version}", icon='CHECKMARK')
        except ImportError:
            col.label(text="NumPy が見つかりません", icon='ERROR')
        
        try:
            import scipy
            scipy_version = scipy.__version__
            col.label(text=f"現在のSciPy: {scipy_version}", icon='CHECKMARK')
        except ImportError:
            col.label(text="SciPy が見つかりません（新規インストールされます）", icon='INFO')
        
        row = col.row()
        row.scale_y = 1.2
        row.operator("rbf.reinstall_numpy_scipy_multithreaded", text="NumPy・SciPy・psutil 再インストール", icon='FILE_REFRESH')
        
        # 区切り線
        layout.separator()
        
        # デバッグセクション
        box = layout.box()
        box.label(text="デバッグ・トラブルシューティング", icon='CONSOLE')
        
        row = box.row()
        row.prop(scene, "rbf_show_debug_info")
        
        if scene.rbf_show_debug_info:
            col = box.column(align=True)
            col.label(text="Pythonパス診断:", icon='INFO')
            
            row = col.row(align=True)
            row.operator("rbf.debug_show_python_paths", text="パス情報表示")
            row.operator("rbf.debug_test_external_python", text="外部Python テスト")
            
            col.separator()
            col.label(text="トラブルシューティング:")
            col.label(text="• importエラーが出る場合は上記テストを実行")
            col.label(text="• パス情報をコンソールで確認")
            col.label(text="• rbf_multithread_processor.pyを同じフォルダに配置")
            col.label(text="• NumPy・SciPy再インストールでマルチスレッド対応版を利用")


# シェイプキー選択用のオペレーター
class SELECT_OT_RBFShapeKey(bpy.types.Operator):
    bl_idname = "object.select_rbf_shape_key"
    bl_label = "Select Shape Key"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        scene = context.scene

        if scene.rbf_source_obj and scene.rbf_source_obj.data.shape_keys:
            # シェイプキーのリストを作成
            type(self).shape_keys = [key.name for key in scene.rbf_source_obj.data.shape_keys.key_blocks if key.name != "Basis"]

            if self.shape_keys:
                return context.window_manager.invoke_popup(self, width=200)

        return {'CANCELLED'}

    def draw(self, context):
        layout = self.layout
        for key_name in type(self).shape_keys:
            op = layout.operator("object.set_rbf_shape_key", text=key_name)
            op.shape_key_name = key_name


# シェイプキー設定用のオペレーター
class SET_OT_RBFShapeKey(bpy.types.Operator):
    bl_idname = "object.set_rbf_shape_key"
    bl_label = "Set Shape Key"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    shape_key_name: bpy.props.StringProperty()
    
    def execute(self, context):
        context.scene.rbf_source_shape_key = self.shape_key_name
        return {'FINISHED'}


# RBF変形生成オペレーター
class CREATE_OT_RBFDeformation(bpy.types.Operator, ExportHelper):
    bl_idname = "object.create_rbf_deformation"
    bl_label = "Save Deformation Data"
    bl_options = {'REGISTER', 'UNDO'}

    # ExportHelperのプロパティ
    filename_ext = ".npz"
    filter_glob: bpy.props.StringProperty(
        default="*.npz",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def execute(self, context):
        # オブジェクトモードに切り替え
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        
        # 必要なパラメータを取得
        source_obj = scene.rbf_source_obj
        source_shape_key_name = scene.rbf_source_shape_key
        selected_only = scene.rbf_selected_only
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        keep_first_field = scene.rbf_keep_first_field
        epsilon = scene.rbf_epsilon
        num_steps = scene.rbf_num_steps
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        add_normal_control_points = scene.rbf_add_normal_control_points
        normal_distance = scene.rbf_normal_distance
        shape_key_start_value = scene.rbf_shape_key_start_value
        shape_key_end_value = scene.rbf_shape_key_end_value
        
        # アバター名の検証
        if not source_avatar_name or not target_avatar_name:
            self.report({'ERROR'}, "Please set avatar name")
            return {'CANCELLED'}
        
        # シェイプキーの値の範囲検証
        if shape_key_start_value == shape_key_end_value:
            self.report({'ERROR'}, "Shape key start and end values must be different")
            return {'CANCELLED'}
        
        default_paths = []
        scene_folder = get_scene_folder()
        
        if scene.rbf_save_shape_key_mode:
            # シェイプキー変形モードの場合
            default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}.npz"))
            default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}_inv.npz"))
        else:
            # 通常のアバター間変形の場合
            default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}.npz"))
        
        try:
            # RBF補間を実行してフィールドを生成し、Deformation Fieldデータを保存
            field_objects, target_world_vertices, displacements = create_shape_key_from_rbf(
                source_obj, 
                source_shape_key_name, 
                selected_only,
                epsilon,
                num_steps,
                source_avatar_name,
                target_avatar_name,
                save_shape_key_mode,
                keep_first_field,
                add_normal_control_points,
                normal_distance,
                shape_key_start_value,
                shape_key_end_value
            )

            filelist = []
            if default_paths[0] and os.path.exists(default_paths[0]):
                if os.path.abspath(default_paths[0]) != os.path.abspath(self.filepath):
                    shutil.copy2(default_paths[0], self.filepath)
                filelist.append(self.filepath)
            if scene.rbf_save_shape_key_mode and default_paths[1] and os.path.exists(default_paths[1]):
                inv_filepath = self.filepath[:-4] + "_inv.npz"
                if os.path.abspath(default_paths[1]) != os.path.abspath(inv_filepath):
                    shutil.copy2(default_paths[1], inv_filepath)
                filelist.append(self.filepath[:-4] + "_inv.npz")

            self.report({'INFO'}, f"Deformation data saved: {', '.join(filelist)}")
            return {'FINISHED'}
        
        except Exception as e:
            error_msg = f"エラーが発生しました: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}
    
    def invoke(self, context, event):
        # デフォルトファイル名を設定
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        source_shape_key_name = scene.rbf_source_shape_key
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        
        filename = "deformation.npz"
        if source_avatar_name:
            if save_shape_key_mode:
                # シェイプキーモードの場合
                filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_shape_{source_shape_key_name}"
            elif target_avatar_name:
                # 通常の変形モードの場合
                filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}"
            self.filepath = filename + ".npz"

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# 保存したフィールドデータを適用するオペレーター
class APPLY_OT_FieldData(bpy.types.Operator):
    bl_idname = "rbf.apply_field_data"
    bl_label = "Apply Field Data"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # オブジェクトモードに切り替え
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name.strip()
        target_avatar_name = scene.rbf_target_avatar_name.strip()
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        source_shape_key_name = scene.rbf_source_shape_key
        
        if not source_avatar_name:
            self.report({'ERROR'}, "Please specify source avatar name")
            return {'CANCELLED'}
        
        # ファイルパスを現在の設定に基づいて生成（後方互換性対応）
        scene_folder = get_scene_folder()
        if save_shape_key_mode:
            # シェイプキー変形モードの場合
            if not source_shape_key_name:
                self.report({'ERROR'}, "Please specify shape key name in shape key mode")
                return {'CANCELLED'}
            display_name = "Shape key deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                source_shape_key_name=source_shape_key_name
            )
        else:
            # 通常のアバター間変形の場合
            if not target_avatar_name:
                self.report({'ERROR'}, "Please specify target avatar name")
                return {'CANCELLED'}
            display_name = "Inter-avatar deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                target_avatar_name=target_avatar_name
            )

        if not field_data_path:
            expected_filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_{f'shape_{source_shape_key_name}' if save_shape_key_mode else f'to_{normalize_avatar_name_for_filename(target_avatar_name)}'}.npz"
            self.report({'ERROR'}, f"{display_name} file not found: {expected_filename}")
            print(f"Deformation data file not found in: {scene_folder}")
            return {'CANCELLED'}
        
        try:
            target_obj = context.active_object
            if not target_obj or target_obj.type != 'MESH':  
                self.report({'ERROR'}, "Please select a Mesh object")
                return {'CANCELLED'}
            
            shape_key_name = scene.rbf_apply_shape_key_name if scene.rbf_apply_shape_key_name else "RBFDeform"
            apply_field_data(target_obj, field_data_path, shape_key_name)
            self.report({'INFO'}, f"{display_name} applied: {os.path.basename(field_data_path)}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# ベースポーズ差分保存オペレーター
class SAVE_OT_BasePoseDiff(bpy.types.Operator, ExportHelper):
    bl_idname = "object.save_base_pose_diff"
    bl_label = "Save Base Pose"
    bl_options = {'REGISTER', 'UNDO'}
    
    # ExportHelperのプロパティ
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def execute(self, context):
        # オブジェクトモードに切り替え
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        source_avatar_data_file = scene.rbf_source_avatar_data_file
        
        # アバター名の検証
        if not source_avatar_name:
            self.report({'ERROR'}, "Please set source avatar name")
            return {'CANCELLED'}
        
        if not source_avatar_data_file:
            self.report({'ERROR'}, "Please specify source avatar data file")
            return {'CANCELLED'}
        
        # アクティブオブジェクトを取得
        active_obj = context.active_object
        if not active_obj:
            self.report({'ERROR'}, "Please select an object")
            return {'CANCELLED'}
        
        # アクティブオブジェクトがArmatureかチェック
        if active_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an Armature object")
            return {'CANCELLED'}
        
        armature_obj = active_obj
        
        # 指定されたパスに保存
        filepath = self.filepath
        
        # アバターデータファイルのパスを絶対パスに変換
        avatar_data_filename = bpy.path.abspath(source_avatar_data_file)
        
        try:
            # 元のsave_armature_pose関数を使用するために、ファイル名を取得
            filename = os.path.basename(filepath)
            temp_dir = os.path.dirname(filepath)
            
            # 元の関数を使用してデータを保存
            saved_filepath = save_armature_pose(armature_obj, filename, avatar_data_filename)
            
            # 指定された場所に移動
            if saved_filepath != filepath and os.path.abspath(saved_filepath) != os.path.abspath(filepath):
                shutil.copy2(saved_filepath, filepath)

            self.report({'INFO'}, f"Base pose data saved: {filepath}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}
    
    def invoke(self, context, event):
        # デフォルトファイル名を設定
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        if source_avatar_name:
            self.filepath = f"pose_basis_{normalize_avatar_name_for_filename(source_avatar_name)}.json"
        else:
            self.filepath = "pose_basis.json"
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# ベースポーズ差分適用オペレーター
class APPLY_OT_BasePoseDiff(bpy.types.Operator):
    bl_idname = "object.apply_base_pose_diff"
    bl_label = "Apply Base Pose Difference"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # オブジェクトモードに切り替え
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        source_avatar_data_file = scene.rbf_source_avatar_data_file
        target_avatar_data_file = scene.rbf_target_avatar_data_file
        invert = scene.rbf_pose_invert
        
        # アバター名の検証
        if not source_avatar_name:
            self.report({'ERROR'}, "Please set source avatar name")
            return {'CANCELLED'}
        
        if not source_avatar_data_file:
            self.report({'ERROR'}, "Please specify source avatar data file")
            return {'CANCELLED'}
        
        # アクティブオブジェクトを取得
        active_obj = context.active_object
        if not active_obj:
            self.report({'ERROR'}, "Please select an object")
            return {'CANCELLED'}
        
        # アクティブオブジェクトがArmatureかチェック
        if active_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an Armature object")
            return {'CANCELLED'}
        
        armature_obj = active_obj
        
        # ファイル名を自動生成
        pose_filename = f"pose_basis_{normalize_avatar_name_for_filename(source_avatar_name)}.json"
        
        # アバターデータファイルのパスを絶対パスに変換
        if invert:
            avatar_data_filename = bpy.path.abspath(target_avatar_data_file)
        else:
            avatar_data_filename = bpy.path.abspath(source_avatar_data_file)
        
        try:
            add_pose_from_json(pose_filename, avatar_data_filename, invert)
            action = "inverse applied" if invert else "applied"
            self.report({'INFO'}, f"Base pose data {action}: {pose_filename}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# ポーズ保存オペレーター
class SAVE_OT_PoseDiff(bpy.types.Operator, ExportHelper):
    bl_idname = "object.save_pose_diff"
    bl_label = "Save Pose Difference"
    bl_options = {'REGISTER', 'UNDO'}
    
    # ExportHelperのプロパティ
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def execute(self, context):
        # オブジェクトモードに切り替え
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        source_avatar_data_file = scene.rbf_source_avatar_data_file
        
        # アバター名の検証
        if not source_avatar_name or not target_avatar_name:
            self.report({'ERROR'}, "Please set avatar name")
            return {'CANCELLED'}
        
        if not source_avatar_data_file:
            self.report({'ERROR'}, "Please specify source avatar data file")
            return {'CANCELLED'}
        
        # アクティブオブジェクトを取得
        active_obj = context.active_object
        if not active_obj:
            self.report({'ERROR'}, "Please select an object")
            return {'CANCELLED'}
        
        # アクティブオブジェクトがArmatureかチェック
        if active_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an Armature object")
            return {'CANCELLED'}
        
        armature_obj = active_obj
        
        # 指定されたパスに保存
        filepath = self.filepath
        
        # アバターデータファイルのパスを絶対パスに変換
        avatar_data_filename = bpy.path.abspath(source_avatar_data_file)
        
        try:
            # 元のsave_armature_pose関数を使用するために、ファイル名を取得
            filename = os.path.basename(filepath)
            temp_dir = os.path.dirname(filepath)
            
            # 元の関数を使用してデータを保存
            saved_filepath = save_armature_pose(armature_obj, filename, avatar_data_filename)
            
            # 指定された場所に移動
            if saved_filepath != filepath and os.path.abspath(saved_filepath) != os.path.abspath(filepath):
                shutil.copy2(saved_filepath, filepath)

            self.report({'INFO'}, f"Pose data saved: {filepath}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}

    def invoke(self, context, event):
        # デフォルトファイル名を設定
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        if source_avatar_name and target_avatar_name:
            self.filepath = f"posediff_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}.json"
        else:
            self.filepath = "posediff.json"
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# ポーズ適用オペレーター
class APPLY_OT_PoseDiff(bpy.types.Operator):
    bl_idname = "object.apply_pose_diff"
    bl_label = "Apply Pose Difference"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # オブジェクトモードに切り替え
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        source_avatar_data_file = scene.rbf_source_avatar_data_file
        target_avatar_data_file = scene.rbf_target_avatar_data_file
        invert = scene.rbf_pose_invert
        
        # アバター名の検証
        if not source_avatar_name or not target_avatar_name:
            self.report({'ERROR'}, "Please set avatar name")
            return {'CANCELLED'}
        
        if not source_avatar_data_file:
            self.report({'ERROR'}, "Please specify source avatar data file")
            return {'CANCELLED'}
        
        # アクティブオブジェクトを取得
        active_obj = context.active_object
        if not active_obj:
            self.report({'ERROR'}, "Please select an object")
            return {'CANCELLED'}
        
        # アクティブオブジェクトがArmatureかチェック
        if active_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an Armature object")
            return {'CANCELLED'}
        
        armature_obj = active_obj
        
        # ファイル名を自動生成
        pose_filename = f"posediff_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}.json"
        
        # アバターデータファイルのパスを絶対パスに変換
        if invert:
            avatar_data_filename = bpy.path.abspath(target_avatar_data_file)
        else:
            avatar_data_filename = bpy.path.abspath(source_avatar_data_file)
        
        try:
            add_pose_from_json(pose_filename, avatar_data_filename, invert)
            action = "inverse applied" if invert else "applied"
            self.report({'INFO'}, f"Pose data {action}: {pose_filename}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# アバター設定入れ替えオペレーター
class SWAP_OT_AvatarSettings(bpy.types.Operator):
    bl_idname = "object.swap_avatar_settings"
    bl_label = "Swap Avatar Settings"
    bl_description = "ソースとターゲットのアバター設定を入れ替え"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # 現在の値を取得
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        source_avatar_data_file = scene.rbf_source_avatar_data_file
        target_avatar_data_file = scene.rbf_target_avatar_data_file
        
        # 値を入れ替え
        scene.rbf_source_avatar_name = target_avatar_name
        scene.rbf_target_avatar_name = source_avatar_name
        scene.rbf_source_avatar_data_file = target_avatar_data_file
        scene.rbf_target_avatar_data_file = source_avatar_data_file
        
        self.report({'INFO'}, "Avatar settings swapped")
        return {'FINISHED'}


class SET_OT_HumanoidBoneInheritScale(bpy.types.Operator):
    bl_idname = "object.set_humanoid_bone_inherit_scale"
    bl_label = "Set Humanoid Bone Inherit Scale"
    bl_description = "選択されたArmatureのHumanoidボーンのInherit ScaleをAverageに設定"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # アクティブオブジェクトがArmatureかチェック
        if not context.active_object or context.active_object.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an Armature object")
            return {'CANCELLED'}
        
        armature_obj = context.active_object
        
        # ソースアバターデータファイルが設定されているかチェック
        if not scene.rbf_source_avatar_data_file:
            self.report({'ERROR'}, "Please set source avatar data file")
            return {'CANCELLED'}
        
        try:
            # アバターデータを読み込み
            avatar_data = load_avatar_data(scene.rbf_source_avatar_data_file)
            
            # Humanoidボーンの情報を取得
            bone_parents, humanoid_to_bone, bone_to_humanoid = get_humanoid_bone_hierarchy(avatar_data)
            
            # EditModeに切り替え
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode='EDIT')
            
            modified_count = 0
            
            # 各Humanoidボーンに対してInherit Scaleを設定
            for humanoid_bone_name, bone_name in humanoid_to_bone.items():
                if bone_name in armature_obj.data.edit_bones:
                    edit_bone = armature_obj.data.edit_bones[bone_name]
                    
                    # Inherit ScaleがNone以外の場合のみ設定
                    if edit_bone.inherit_scale != 'NONE':
                        # UpperChest、胸、つま先、足の指のヒューマノイドボーンはFullに設定
                        if 'Breast' in humanoid_bone_name or 'UpperChest' in humanoid_bone_name or 'Toe' in humanoid_bone_name or ('Foot' in humanoid_bone_name and ('Index' in humanoid_bone_name or 'Little' in humanoid_bone_name or 'Middle' in humanoid_bone_name or 'Ring' in humanoid_bone_name or 'Thumb' in humanoid_bone_name)):
                            edit_bone.inherit_scale = 'FULL'
                        else:
                            edit_bone.inherit_scale = 'AVERAGE'
                        modified_count += 1
            
            # ObjectModeに戻る
            bpy.ops.object.mode_set(mode='OBJECT')
            
            if modified_count > 0:
                self.report({'INFO'}, f"Set Inherit Scale to Average for {modified_count} Humanoid bones")
            else:
                self.report({'INFO'}, "No bones needed modification")

            return {'FINISHED'}

        except Exception as e:
            # エラー時はObjectModeに戻る
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except:
                pass
            self.report({'ERROR'}, f"An error occurred: {str(e)}")
            return {'CANCELLED'}


# 登録関数
def register():
    bpy.utils.register_class(RBF_PT_DeformationPanel)
    bpy.utils.register_class(SELECT_OT_RBFShapeKey)
    bpy.utils.register_class(SET_OT_RBFShapeKey)
    bpy.utils.register_class(CREATE_OT_RBFDeformation)
    bpy.utils.register_class(EXPORT_OT_RBFTempData)
    bpy.utils.register_class(APPLY_OT_FieldData)
    bpy.utils.register_class(APPLY_OT_InverseFieldData)
    bpy.utils.register_class(CREATE_OT_FieldVisualization)
    bpy.utils.register_class(SAVE_OT_BasePoseDiff)
    bpy.utils.register_class(APPLY_OT_BasePoseDiff)
    bpy.utils.register_class(SAVE_OT_PoseDiff)
    bpy.utils.register_class(APPLY_OT_PoseDiff)
    bpy.utils.register_class(SWAP_OT_AvatarSettings)
    bpy.utils.register_class(SET_OT_HumanoidBoneInheritScale)
    bpy.utils.register_class(DEBUG_OT_ShowPythonPaths)
    bpy.utils.register_class(DEBUG_OT_TestExternalPython)
    bpy.utils.register_class(REINSTALL_OT_NumpyScipyMultithreaded)
    register_properties()


# 登録解除関数
def unregister():
    bpy.utils.unregister_class(RBF_PT_DeformationPanel)
    bpy.utils.unregister_class(SELECT_OT_RBFShapeKey)
    bpy.utils.unregister_class(SET_OT_RBFShapeKey)
    bpy.utils.unregister_class(CREATE_OT_RBFDeformation)
    bpy.utils.unregister_class(EXPORT_OT_RBFTempData)
    bpy.utils.unregister_class(APPLY_OT_FieldData)
    bpy.utils.unregister_class(APPLY_OT_InverseFieldData)
    bpy.utils.unregister_class(CREATE_OT_FieldVisualization)
    bpy.utils.unregister_class(SAVE_OT_BasePoseDiff)
    bpy.utils.unregister_class(APPLY_OT_BasePoseDiff)
    bpy.utils.unregister_class(SAVE_OT_PoseDiff)
    bpy.utils.unregister_class(APPLY_OT_PoseDiff)
    bpy.utils.unregister_class(SWAP_OT_AvatarSettings)
    bpy.utils.unregister_class(SET_OT_HumanoidBoneInheritScale)
    bpy.utils.unregister_class(DEBUG_OT_ShowPythonPaths)
    bpy.utils.unregister_class(DEBUG_OT_TestExternalPython)
    bpy.utils.unregister_class(REINSTALL_OT_NumpyScipyMultithreaded)
    
    # プロパティの削除
    del bpy.types.Scene.rbf_source_obj
    del bpy.types.Scene.rbf_source_shape_key
    del bpy.types.Scene.rbf_selected_only
    del bpy.types.Scene.rbf_save_shape_key_mode
    del bpy.types.Scene.rbf_keep_first_field
    del bpy.types.Scene.rbf_epsilon
    del bpy.types.Scene.rbf_num_steps
    del bpy.types.Scene.rbf_apply_shape_key_name
    del bpy.types.Scene.rbf_base_grid_spacing
    del bpy.types.Scene.rbf_surface_distance
    del bpy.types.Scene.rbf_max_distance
    del bpy.types.Scene.rbf_min_distance
    del bpy.types.Scene.rbf_density_falloff
    del bpy.types.Scene.rbf_bbox_scale_factor
    del bpy.types.Scene.rbf_source_avatar_name
    del bpy.types.Scene.rbf_target_avatar_name
    del bpy.types.Scene.rbf_pose_invert
    del bpy.types.Scene.rbf_source_avatar_data_file
    del bpy.types.Scene.rbf_target_avatar_data_file
    del bpy.types.Scene.rbf_add_normal_control_points
    del bpy.types.Scene.rbf_normal_distance
    del bpy.types.Scene.rbf_show_debug_info
    del bpy.types.Scene.rbf_field_step
    del bpy.types.Scene.rbf_field_use_inverse
    del bpy.types.Scene.rbf_field_object_name


class APPLY_OT_InverseFieldData(bpy.types.Operator):
    """逆変形データを適用するオペレーター"""
    bl_idname = "rbf.apply_inverse_field_data"
    bl_label = "Apply Inverse Field Data"
    bl_description = "逆変形データを適用"
    
    def execute(self, context):
        # オブジェクトモードに切り替え
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name.strip()
        target_avatar_name = scene.rbf_target_avatar_name.strip()
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        source_shape_key_name = scene.rbf_source_shape_key
        
        if not source_avatar_name:
            self.report({'ERROR'}, "Please specify source avatar name")
            return {'CANCELLED'}
        
        # ファイルパスを現在の設定に基づいて生成（逆変形、後方互換性対応）
        scene_folder = get_scene_folder()
        if save_shape_key_mode:
            # シェイプキー変形モードの場合
            if not source_shape_key_name:
                self.report({'ERROR'}, "Please specify shape key name in shape key mode")
                return {'CANCELLED'}
            display_name = "Inverse shape key deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                source_shape_key_name=source_shape_key_name,
                inverse_suffix="_inv"
            )
        else:
            # 通常のアバター間変形の場合
            if not target_avatar_name:
                self.report({'ERROR'}, "Please specify target avatar name")
                return {'CANCELLED'}
            display_name = "Inverse inter-avatar deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                target_avatar_name=target_avatar_name,
                inverse_suffix="_inv"
            )

        if not field_data_path:
            expected_filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_{f'shape_{source_shape_key_name}' if save_shape_key_mode else f'to_{normalize_avatar_name_for_filename(target_avatar_name)}'}_inv.npz"
            self.report({'ERROR'}, f"{display_name} file not found: {expected_filename}")
            print(f"Inverse deformation data file not found in: {scene_folder}")
            return {'CANCELLED'}
        
        try:
            target_obj = context.active_object
            if not target_obj or target_obj.type != 'MESH':
                self.report({'ERROR'}, "Please select a Mesh object")
                return {'CANCELLED'}
            
            shape_key_name = scene.rbf_apply_shape_key_name if scene.rbf_apply_shape_key_name else "RBFDeform_inv"
            apply_field_data(target_obj, field_data_path, shape_key_name)
            self.report({'INFO'}, f"{display_name} applied: {os.path.basename(field_data_path)}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


def export_rbf_temp_data(source_obj, source_shape_key_name, selected_only=True, epsilon=0.0, num_steps=1, source_avatar_name="", target_avatar_name="", save_shape_key_mode=False, add_normal_control_points=False, normal_distance=-0.0002, shape_key_start_value=0.0, shape_key_end_value=1.0, enable_x_mirror=True):
    """
    RBF処理に必要な一時データをエクスポートする
    
    Parameters:
    - source_obj: ソースオブジェクト（シェイプキーを持つ）
    - source_shape_key_name: 使用するソースオブジェクトのシェイプキー名
    - selected_only: 選択された頂点のみを制御点として使用するか
    - epsilon: RBFパラメータ（0または負の値の場合は自動計算）
    - num_steps: 分割するステップ数
    - source_avatar_name: 変換元のアバター名
    - target_avatar_name: 変換先のアバター名
    - save_shape_key_mode: シェイプキー変形モード（通常と逆の両方向を保存）
    - add_normal_control_points: 制御点の法線方向に追加制御点を配置するか
    - normal_distance: 法線方向への距離（ワールド座標系）
    - shape_key_start_value: シェイプキーの開始値
    - shape_key_end_value: シェイプキーの終了値
    
    Returns:
    - 一時ファイルのパス
    """
    
    # 対象オブジェクトの表示状態を確認し、必要に応じて表示状態にする
    armature_obj = get_armature_from_source_object(source_obj)
    objects_to_check = [source_obj]
    if armature_obj:
        objects_to_check.append(armature_obj)
    
    original_visibility_states = ensure_objects_visible(objects_to_check)
    
    try:
        # 保存する方向のリスト（save_shape_key_modeに基づいて決定）
        if save_shape_key_mode:
            directions = [False, True]  # 通常の変形と逆変形の両方
        else:
            directions = [False]  # 通常の変形のみ
        
        results = []
        
        for invert in directions:
            direction_suffix = "_inv" if invert else ""
            print(f"\n=== Starting temporary data preparation for {'inverse' if invert else 'normal'} deformation ===")
            
            # 一時データの保存パスを自動生成
            scene_folder = get_scene_folder()
            
            if save_shape_key_mode:
                # シェイプキー変形モードの場合
                temp_data_path = os.path.join(scene_folder, f"temp_rbf_{normalize_avatar_name_for_filename(source_avatar_name)}_shape_{source_shape_key_name}{direction_suffix}.npz")
            else:
                # 通常のアバター間変形の場合
                temp_data_path = os.path.join(scene_folder, f"temp_rbf_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}{direction_suffix}.npz")
            
            # シェイプキーの値を保存
            original_values = {}
            for key in source_obj.data.shape_keys.key_blocks:
                original_values[key.name] = key.value
                key.value = 0.0
            
            # Invertオプションに応じて初期状態を設定
            if invert:
                # シェイプキーの終了値を基準にする
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = shape_key_end_value
            else:
                # シェイプキーの開始値を基準にする
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = shape_key_start_value
            
            # シーンを更新
            bpy.context.view_layer.update()
            
            # 評価後のデプスグラフを取得
            depsgraph = bpy.context.evaluated_depsgraph_get()
            
            # ソースオブジェクトの評価後のオブジェクトを取得
            evaluated_source = source_obj.evaluated_get(depsgraph)
            
            # ソースオブジェクトのシェイプキーを取得
            if source_obj.data.shape_keys is None or source_shape_key_name not in source_obj.data.shape_keys.key_blocks:
                raise ValueError(f"シェイプキー '{source_shape_key_name}' がソースオブジェクトに見つかりません")
            
            # ソースオブジェクトのワールド行列を取得
            source_world_matrix = source_obj.matrix_world
            
            # 制御点として使用する頂点のインデックスを取得
            original_selected_vertices = []  # 元の選択頂点を記録
            if selected_only:
                # 編集モードでの選択を反映するために、オブジェクトモードに切り替える
                was_edit_mode = False
                if bpy.context.object == source_obj and bpy.context.object.mode == 'EDIT':
                    was_edit_mode = True
                    bpy.ops.object.mode_set(mode='OBJECT')
                
                # 選択された頂点があるかチェック
                original_selected_vertices = [i for i, v in enumerate(source_obj.data.vertices) if v.select]
                
                # 編集モードに戻す
                if was_edit_mode:
                    bpy.ops.object.mode_set(mode='EDIT')
                
                if len(original_selected_vertices) == 0:
                    raise ValueError("選択された頂点がありません。少なくとも1つの頂点を選択してください。")
                
                # 選択された頂点から計算されるスケールされたBounding Box内の全ての頂点を制御点として使用
                selected_indices = get_vertices_in_scaled_bbox(source_obj, bpy.context.scene.rbf_bbox_scale_factor)
                
                if len(selected_indices) < 4:
                    print(f"Warning: Very few control points ({len(selected_indices)}). Consider selecting more control points.")
            else:
                # すべての頂点を使用
                selected_indices = list(range(len(source_obj.data.vertices)))
            
            # シェイプキー値0と1での重複制御点を事前に特定
            print("Pre-checking for duplicate control points at shape key values 0 and 1...")
            overlapping_indices = identify_overlapping_control_points_for_shape_keys(
                source_obj, 
                source_shape_key_name, 
                selected_indices, 
                source_world_matrix, 
                add_normal_control_points, 
                normal_distance
            )
            
            # 各ステップでのフィールドデータと変形データを収集
            all_step_data = []
            all_field_world_vertices = []
            
            for step in range(num_steps):
                print(f"\n=== Collecting data for step {step+1}/{num_steps} ===")
                
                # 現在のステップの値を計算
                progress = (step + 1) / num_steps
                if invert:
                    # Invertモードでは終了値から開始値へ変化
                    step_value = shape_key_end_value - (shape_key_end_value - shape_key_start_value) * progress
                else:
                    # 通常モードでは開始値から終了値へ変化
                    step_value = shape_key_start_value + (shape_key_end_value - shape_key_start_value) * progress
                
                print(f"Shape key value: {step_value}")
                
                # 頂点グループに基づいて制御点をフィルタリング
                filtered_indices = filter_control_points_by_vertex_groups(source_obj, selected_indices, step_value)
                
                if len(filtered_indices) < 4:
                    print(f"Warning: Very few valid control points at step {step+1} ({len(filtered_indices)}).")
                    if len(filtered_indices) == 0:
                        print(f"Skipping step {step+1}: No valid control points.")
                        continue
                
                print(f"Control points: {len(selected_indices)} -> {len(filtered_indices)} (after vertex group filtering)")
                
                # 変形前の状態を取得（バウンディングボックス計算用）
                current_basis_local = np.array([evaluated_source.data.vertices[i].co.copy() for i in filtered_indices])
                
                # 変form前の状態をワールド座標に変換
                current_basis = np.zeros_like(current_basis_local)
                for i, basis_co in enumerate(current_basis_local):
                    basis_v = Vector((basis_co[0], basis_co[1], basis_co[2], 1.0))
                    world_basis = source_world_matrix @ basis_v
                    current_basis[i] = np.array([world_basis[0], world_basis[1], world_basis[2]])
                
                # 現在のステップでのフィールドを生成（変形前のソースオブジェクトを使用）
                print(f"Generating Deformation Field for step {step+1}...")
                field_vertices = create_adaptive_deformation_field(
                    target_obj=source_obj,
                    base_grid_spacing=bpy.context.scene.rbf_base_grid_spacing,
                    surface_distance=bpy.context.scene.rbf_surface_distance,
                    max_distance=bpy.context.scene.rbf_max_distance,
                    min_distance=bpy.context.scene.rbf_min_distance,
                    density_falloff=bpy.context.scene.rbf_density_falloff,
                    bbox_scale_factor=bpy.context.scene.rbf_bbox_scale_factor,
                    use_selected_vertices=selected_only
                )
                
                if field_vertices is None:
                    print(f"Failed to generate field at step {step+1}")
                    continue
                
                # VectorオブジェクトをPython配列に変換（Pickle化可能にするため）
                field_vertices_array = np.array([[v.x, v.y, v.z] for v in field_vertices])
                all_field_world_vertices.append(field_vertices_array)
                
                # シェイプキーの値を更新して変形後の状態を取得
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = step_value
                
                # シーンを更新
                bpy.context.view_layer.update()
                
                # 評価後のオブジェクトを再取得
                depsgraph.update()
                evaluated_source_deformed = source_obj.evaluated_get(depsgraph)
                
                # 変形後の頂点位置を取得
                current_deformed_local = np.array([evaluated_source_deformed.data.vertices[i].co.copy() for i in filtered_indices])
                
                # 変形後の位置をワールド座標に変換
                current_deformed = np.zeros_like(current_deformed_local)
                for i, deformed_co in enumerate(current_deformed_local):
                    deformed_v = Vector((deformed_co[0], deformed_co[1], deformed_co[2], 1.0))
                    world_deformed = source_world_matrix @ deformed_v
                    current_deformed[i] = np.array([world_deformed[0], world_deformed[1], world_deformed[2]])
                
                # 法線方向に制御点を追加する場合
                if add_normal_control_points:
                    current_basis_extended, current_deformed_extended = add_normal_control_points_func(
                        source_obj, 
                        filtered_indices, 
                        current_basis, 
                        current_deformed, 
                        normal_distance
                    )
                else:
                    current_basis_extended = current_basis
                    current_deformed_extended = current_deformed
                
                # 事前に特定された重複制御点を除外
                selected_indices_updated = filtered_indices
                if len(overlapping_indices) > 0:
                    print(f"Excluding {len(overlapping_indices)} pre-identified duplicate control points")
                    # 重複していない制御点のインデックスを取得
                    all_indices = np.arange(len(current_basis_extended))
                    valid_indices = np.setdiff1d(all_indices, overlapping_indices)
                    
                    if len(valid_indices) < len(current_basis_extended):
                        # filtered_indicesも同様に更新する必要がある
                        if len(filtered_indices) == len(current_basis_extended):
                            selected_indices_updated = np.array(filtered_indices)[valid_indices].tolist()
                        else:
                            selected_indices_updated = filtered_indices
                        
                        current_basis_extended = current_basis_extended[valid_indices]
                        current_deformed_extended = current_deformed_extended[valid_indices]
                        print(f"Excluded duplicate control points: using {len(valid_indices)} control points")
                
                # ステップデータを保存
                step_data = {
                    'step_value': step_value,
                    'control_points_original': current_basis_extended,
                    'control_points_deformed': current_deformed_extended,
                    'selected_indices': selected_indices_updated
                }
                
                all_step_data.append(step_data)
                
                print(f"Step {step+1} data collection complete")
            
            # シェイプキーの値を元に戻す
            for key_name, value in original_values.items():
                source_obj.data.shape_keys.key_blocks[key_name].value = value
            
            # シーンを更新
            bpy.context.view_layer.update()
            
            # 一時データを保存
            temp_data = {
                'all_field_world_vertices': all_field_world_vertices if len(all_field_world_vertices) == 1 else np.array(all_field_world_vertices, dtype=object),  # 各ステップのフィールド座標
                'field_world_matrix': np.identity(4),
                'all_step_data': np.array(all_step_data, dtype=object),
                'source_world_matrix': np.array(source_world_matrix),
                'epsilon': epsilon,
                'num_steps': num_steps,
                'invert': invert,
                'source_avatar_name': source_avatar_name,
                'target_avatar_name': target_avatar_name,
                'source_shape_key_name': source_shape_key_name,
                'save_shape_key_mode': save_shape_key_mode,
                'add_normal_control_points': add_normal_control_points,
                'normal_distance': normal_distance,
                'shape_key_start_value': shape_key_start_value,  # シェイプキー開始値を追加
                'shape_key_end_value': shape_key_end_value,      # シェイプキー終了値を追加
                'original_selected_vertices': original_selected_vertices,  # 元の選択頂点を追加
                'selected_only': selected_only,  # selected_onlyフラグも追加
                'rbf_base_grid_spacing': bpy.context.scene.rbf_base_grid_spacing,
                'rbf_surface_distance': bpy.context.scene.rbf_surface_distance,
                'rbf_max_distance': bpy.context.scene.rbf_max_distance,
                'rbf_min_distance': bpy.context.scene.rbf_min_distance,
                'rbf_density_falloff': bpy.context.scene.rbf_density_falloff,
                'rbf_bbox_scale_factor': bpy.context.scene.rbf_bbox_scale_factor,
                'enable_x_mirror': enable_x_mirror
            }
            
            # numpy形式で保存
            np.savez(temp_data_path, **temp_data)
            
            print(f"Saved temporary data: {temp_data_path}")
            results.append(temp_data_path)
        
        return results
    
    finally:
        # 処理完了後、オブジェクトの表示状態を元に戻す
        restore_objects_visibility(objects_to_check, original_visibility_states)

# 一時データエクスポート用オペレーター
class EXPORT_OT_RBFTempData(bpy.types.Operator, ExportHelper):
    bl_idname = "object.export_rbf_temp_data"
    bl_label = "Save Deformation Data"
    bl_options = {'REGISTER', 'UNDO'}

    # ExportHelperのプロパティ
    filename_ext = ".npz"
    filter_glob: bpy.props.StringProperty(
        default="*.npz",
        options={'HIDDEN'},
        maxlen=255,
    )

    # Modal処理用のクラス変数
    _timer = None
    _thread = None
    _process = None
    _queue = None
    _progress = 0.0
    _status_message = ""
    _default_paths = None
    _save_shape_key_mode = False
    _dot_count = 0
    _temp_file_paths = None  # Phase 2: 一時ファイルパスを保持（キャンセル時クリーンアップ用）
    # Phase 3: 進捗UI強化用
    _current_phase = ""  # 現在のフェーズ名（距離計算/変形/フォールオフ）
    _progress_started = False  # window_manager.progress が開始されているか

    def modal(self, context, event):
        import queue as queue_module
        import re

        if event.type == 'TIMER':
            # キャンセル後に Queue が None になっている場合はスキップ
            if not self._queue:
                return {'PASS_THROUGH'}

            # アニメーション更新
            self._dot_count = (self._dot_count + 1) % 4
            dots = "." * (self._dot_count + 1)

            # Queue から非ブロッキングでログを取得
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item[0] == 'LOG':
                        line = item[1]
                        print(f"[RBF Processing] {line}")

                        # Phase 3: フェーズ検出
                        if '距離計算進捗' in line or '距離計算を' in line:
                            self._current_phase = "距離計算"
                        elif 'マルチプロセスRBF補間を開始' in line:
                            self._current_phase = "変形"
                        elif 'フォールオフ処理を適用中' in line:
                            self._current_phase = "フォールオフ"
                            self._progress = 95.0  # フォールオフは最終段階
                            # Phase 3: フォールオフ突入時にプログレスバーを即座に更新
                            if self._progress_started:
                                context.window_manager.progress_update(95)

                        # 進捗パース（例: "進捗: 1000/10000 頂点処理完了 (10.0%)"）
                        match = re.search(r'\((\d+\.?\d*)%\)', line)
                        if match:
                            raw_progress = float(match.group(1))
                            # Phase 3: フェーズに応じて全体進捗を計算
                            # 距離計算: 0-30%, 変形: 30-95%, フォールオフ: 95-100%
                            if self._current_phase == "距離計算":
                                self._progress = raw_progress * 0.30
                            elif self._current_phase == "変形":
                                self._progress = 30.0 + raw_progress * 0.65
                            else:
                                self._progress = raw_progress

                            # Phase 3: window_manager.progress を更新
                            if self._progress_started:
                                context.window_manager.progress_update(int(self._progress))

                        # ステータスメッセージを更新（最後の50文字）
                        self._status_message = line[-50:] if len(line) > 50 else line
                    elif item[0] == 'DONE':
                        returncode = item[1]
                        # Phase 3: 完了時に100%を表示
                        if self._progress_started:
                            self._progress = 100.0
                            context.window_manager.progress_update(100)
                        self._finish(context, returncode == 0)
                        return {'FINISHED'}
                    elif item[0] == 'ERROR':
                        error_msg = item[1]
                        self._finish_with_error(context, error_msg)
                        return {'CANCELLED'}
            except queue_module.Empty:
                pass

            # Phase 3: UI更新（フェーズ名を含む）
            if self._progress > 0:
                phase_str = f"[{self._current_phase}] " if self._current_phase else ""
                context.workspace.status_text_set(
                    f"RBF Processing{dots} {phase_str}{self._progress:.1f}%"
                )
            else:
                context.workspace.status_text_set(f"RBF Processing{dots}")

        elif event.type == 'ESC':
            self._cancel_process(context)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        import threading
        import queue as queue_module

        # オブジェクトモードに切り替え
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        scene = context.scene

        # 必要なパラメータを取得
        source_obj = scene.rbf_source_obj
        source_shape_key_name = scene.rbf_source_shape_key
        selected_only = scene.rbf_selected_only
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        epsilon = scene.rbf_epsilon
        num_steps = scene.rbf_num_steps
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        add_normal_control_points = scene.rbf_add_normal_control_points
        normal_distance = scene.rbf_normal_distance
        shape_key_start_value = scene.rbf_shape_key_start_value
        shape_key_end_value = scene.rbf_shape_key_end_value
        enable_x_mirror = scene.rbf_enable_x_mirror

        # アバター名の検証
        if not source_avatar_name or not target_avatar_name:
            self.report({'ERROR'}, "Please set avatar name")
            return {'CANCELLED'}

        # シェイプキーの値の範囲検証
        if shape_key_start_value == shape_key_end_value:
            self.report({'ERROR'}, "Shape key start and end values must be different")
            return {'CANCELLED'}


        self._default_paths = []
        scene_folder = get_scene_folder()
        self._save_shape_key_mode = save_shape_key_mode

        if scene.rbf_save_shape_key_mode:
            # シェイプキー変形モードの場合
            self._default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}.npz"))
            self._default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}_inv.npz"))
        else:
            # 通常のアバター間変形の場合
            self._default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}.npz"))

        try:
            # 一時データをエクスポート（同期処理 - 高速なため問題なし）
            self._temp_file_paths = export_rbf_temp_data(
                source_obj,
                source_shape_key_name,
                selected_only,
                epsilon,
                num_steps,
                source_avatar_name,
                target_avatar_name,
                save_shape_key_mode,
                add_normal_control_points,
                normal_distance,
                shape_key_start_value,
                shape_key_end_value,
                enable_x_mirror
            )

            # 保存されたファイルの情報を生成
            file_list = ", ".join([os.path.basename(path) for path in self._temp_file_paths])
            self.report({'INFO'}, f"Temporary data exported: {file_list}")

            base_temp_path = self._temp_file_paths[0]

            print(f"\n{'='*60}")
            print(f"RBF processing started: {os.path.basename(base_temp_path)}")
            print(f"{'='*60}")

            # Queue 初期化
            self._queue = queue_module.Queue()
            self._progress = 0.0
            self._status_message = ""
            self._dot_count = 0
            # Phase 3: 進捗UI強化用の初期化
            self._current_phase = ""
            self._progress_started = True
            context.window_manager.progress_begin(0, 100)

            # メインスレッドで事前に取得する必要がある値
            python_path = get_blender_python_path()
            processor_path = get_rbf_processor_script_path()
            blender_lib_paths = get_blender_python_lib_paths()
            user_site_packages = get_blender_python_user_site_packages(python_path)
            blender_deps_path = os.path.join(os.path.dirname(__file__), 'deps')
            filepath = self.filepath  # ExportHelperで設定されたファイルパス

            # バックグラウンドスレッドで実行する関数
            def run_rbf_background():
                # キャンセル時の競合を避けるため、Queue をローカル参照で保持
                # （_cleanup() で self._queue = None になっても安全に動作）
                q = self._queue

                try:
                    # パスの存在確認
                    if not os.path.exists(python_path):
                        if q:
                            q.put(('ERROR', f"Python binary not found: {python_path}"))
                        return

                    if not os.path.exists(processor_path):
                        if q:
                            q.put(('ERROR', f"RBF processor script not found: {processor_path}"))
                        return

                    # 環境変数を設定
                    env = os.environ.copy()
                    env['PYTHONIOENCODING'] = 'utf-8'
                    env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
                    env['PYTHONUNBUFFERED'] = '1'

                    # PYTHONPATHを設定
                    pythonpath_parts = []
                    if 'PYTHONPATH' in env:
                        pythonpath_parts.append(env['PYTHONPATH'])
                    if blender_deps_path:
                        pythonpath_parts.append(blender_deps_path)
                    if user_site_packages:
                        pythonpath_parts.append(user_site_packages)
                    pythonpath_parts.extend(blender_lib_paths)
                    env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)

                    # Phase 1-B: BLAS スレッド数を固定値に制限（オーバーサブスクライブ防止）
                    env['OMP_NUM_THREADS'] = '2'
                    env['OPENBLAS_NUM_THREADS'] = '2'
                    env['MKL_NUM_THREADS'] = '2'
                    env['VECLIB_MAXIMUM_THREADS'] = '2'
                    env['NUMEXPR_NUM_THREADS'] = '2'

                    # Phase 1-B: ワーカー数を制限
                    max_workers = min(4, os.cpu_count() or 4)

                    # コマンドを構築
                    cmd = [python_path, '-u', processor_path, base_temp_path,
                           '--max-workers', str(max_workers)]

                    print(f"Executing command: {' '.join(cmd)}")
                    print(f"max_workers: {max_workers}, OMP_NUM_THREADS: 2")

                    # プロセスを起動
                    self._process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        cwd=os.path.dirname(base_temp_path),
                        env=env,
                        bufsize=1,
                        universal_newlines=True
                    )

                    # stdout を読み取り、Queue に送信
                    for line in iter(self._process.stdout.readline, ''):
                        if line and q:
                            q.put(('LOG', line.rstrip('\n\r')))

                    # プロセス完了を待つ
                    self._process.wait()

                    # 成功時はファイルをコピー
                    if self._process.returncode == 0:
                        if self._default_paths[0] and os.path.exists(self._default_paths[0]):
                            if os.path.abspath(self._default_paths[0]) != os.path.abspath(filepath):
                                shutil.copy2(self._default_paths[0], filepath)
                        if self._save_shape_key_mode and len(self._default_paths) > 1 and self._default_paths[1] and os.path.exists(self._default_paths[1]):
                            inv_filepath = filepath[:-4] + "_inv.npz"
                            if os.path.abspath(self._default_paths[1]) != os.path.abspath(inv_filepath):
                                shutil.copy2(self._default_paths[1], inv_filepath)

                    if q:
                        q.put(('DONE', self._process.returncode))

                except Exception as e:
                    error_msg = f"Error during RBF processing: {str(e)}"
                    print(error_msg)
                    print(traceback.format_exc())
                    if q:
                        q.put(('ERROR', error_msg))

            # バックグラウンドスレッドを起動
            self._thread = threading.Thread(target=run_rbf_background)
            self._thread.start()

            # タイマーを設定（0.1秒ごとにチェック）
            self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
            context.window_manager.modal_handler_add(self)

            # ステータスバーに表示開始
            context.workspace.status_text_set("Starting RBF processing...")

            self.report({'INFO'}, "Multiprocess processing started (running in background)")

            return {'RUNNING_MODAL'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            # Phase 3: 例外時もプログレスバーを確実に終了
            if self._progress_started:
                context.window_manager.progress_end()
                self._progress_started = False
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}

    def _finish(self, context, success):
        """処理完了時の処理"""
        # 正常終了時は一時ファイルを削除せず、参照のみクリア
        self._temp_file_paths = None
        self._cleanup(context)

        if success:
            self.report({'INFO'}, "RBF processing completed successfully")
            print("RBF processing completed successfully")

            # 成功ポップアップを表示
            def draw_success_popup(self, context):
                self.layout.label(text="Deformation Field generation completed")

            context.window_manager.popup_menu(draw_success_popup, title="Complete", icon='CHECKMARK')
        else:
            self.report({'ERROR'}, "RBF processing failed")
            print("RBF processing failed")

        # UIを更新
        for area in context.screen.areas:
            area.tag_redraw()

    def _finish_with_error(self, context, error_msg):
        """エラー終了時の処理"""
        # エラー終了時も参照のみクリア（一時ファイルはデバッグ用に残す）
        self._temp_file_paths = None
        self._cleanup(context)
        self.report({'ERROR'}, error_msg)
        print(f"RBF processing error: {error_msg}")

        # UIを更新
        for area in context.screen.areas:
            area.tag_redraw()

    def _cancel_process(self, context):
        """処理をキャンセル"""
        import sys

        if self._process:
            try:
                pid = self._process.pid
                print(f"Cancelling RBF processing (PID: {pid})...")

                # Phase 2: Windows では taskkill を使用して子プロセスも含めて終了
                if sys.platform == 'win32':
                    try:
                        # /T: 子プロセスも終了, /F: 強制終了
                        kill_cmd = ['taskkill', '/T', '/F', '/PID', str(pid)]
                        result = subprocess.run(kill_cmd, capture_output=True, timeout=5)
                        if result.returncode == 0:
                            print(f"Terminated including child processes with taskkill (PID: {pid})")
                        else:
                            # taskkill が失敗した場合（プロセスが既に終了している等）
                            stderr_msg = result.stderr.decode('utf-8', errors='replace').strip()
                            print(f"taskkill exited with returncode={result.returncode}: {stderr_msg}")
                            print("Trying terminate()...")
                            self._process.terminate()
                    except subprocess.TimeoutExpired:
                        print("taskkill timed out. Trying terminate()...")
                        self._process.terminate()
                    except Exception as e:
                        print(f"Error with taskkill: {e}. Trying terminate()...")
                        self._process.terminate()
                else:
                    # Unix 系では terminate() を使用
                    self._process.terminate()

                print("RBF processing cancelled")
            except Exception as e:
                print(f"Error while terminating process: {e}")

        # Phase 2: 一時ファイルをクリーンアップ
        self._cleanup_temp_files()

        self._cleanup(context)
        self.report({'WARNING'}, "RBF processing was cancelled")

    def _cleanup_temp_files(self):
        """一時ファイルをクリーンアップ（キャンセル時のみ）"""
        if self._temp_file_paths:
            for temp_path in self._temp_file_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        print(f"Deleted temporary file: {os.path.basename(temp_path)}")
                except Exception as e:
                    print(f"Error deleting temporary file ({os.path.basename(temp_path)}): {e}")
            self._temp_file_paths = None

    def _cleanup(self, context):
        """リソースのクリーンアップ"""
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)
        # Phase 3: プログレスバーを終了
        if self._progress_started:
            context.window_manager.progress_end()
            self._progress_started = False
        self._current_phase = ""
        self._process = None
        self._thread = None
        self._queue = None

    def cancel(self, context):
        """キャンセル時の処理（Blenderから呼ばれる）"""
        self._cancel_process(context)

    def invoke(self, context, event):
        # デフォルトファイル名を設定
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        source_shape_key_name = scene.rbf_source_shape_key
        save_shape_key_mode = scene.rbf_save_shape_key_mode

        filename = "deformation.npz"
        if source_avatar_name:
            if save_shape_key_mode:
                # シェイプキーモードの場合
                filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_shape_{source_shape_key_name}"
            elif target_avatar_name:
                # 通常の変形モードの場合
                filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}"
            self.filepath = filename + ".npz"

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


def safe_decode(data):
    """
    バイナリデータを安全にテキストにデコードする。
    Windows のコンソール出力で発生する UnicodeDecodeError を回避するため、
    UTF-8 → CP932 → 置換モード UTF-8 の順でフォールバックする。

    Parameters:
        data (bytes): デコードするバイナリデータ

    Returns:
        str: デコードされたテキスト
    """
    if not data:
        return ""
    # まず UTF-8 を試す
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass
    # 次に CP932 (Shift-JIS) を試す
    try:
        return data.decode('cp932')
    except UnicodeDecodeError:
        pass
    # 最後に置換モードで UTF-8
    return data.decode('utf-8', errors='replace')


def run_subprocess_safe(cmd, env=None, timeout=None, cwd=None):
    """
    UnicodeDecodeError を回避して subprocess を実行する。
    Windows 環境でテキストモードを使わず、バイナリモードで実行して
    手動でデコードする。

    Parameters:
        cmd (list): 実行するコマンド
        env (dict, optional): 環境変数
        timeout (int, optional): タイムアウト秒数
        cwd (str, optional): 作業ディレクトリ

    Returns:
        tuple: (returncode, stdout_text, stderr_text)
    """
    creationflags = 0
    if platform.system() == "Windows":
        creationflags = subprocess.CREATE_NO_WINDOW

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        creationflags=creationflags
    )

    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_bytes, stderr_bytes = process.communicate()
        return -1, "", "Timeout"

    stdout_text = safe_decode(stdout_bytes)
    stderr_text = safe_decode(stderr_bytes)

    return process.returncode, stdout_text, stderr_text


def get_blender_python_path():
    """
    Blenderに含まれるPythonバイナリのパスを取得する
    
    Returns:
        str: Pythonバイナリのパス
    """
    # Blenderの実行可能ファイルのパス
    blender_binary = bpy.app.binary_path
    blender_dir = os.path.dirname(blender_binary)
    
    # OS別のPythonバイナリパスを構築
    system = platform.system()
    
    if system == "Windows":
        # Windows: Blender/{version}/python/bin/python.exe
        version = bpy.app.version_string[:4]  # "4.2"形式
        python_path = os.path.join(blender_dir, version, "python", "bin", "python.exe")
        
        # バックアップパス（別のディレクトリ構造の場合）
        if not os.path.exists(python_path):
            python_path = os.path.join(blender_dir, "python", "bin", "python.exe")
            
        # それでも見つからない場合は、Blenderと同じディレクトリを探す
        if not os.path.exists(python_path):
            python_path = os.path.join(blender_dir, "python.exe")
            
    elif system == "Darwin":  # macOS
        # macOS: Blender.app/Contents/Resources/{version}/python/bin/python
        version = bpy.app.version_string[:4]
        python_path = os.path.join(blender_dir, "..", "Resources", version, "python", "bin", "python")
        
        # バックアップパス
        if not os.path.exists(python_path):
            python_path = os.path.join(blender_dir, "..", "Resources", "python", "bin", "python")
            
    else:  # Linux
        # Linux: blender/{version}/python/bin/python
        version = bpy.app.version_string[:4]
        python_path = os.path.join(blender_dir, version, "python", "bin", "python")
        
        # バックアップパス
        if not os.path.exists(python_path):
            python_path = os.path.join(blender_dir, "python", "bin", "python")
    
    # パスの正規化
    python_path = os.path.abspath(python_path)
    
    return python_path


def get_rbf_processor_script_path():
    """
    rbf_multithread_processor.pyスクリプトのパスを取得する
    
    Returns:
        str: rbf_multithread_processor.pyのパス
    """
    # このスクリプトと同じディレクトリにあると仮定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processor_path = os.path.join(current_dir, "rbf_multithread_processor.py")
    
    # 見つからない場合は、Blenderファイルと同じディレクトリを確認
    if not os.path.exists(processor_path):
        scene_folder = get_scene_folder()
        processor_path = os.path.join(scene_folder, "rbf_multithread_processor.py")
    
    return processor_path


def get_blender_python_user_site_packages(python_path=None):
    """
    BlenderのPythonで--userインストールされたパッケージの場所を取得する
    
    Parameters:
        python_path (str, optional): Pythonバイナリのパス
    
    Returns:
        str: ユーザーサイトパッケージのパス、見つからない場合はNone
    """
    try:
        if python_path is None:
            python_path = get_blender_python_path()
        
        if not os.path.exists(python_path):
            return None
            
        # Pythonでユーザーサイトパッケージディレクトリを取得
        cmd = [python_path, '-c', 'import site; print(site.getusersitepackages())']

        try:
            returncode, stdout, stderr = run_subprocess_safe(cmd, timeout=10)

            if returncode == 0:
                user_site_path = stdout.strip()
                if user_site_path and os.path.exists(user_site_path):
                    return user_site_path
        except:
            pass

        # フォールバック: 手動でパスを構築
        if platform.system() == "Windows":
            # Pythonのバージョンを取得
            try:
                version_cmd = [python_path, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")']
                returncode, stdout, stderr = run_subprocess_safe(version_cmd, timeout=5)

                if returncode == 0:
                    python_version = stdout.strip()
                    # Windows: %APPDATA%\Python\PythonXX\site-packages
                    appdata = os.environ.get('APPDATA', '')
                    if appdata:
                        user_site_path = os.path.join(appdata, 'Python', f'Python{python_version.replace(".", "")}', 'site-packages')
                        if os.path.exists(user_site_path):
                            return user_site_path
            except:
                pass
                
    except Exception as e:
        print(f"Failed to get user site packages: {e}")
        
    return None


def get_blender_python_lib_paths():
    """
    BlenderのPythonライブラリパスを取得する
    
    Returns:
        list: Pythonライブラリパスのリスト
    """
    import site
    
    # Blenderの実行可能ファイルのパス
    blender_binary = bpy.app.binary_path
    blender_dir = os.path.dirname(blender_binary)
    
    # OS別のPythonライブラリパスを構築
    system = platform.system()
    lib_paths = []
    
    # Blenderのバージョンを取得
    version = bpy.app.version_string[:3]  # "4.0"形式
    
    # ユーザーディレクトリのBlenderパスを取得
    def get_user_blender_path():
        if system == "Windows":
            appdata = os.environ.get('APPDATA', '')
            if appdata:
                return os.path.join(appdata, "Blender Foundation", "Blender", version)
        elif system == "Darwin":  # macOS
            home = os.path.expanduser("~")
            return os.path.join(home, "Library", "Application Support", "Blender", version)
        else:  # Linux
            home = os.path.expanduser("~")
            return os.path.join(home, ".config", "blender", version)
        return None
    
    user_blender_path = get_user_blender_path()
    
    if system == "Windows":
        # Windows: 指定されたパス構造に対応
        lib_paths.extend([
            # scripts関連（Blenderインストールディレクトリ）
            os.path.join(blender_dir, version, "scripts", "startup"),
            os.path.join(blender_dir, version, "scripts", "modules"), 
            os.path.join(blender_dir, version, "scripts", "addons", "modules"),
            os.path.join(blender_dir, version, "scripts", "addons"),
            os.path.join(blender_dir, version, "scripts", "addons_contrib"),
            
            # python関連
            os.path.join(blender_dir, f"python{sys.version_info.major}{sys.version_info.minor}.zip"),
            os.path.join(blender_dir, version, "python", "DLLs"),
            os.path.join(blender_dir, version, "python", "lib"),
            os.path.join(blender_dir, version, "python", "bin"),
            os.path.join(blender_dir, version, "python"),
            os.path.join(blender_dir, version, "python", "lib", "site-packages"),
            
            # バックアップパス（バージョンなし）
            os.path.join(blender_dir, "scripts", "startup"),
            os.path.join(blender_dir, "scripts", "modules"),
            os.path.join(blender_dir, "scripts", "addons", "modules"),
            os.path.join(blender_dir, "scripts", "addons"),
            os.path.join(blender_dir, "scripts", "addons_contrib"),
            os.path.join(blender_dir, "python", "lib", "site-packages"),
            os.path.join(blender_dir, "python", "lib"),
            os.path.join(blender_dir, "python")
        ])
        
        # ユーザーディレクトリのBlenderパス（Windows）
        if user_blender_path:
            lib_paths.extend([
                os.path.join(user_blender_path, "scripts", "startup"),
                os.path.join(user_blender_path, "scripts", "modules"),
                os.path.join(user_blender_path, "scripts", "addons", "modules"),
                os.path.join(user_blender_path, "scripts", "addons"),
                os.path.join(user_blender_path, "scripts", "addons_contrib")
            ])
        
    elif system == "Darwin":  # macOS
        # macOS: Blender.app/Contents/Resources/{version}/
        lib_paths.extend([
            # scripts関連
            os.path.join(blender_dir, "..", "Resources", version, "scripts", "startup"),
            os.path.join(blender_dir, "..", "Resources", version, "scripts", "modules"),
            os.path.join(blender_dir, "..", "Resources", version, "scripts", "addons", "modules"),
            os.path.join(blender_dir, "..", "Resources", version, "scripts", "addons"),
            os.path.join(blender_dir, "..", "Resources", version, "scripts", "addons_contrib"),
            
            # python関連
            os.path.join(blender_dir, "..", "Resources", f"python{sys.version_info.major}{sys.version_info.minor}.zip"),
            os.path.join(blender_dir, "..", "Resources", version, "python", "lib", "python3.11", "site-packages"),
            os.path.join(blender_dir, "..", "Resources", version, "python", "lib"),
            os.path.join(blender_dir, "..", "Resources", version, "python", "bin"),
            os.path.join(blender_dir, "..", "Resources", version, "python"),
            
            # バックアップパス
            os.path.join(blender_dir, "..", "Resources", "scripts", "startup"),
            os.path.join(blender_dir, "..", "Resources", "scripts", "modules"),
            os.path.join(blender_dir, "..", "Resources", "scripts", "addons", "modules"),
            os.path.join(blender_dir, "..", "Resources", "scripts", "addons"),
            os.path.join(blender_dir, "..", "Resources", "scripts", "addons_contrib"),
            os.path.join(blender_dir, "..", "Resources", "python", "lib", "python3.11", "site-packages")
        ])
        
        # ユーザーディレクトリのBlenderパス（macOS）
        if user_blender_path:
            lib_paths.extend([
                os.path.join(user_blender_path, "scripts", "startup"),
                os.path.join(user_blender_path, "scripts", "modules"),
                os.path.join(user_blender_path, "scripts", "addons", "modules"),
                os.path.join(user_blender_path, "scripts", "addons"),
                os.path.join(user_blender_path, "scripts", "addons_contrib")
            ])
        
    else:  # Linux
        # Linux: blender/{version}/
        lib_paths.extend([
            # scripts関連
            os.path.join(blender_dir, version, "scripts", "startup"),
            os.path.join(blender_dir, version, "scripts", "modules"),
            os.path.join(blender_dir, version, "scripts", "addons", "modules"),
            os.path.join(blender_dir, version, "scripts", "addons"),
            os.path.join(blender_dir, version, "scripts", "addons_contrib"),
            
            # python関連
            os.path.join(blender_dir, f"python{sys.version_info.major}{sys.version_info.minor}.zip"),
            os.path.join(blender_dir, version, "python", "lib", "python3.11", "site-packages"),
            os.path.join(blender_dir, version, "python", "lib"),
            os.path.join(blender_dir, version, "python", "bin"),
            os.path.join(blender_dir, version, "python"),
            
            # バックアップパス
            os.path.join(blender_dir, "scripts", "startup"),
            os.path.join(blender_dir, "scripts", "modules"),
            os.path.join(blender_dir, "scripts", "addons", "modules"),
            os.path.join(blender_dir, "scripts", "addons"),
            os.path.join(blender_dir, "scripts", "addons_contrib"),
            os.path.join(blender_dir, "python", "lib", "python3.11", "site-packages")
        ])
        
        # ユーザーディレクトリのBlenderパス（Linux）
        if user_blender_path:
            lib_paths.extend([
                os.path.join(user_blender_path, "scripts", "startup"),
                os.path.join(user_blender_path, "scripts", "modules"),
                os.path.join(user_blender_path, "scripts", "addons", "modules"),
                os.path.join(user_blender_path, "scripts", "addons"),
                os.path.join(user_blender_path, "scripts", "addons_contrib")
            ])
    
    # 現在のBlenderのsite-packagesパスも追加
    for path in site.getsitepackages():
        if path not in lib_paths:
            lib_paths.append(path)
    
    # アドオンディレクトリ内の個別の依存関係パスを動的に検索
    def find_addon_deps_paths():
        addon_deps_paths = []
        addon_dirs = []
        
        # インストールディレクトリのaddonsパス
        addon_dirs.append(os.path.join(blender_dir, version, "scripts", "addons"))
        
        # ユーザーディレクトリのaddonsパス
        if user_blender_path:
            addon_dirs.append(os.path.join(user_blender_path, "scripts", "addons"))
        
        for addon_dir in addon_dirs:
            if os.path.exists(addon_dir):
                try:
                    for addon_name in os.listdir(addon_dir):
                        addon_path = os.path.join(addon_dir, addon_name)
                        if os.path.isdir(addon_path):
                            # depsディレクトリがあるか確認
                            deps_path = os.path.join(addon_path, "deps")
                            if os.path.exists(deps_path):
                                addon_deps_paths.append(deps_path)
                except (OSError, PermissionError):
                    # アクセス権限がない場合などはスキップ
                    continue
        
        return addon_deps_paths
    
    # アドオンの依存関係パスを追加
    addon_deps_paths = find_addon_deps_paths()
    lib_paths.extend(addon_deps_paths)
    
    # 重複を除去し、存在するパスのみを返す
    unique_paths = []
    for path in lib_paths:
        if path not in unique_paths and os.path.exists(path):
            unique_paths.append(path)
    
    return unique_paths

def run_rbf_processor(temp_file_path, python_path=None, processor_path=None, old_version=False):
    """
    rbf_multithread_processor.pyを実行する
    
    Parameters:
        temp_file_path (str): 一時データファイルのパス
        python_path (str, optional): Pythonバイナリのパス
        processor_path (str, optional): プロセッサスクリプトのパス
        old_version (bool, optional): 古いバージョン形式で出力するかどうか
    
    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    try:
        np.show_config()
        
        # デフォルトパスを取得
        if python_path is None:
            python_path = get_blender_python_path()
        
        if processor_path is None:
            processor_path = get_rbf_processor_script_path()
        
        # パスの存在確認
        if not os.path.exists(python_path):
            return False, "", f"Pythonバイナリが見つかりません: {python_path}"
        
        if not os.path.exists(processor_path):
            return False, "", f"RBFプロセッサスクリプトが見つかりません: {processor_path}"
        
        # BlenderのPythonライブラリパスを取得
        blender_lib_paths = get_blender_python_lib_paths()
        
        print(f"Detected Blender library paths:")
        for path in blender_lib_paths:
            print(f"  - {path}")

        # --userでインストールされたパッケージのパスを取得
        user_site_packages = get_blender_python_user_site_packages(python_path)
        if user_site_packages:
            print(f"Detected user site packages path: {user_site_packages}")
        else:
            print("User site packages path not found")

        blender_deps_path = os.path.join(os.path.dirname(__file__), 'deps')
        print(f"Blender deps path: {blender_deps_path}")
        
        # 方法1: 環境変数を使用してライブラリパスを設定
        env = os.environ.copy()
        
        # Windows特有の文字エンコーディング問題を回避
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
        
        # PYTHONPATHにBlenderのライブラリパスとユーザーサイトパッケージパスを追加
        pythonpath_parts = []
        if 'PYTHONPATH' in env:
            pythonpath_parts.append(env['PYTHONPATH'])
        
        # ユーザーサイトパッケージを最初に追加（優先度を高くする）
        if blender_deps_path:
            pythonpath_parts.append(blender_deps_path)
        if user_site_packages:
            pythonpath_parts.append(user_site_packages)
            
        pythonpath_parts.extend(blender_lib_paths)
        env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
        
        # Pythonの標準出力バッファリングを無効にする
        env['PYTHONUNBUFFERED'] = '1'
        
        print(f"Configured PYTHONPATH: {env['PYTHONPATH']}")
        
        # コマンドを構築（-uフラグでバッファリングを無効化）
        cmd = [python_path, '-u', processor_path, temp_file_path]
        
        # 古いバージョン形式のオプションを追加
        if old_version:
            cmd.append('--old-version')
        
        max_workers = os.cpu_count()
        env['OMP_NUM_THREADS'] = str(max_workers)
        env['OPENBLAS_NUM_THREADS'] = str(max_workers)
        env['MKL_NUM_THREADS'] = str(max_workers)
        env['VECLIB_MAXIMUM_THREADS'] = str(max_workers)
        env['NUMEXPR_NUM_THREADS'] = str(max_workers)
        cmd.append('--max-workers')
        cmd.append(str(max_workers))

        print(f"Configured environment variables: {env}")

        print(f"Executing command: {' '.join(cmd)}")

        # プロセスを実行（リアルタイム出力）
        print("Starting RBF processing...")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 標準エラーを標準出力にリダイレクト
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=os.path.dirname(temp_file_path),
                env=env,
                bufsize=1,  # 行バッファリング
                universal_newlines=True
            )
            
            # リアルタイムで出力を読み取り、表示
            import select
            import sys
            output_lines = []
            
            # Windows環境でのリアルタイム読み取り
            if sys.platform == "win32":
                # Windowsの場合、selectが使えないので別のアプローチ
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        line = line.rstrip('\n\r')
                        print(f"[RBF Processing] {line}")
                        output_lines.append(line)
            else:
                # Unix系の場合はselectを使用
                while True:
                    if process.poll() is not None:
                        # プロセスが終了している場合、残りの出力を読み取る
                        remaining = process.stdout.read()
                        if remaining:
                            for line in remaining.splitlines():
                                print(f"[RBF Processing] {line}")
                                output_lines.append(line)
                        break
                    
                    # 読み取り可能かチェック
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready:
                        line = process.stdout.readline()
                        if line:
                            line = line.rstrip('\n\r')
                            print(f"[RBF Processing] {line}")
                            output_lines.append(line)
            
            # プロセスの完了を待つ
            process.wait()
            success = process.returncode == 0
            output = '\n'.join(output_lines)
            
            if success:
                print("RBF processing completed successfully")
            else:
                print(f"RBF processing failed (return code: {process.returncode})")

        except Exception as e:
            print(f"Error during process execution: {e}")
            success = False
            output = ""
        
        return success, output, ""
        
    except Exception as e:
        error_msg = f"RBF処理の実行中にエラーが発生しました: {str(e)}"
        print(error_msg)
        return False, "", error_msg


# デバッグ用オペレーター：Pythonパス情報表示
class DEBUG_OT_ShowPythonPaths(bpy.types.Operator):
    bl_idname = "rbf.debug_show_python_paths"
    bl_label = "Show Python Paths"
    bl_description = "Pythonパスとライブラリパスを表示"
    
    def execute(self, context):
        try:
            # BlenderのPythonバイナリパス
            python_path = get_blender_python_path()
            print(f"\n{'='*60}")
            print(f"PYTHON Path Information")
            print(f"{'='*60}")
            print(f"Python binary path: {python_path}")
            print(f"Exists: {os.path.exists(python_path)}")

            # ユーザーサイトパッケージパス
            user_site_packages = get_blender_python_user_site_packages(python_path)
            print(f"\nUser site packages path:")
            if user_site_packages:
                print(f"  {user_site_packages}")
                print(f"  Exists: {'Yes' if os.path.exists(user_site_packages) else 'No'}")
            else:
                print("  Not found")

            # BlenderのPythonライブラリパス
            lib_paths = get_blender_python_lib_paths()
            print(f"\nBlender library paths:")
            for i, path in enumerate(lib_paths, 1):
                print(f"  {i}. {path}")
                print(f"     Exists: {'Yes' if os.path.exists(path) else 'No'}")

            # RBFプロセッサスクリプトパス
            processor_path = get_rbf_processor_script_path()
            print(f"\nRBF processor script path: {processor_path}")
            print(f"Exists: {os.path.exists(processor_path)}")

            # 現在のPYTHONPATH
            current_pythonpath = os.environ.get('PYTHONPATH', 'Not set')
            print(f"\nCurrent PYTHONPATH: {current_pythonpath}")

            # Blender内でのscipysチェック
            try:
                import scipy
                print(f"\nSciPy in Blender: Available (version: {scipy.__version__})")
                print(f"SciPy path: {scipy.__file__}")
            except ImportError as e:
                print(f"\nSciPy in Blender: Not available ({e})")
            
            print(f"{'='*60}")
            
            self.report({'INFO'}, "Debug info printed to console")
            return {'FINISHED'}
        
        except Exception as e:
            error_msg = f"デバッグ情報の取得に失敗しました: {str(e)}"
            print(error_msg)
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# フィールド可視化オペレーター
class CREATE_OT_FieldVisualization(bpy.types.Operator):
    bl_idname = "rbf.create_field_visualization"
    bl_label = "Create Field Visualization"
    bl_description = "既存の変形データからフィールドをBlenderオブジェクトとして可視化"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # オブジェクトモードに切り替え
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name.strip()
        target_avatar_name = scene.rbf_target_avatar_name.strip()
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        source_shape_key_name = scene.rbf_source_shape_key
        field_step = scene.rbf_field_step
        use_inverse = scene.rbf_field_use_inverse
        object_name = scene.rbf_field_object_name.strip()
        
        if not source_avatar_name:
            self.report({'ERROR'}, "Please specify source avatar name")
            return {'CANCELLED'}
        
        if not object_name:
            object_name = "FieldVisualization"
        
        # ファイルパスを現在の設定に基づいて生成（後方互換性対応）
        scene_folder = get_scene_folder()
        inverse_suffix = "_inv" if use_inverse else ""

        if save_shape_key_mode:
            # シェイプキー変形モードの場合
            if not source_shape_key_name:
                self.report({'ERROR'}, "Please specify shape key name in shape key mode")
                return {'CANCELLED'}
            display_name = "Shape key deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                source_shape_key_name=source_shape_key_name,
                inverse_suffix=inverse_suffix
            )
        else:
            # 通常のアバター間変形の場合
            if not target_avatar_name:
                self.report({'ERROR'}, "Please specify target avatar name")
                return {'CANCELLED'}
            display_name = "Inter-avatar deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                target_avatar_name=target_avatar_name,
                inverse_suffix=inverse_suffix
            )

        if not field_data_path:
            expected_filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_{f'shape_{source_shape_key_name}' if save_shape_key_mode else f'to_{normalize_avatar_name_for_filename(target_avatar_name)}'}{inverse_suffix}.npz"
            self.report({'ERROR'}, f"{display_name} file not found: {expected_filename}")
            print(f"Deformation data file not found in: {scene_folder}")
            return {'CANCELLED'}
        
        try:
            # フィールドオブジェクトを作成
            field_obj = create_field_object_from_data(
                field_data_path=field_data_path,
                target_step=field_step,
                object_name=object_name
            )
            
            direction_text = "inverse" if use_inverse else "normal"
            self.report({'INFO'}, f"Field object '{field_obj.name}' created ({direction_text}, step {field_step})")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# Numpy・Scipy再インストール関数
def get_numpy_version():
    """
    現在インストールされているnumpyのバージョンを取得

    importlib.metadata を使用してモジュールをロードせずにバージョンを取得する。
    これによりファイルロックを防止する。
    """
    try:
        from importlib.metadata import version
        return version("numpy")
    except Exception:
        return None

def get_scipy_version():
    """
    現在インストールされているscipyのバージョンを取得

    importlib.metadata を使用してモジュールをロードせずにバージョンを取得する。
    これによりファイルロックを防止する。
    """
    try:
        from importlib.metadata import version
        return version("scipy")
    except Exception:
        return None

def _rmtree_onerror(func, path, exc_info):
    """
    shutil.rmtree の onerror ハンドラー

    Windows で read-only 属性のファイルを削除できるようにする。
    """
    import stat
    # read-only 属性を解除して再試行
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise exc_info[1]

def safe_rmtree(path: str) -> tuple:
    """
    安全にディレクトリを削除する

    Args:
        path: 削除するディレクトリのパス

    Returns:
        tuple: (success: bool, error_type: str, error_message: str)
            error_type: "LOCK", "ERROR", "" (成功時)
    """
    if not os.path.exists(path):
        return True, "", ""

    try:
        shutil.rmtree(path, onerror=_rmtree_onerror)
        return True, "", ""
    except PermissionError as e:
        return False, "LOCK", f"ファイルがロックされています。Blenderを再起動してから再実行してください: {e}"
    except OSError as e:
        return False, "ERROR", f"ディレクトリの削除に失敗: {str(e)}"

def safe_rename(src: str, dst: str) -> tuple:
    """
    安全にディレクトリをリネームする

    Args:
        src: 元のパス
        dst: 新しいパス

    Returns:
        tuple: (success: bool, error_type: str, error_message: str)
    """
    try:
        if os.path.exists(dst):
            # 既存の dst を削除
            success, err_type, err_msg = safe_rmtree(dst)
            if not success:
                return False, err_type, err_msg

        os.rename(src, dst)
        return True, "", ""
    except PermissionError as e:
        return False, "LOCK", f"ファイルがロックされています。Blenderを再起動してから再実行してください: {e}"
    except OSError as e:
        return False, "ERROR", f"リネームに失敗: {str(e)}"

def reinstall_numpy_scipy_multithreaded(python_path, numpy_version, scipy_version):
    """
    numpyとscipyをマルチスレッド対応版で強制再インストール

    安全なインストール方式:
    1. 一時ディレクトリ (deps_new) にインストール
    2. 成功したら既存の deps を deps_old にリネーム
    3. deps_new を deps にリネーム
    4. deps_old を削除

    これにより pip 失敗時も既存の deps が保持される。

    Parameters:
        python_path (str): BlenderのPythonバイナリのパス（メインスレッドで事前取得）
        numpy_version (str): NumPyのバージョン
        scipy_version (str or None): SciPyのバージョン（未インストールの場合はNone）

    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    try:
        # パラメータの検証（bpy依存の値はメインスレッドで事前取得済み）
        if not numpy_version:
            return False, "", "numpy が見つかりません"

        if not python_path or not os.path.exists(python_path):
            return False, "", f"Pythonパスが見つかりません: {python_path}"

        # インストールするパッケージリストを作成
        packages = [f"numpy=={numpy_version}"]
        if scipy_version:
            packages.append(f"scipy=={scipy_version}")
        else:
            # scipyがインストールされていない場合は最新版をインストール
            packages.append("scipy")
        # psutilも一緒にインストール（メモリ監視用）
        packages.append("psutil")

        addon_dir = os.path.dirname(__file__)
        deps_path = os.path.join(addon_dir, 'deps')
        deps_new_path = os.path.join(addon_dir, 'deps_new')
        deps_old_path = os.path.join(addon_dir, 'deps_old')

        print(f"\n{'='*60}")
        print(f"NumPy/SciPy/psutil Reinstallation Starting")
        print(f"{'='*60}")
        print(f"NumPy version: {numpy_version}")
        if scipy_version:
            print(f"SciPy version: {scipy_version}")
        else:
            print("SciPy: Not installed (will install new)")
        print("psutil: Latest version (for memory monitoring)")

        # 一時ディレクトリをクリーンアップ（前回の失敗時のゴミを削除）
        # 注意: Windows ではファイルシステムの状態が遅延することがあるため
        # os.path.exists() が False でも実際には存在する場合がある
        # そのため、存在チェックをせずに常に削除を試みる
        for tmp_path in [deps_new_path, deps_old_path]:
            print(f"Cleaning up temporary path: {tmp_path}")
            try:
                # まずファイルとして削除を試みる
                try:
                    os.remove(tmp_path)
                    print(f"  Deleted file")
                    continue
                except IsADirectoryError:
                    # ディレクトリの場合は rmtree へ
                    pass
                except FileNotFoundError:
                    # 存在しない場合はスキップ
                    print(f"  Does not exist (skipping)")
                    continue
                except PermissionError:
                    # ディレクトリの可能性があるので rmtree へ
                    pass

                # ディレクトリとして削除を試みる
                try:
                    shutil.rmtree(tmp_path, onerror=_rmtree_onerror)
                    print(f"  Deleted directory")
                except FileNotFoundError:
                    print(f"  Does not exist (skipping)")
                except PermissionError as e:
                    err_msg = f"ファイルがロックされています。Blenderを再起動してから再実行してください: {e}"
                    print(f"  {err_msg}")
                    return False, "", err_msg
                except OSError as e:
                    err_msg = f"削除に失敗: {e}"
                    print(f"  {err_msg}")
                    return False, "", err_msg

            except Exception as e:
                err_msg = f"予期しないエラー: {e}"
                print(f"  {err_msg}")
                return False, "", err_msg

        # 一時ディレクトリを作成
        # 注意: Microsoft Store版Blenderではos.makedirs()が失敗するため、
        # Windowsでは cmd /c mkdir を優先的に使用する
        print(f"Creating temporary directory: {deps_new_path}")

        def create_directory(path: str) -> tuple:
            """
            クロスプラットフォームでディレクトリを作成する。
            Windows: Store版Blenderのサンドボックス環境に対応するため、
                     cmd /c mkdir を優先的に使用する。
            Linux/macOS: os.makedirs() を使用する。
            """
            import sys

            # Windowsの場合: cmd /c mkdir を優先（Store版Blender対応）
            if sys.platform == 'win32':
                try:
                    # 出力は不要なので DEVNULL を使用（UnicodeDecodeError 回避）
                    result = subprocess.run(
                        ['cmd', '/c', 'mkdir', path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        shell=False
                    )
                    if result.returncode == 0:
                        return True, "cmd"
                    # 既に存在する場合もエラーコード1が返る
                    if os.path.isdir(path):
                        return True, "cmd (already exists)"
                except Exception as e:
                    print(f"  cmd /c mkdir exception: {e}")

            # os.makedirs（Linux/macOS、またはWindowsでcmdが失敗した場合のフォールバック）
            try:
                os.makedirs(path, exist_ok=True)
                return True, "os.makedirs"
            except OSError as e:
                print(f"  os.makedirs failed: {e}")

            return False, ""

        success, method = create_directory(deps_new_path)
        if success:
            print(f"Created temporary directory ({method})")
        else:
            return False, "", f"Failed to create temporary directory: {deps_new_path}"

        # pip download + 手動展開方式
        # Microsoft Store版Blenderでは pip install --target が
        # クロスドライブ移動エラー(WinError 17)を起こすため、
        # wheel をダウンロードして手動で展開する
        import zipfile

        wheels_path = os.path.join(deps_new_path, '_wheels')
        success, method = create_directory(wheels_path)
        if not success:
            return False, "", f"Failed to create wheel download directory: {wheels_path}"
        print(f"Created wheel download directory: {wheels_path}")

        # Step 1: pip download で wheel ファイルをダウンロード
        cmd = [python_path, "-m", "pip", "download",
               "--no-cache-dir",
               "--only-binary=:all:",  # ソースビルドを避ける
               "--dest", wheels_path] + packages
        print(f"Executing command: {' '.join(cmd)}")

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
        env['PIP_NO_CACHE_DIR'] = '1'

        # subprocess.run() の capture_output=True は Windows で UnicodeDecodeError を
        # 起こすことがあるため、バイナリモードで実行して手動でデコード
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        stdout_bytes, stderr_bytes = process.communicate()

        # モジュールレベルの safe_decode() を使用してデコード
        stdout_text = safe_decode(stdout_bytes)
        stderr_text = safe_decode(stderr_bytes)

        class Result:
            returncode = process.returncode
            stdout = stdout_text
            stderr = stderr_text

        result = Result()

        print(f"Execution result (return code: {result.returncode}):")
        print(f"Output:\n{result.stdout}")

        if result.stderr:
            print(f"Error output:\n{result.stderr}")

        if result.returncode != 0:
            print("pip download failed. Existing deps will be kept.")
            safe_rmtree(deps_new_path)
            return False, result.stdout, result.stderr

        # Step 2: wheel ファイルを展開
        # zipfile.extractall() は内部で os.makedirs() を使用するため、
        # Microsoft Store版Blenderで WinError 183 が発生する。
        # そのため、ファイルを1つずつ展開し、ディレクトリ作成には
        # cmd /c mkdir を使用する。
        print("Extracting wheel files...")
        wheel_files = [f for f in os.listdir(wheels_path) if f.endswith('.whl')]

        if not wheel_files:
            print("Error: No wheel files found")
            safe_rmtree(deps_new_path)
            return False, result.stdout, "No wheel files found"

        # 作成済みディレクトリを追跡（重複作成を避ける）
        created_dirs = set()

        for wheel_file in wheel_files:
            wheel_path = os.path.join(wheels_path, wheel_file)
            print(f"  Extracting: {wheel_file}")
            try:
                with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        # 展開先パス
                        target_path = os.path.join(deps_new_path, member)

                        # ディレクトリエントリの場合
                        if member.endswith('/'):
                            if target_path not in created_dirs:
                                create_directory(target_path)
                                created_dirs.add(target_path)
                            continue

                        # ファイルの場合：親ディレクトリを作成
                        parent_dir = os.path.dirname(target_path)
                        if parent_dir and parent_dir not in created_dirs:
                            create_directory(parent_dir)
                            created_dirs.add(parent_dir)

                        # ファイルを展開
                        with zip_ref.open(member) as source:
                            with open(target_path, 'wb') as target:
                                target.write(source.read())

            except Exception as e:
                print(f"  Extraction error: {e}")
                import traceback
                traceback.print_exc()
                safe_rmtree(deps_new_path)
                return False, result.stdout, f"Wheel extraction error: {e}"

        # Step 3: wheels ディレクトリを削除
        print("Cleaning up wheel files...")
        safe_rmtree(wheels_path)

        # pip 成功: ディレクトリを置き換え
        print("Installation successful. Replacing directory...")

        # 既存の deps があれば deps_old にリネーム
        if os.path.exists(deps_path):
            print(f"Moving existing deps to deps_old...")
            success, err_type, err_msg = safe_rename(deps_path, deps_old_path)
            if not success:
                print(f"Failed to rename deps: {err_msg}")
                # 失敗しても新しい deps_new は残す（手動復旧用）
                return False, result.stdout, err_msg

        # deps_new を deps にリネーム
        print(f"Moving deps_new to deps...")
        success, err_type, err_msg = safe_rename(deps_new_path, deps_path)
        if not success:
            print(f"Failed to rename deps_new: {err_msg}")
            # deps_old を deps に戻す
            if os.path.exists(deps_old_path):
                safe_rename(deps_old_path, deps_path)
            return False, result.stdout, err_msg

        # deps_old を削除（失敗しても警告のみ）
        if os.path.exists(deps_old_path):
            print(f"Deleting old deps_old...")
            success, _, err_msg = safe_rmtree(deps_old_path)
            if not success:
                print(f"Warning: Failed to delete deps_old (please delete manually): {err_msg}")

        print("Directory replacement complete")
        return True, result.stdout, result.stderr

    except Exception as e:
        error_msg = f"Error occurred during NumPy/SciPy reinstallation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return False, "", error_msg


# numpy・scipy再インストールオペレーター（Modal版 - UIフリーズ回避）
class REINSTALL_OT_NumpyScipyMultithreaded(bpy.types.Operator):
    bl_idname = "rbf.reinstall_numpy_scipy_multithreaded"
    bl_label = "Reinstall NumPy & SciPy & psutil"
    bl_description = "numpy, scipy, psutilを再インストール（マルチスレッド対応版）"

    # インストールスレッドの状態を保持
    _timer = None
    _thread = None
    _result = None  # (success, output, error)
    _numpy_version = None
    _scipy_version = None
    _dot_count = 0  # アニメーション用カウンター

    def modal(self, context, event):
        if event.type == 'TIMER':
            # ステータスバーのアニメーション更新
            self._dot_count = (self._dot_count + 1) % 4
            dots = "." * (self._dot_count + 1)
            context.workspace.status_text_set(f"Installing NumPy/SciPy{dots}")

            # スレッドの完了をチェック
            if self._thread is not None and not self._thread.is_alive():
                # タイマーを停止
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None

                # ステータスバーをクリア
                context.workspace.status_text_set(None)

                # 結果を取得
                success, output, error = self._result if self._result else (False, "", "Unknown error")

                if success:
                    packages_info = f"NumPy {self._numpy_version}"
                    if self._scipy_version:
                        packages_info += f", SciPy {self._scipy_version}"
                    else:
                        packages_info += ", SciPy (new installation)"

                    self.report({'WARNING'}, f"{packages_info} reinstalled. Please restart Blender")
                    print(f"NumPy/SciPy reinstall succeeded. Please restart Blender.")

                    # 成功ポップアップを表示
                    def draw_success_popup(self, context):
                        self.layout.label(text="NumPy/SciPy installation complete")
                        self.layout.label(text="")
                        self.layout.label(text="Please restart Blender", icon='ERROR')

                    context.window_manager.popup_menu(draw_success_popup, title="Installation Complete", icon='CHECKMARK')
                else:
                    if error:
                        self.report({'ERROR'}, error)
                    else:
                        self.report({'ERROR'}, "NumPy/SciPy reinstallation failed")

                    # エラーポップアップを表示
                    def draw_error_popup(self, context):
                        self.layout.label(text="Installation failed")
                        self.layout.label(text="")
                        if error:
                            # エラーメッセージを短く表示
                            short_error = error[:80] + "..." if len(error) > 80 else error
                            self.layout.label(text=short_error)
                        self.layout.label(text="See console for details", icon='INFO')

                    context.window_manager.popup_menu(draw_error_popup, title="Installation Error", icon='ERROR')

                # UIを更新
                for area in context.screen.areas:
                    area.tag_redraw()

                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        import threading

        # 現在のバージョンを取得（メインスレッドで取得）
        self._numpy_version = get_numpy_version()
        self._scipy_version = get_scipy_version()

        if not self._numpy_version:
            self.report({'ERROR'}, "numpy not found")
            return {'CANCELLED'}

        # bpy依存の値をメインスレッドで事前取得（スレッド安全性のため）
        python_path = get_blender_python_path()
        numpy_version = self._numpy_version
        scipy_version = self._scipy_version

        if not python_path:
            self.report({'ERROR'}, "Python path not found")
            return {'CANCELLED'}

        # インストールを別スレッドで実行（純Pythonデータのみ渡す）
        def run_install():
            try:
                self._result = reinstall_numpy_scipy_multithreaded(
                    python_path, numpy_version, scipy_version
                )
            except Exception as e:
                self._result = (False, "", str(e))

        self._thread = threading.Thread(target=run_install)
        self._thread.start()

        # タイマーを設定（0.5秒ごとにチェック）
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)

        # ステータスバーに表示開始
        self._dot_count = 0
        context.workspace.status_text_set("Installing NumPy/SciPy.")

        self.report({'INFO'}, "Installing... (running in background)")

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        # 実行前に確認ダイアログを表示
        numpy_version = get_numpy_version()
        scipy_version = get_scipy_version()

        if numpy_version:
            return context.window_manager.invoke_confirm(self, event)
        else:
            self.report({'ERROR'}, "numpy not found")
            return {'CANCELLED'}

    def cancel(self, context):
        # キャンセル時にタイマーをクリーンアップ
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        # ステータスバーをクリア
        context.workspace.status_text_set(None)


# デバッグ用オペレーター：外部Pythonでscipyテスト
class DEBUG_OT_TestExternalPython(bpy.types.Operator):
    bl_idname = "rbf.debug_test_external_python"
    bl_label = "Test External Python"
    bl_description = "外部Pythonでscipyのインポートをテスト"
    
    def execute(self, context):
        try:
            python_path = get_blender_python_path()
            blender_lib_paths = get_blender_python_lib_paths()
            
            # テストスクリプトを作成（エンコーディング問題を回避するため英語で記述）
            test_script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Add Blender library paths
blender_lib_paths = {repr(blender_lib_paths)}

print("Python executable path:", sys.executable)
print("Python version:", sys.version)
print()

print("Adding library paths:")
for lib_path in blender_lib_paths:
    exists = "YES" if os.path.exists(lib_path) else "NO"
    print(f"  - {{lib_path}} (exists: {{exists}})")
    if os.path.exists(lib_path) and lib_path not in sys.path:
        sys.path.insert(0, lib_path)

print()
print("Current sys.path:")
for i, path in enumerate(sys.path):
    print(f"  {{i+1}}. {{path}}")

print()
print("scipy import test:")
try:
    import scipy
    print(f"scipy: SUCCESS (version: {{scipy.__version__}})")
    print(f"scipy path: {{scipy.__file__}}")
    
    from scipy.spatial import cKDTree
    print("cKDTree: import SUCCESS")
    
    import numpy as np
    print(f"numpy: SUCCESS (version: {{np.__version__}})")
    
    import mathutils
    print(f"mathutils: SUCCESS (version: {{mathutils.__version__}})")
    
    from mathutils.bvhtree import BVHTree
    print("BVHTree: import SUCCESS")
    
except ImportError as e:
    print(f"import FAILED: {{e}}")

print("\\nTest completed")
'''
            
            # テストスクリプトを一時ファイルに保存
            scene_folder = get_scene_folder()
            test_script_path = os.path.join(scene_folder, "test_scipy_import.py")
            
            with open(test_script_path, 'w', encoding='utf-8') as f:
                f.write(test_script_content)
            
            # 環境変数を設定
            env = os.environ.copy()
            
            # Windows特有の文字エンコーディング問題を回避
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
            
            pythonpath_parts = []
            if 'PYTHONPATH' in env:
                pythonpath_parts.append(env['PYTHONPATH'])
            pythonpath_parts.extend(blender_lib_paths)
            env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
            
            # テストスクリプトを実行
            cmd = [python_path, test_script_path]
            print(f"\n{'='*60}")
            print(f"External Python Test Execution")
            print(f"{'='*60}")
            print(f"Executing command: {' '.join(cmd)}")

            returncode, stdout, stderr = run_subprocess_safe(cmd, env=env, cwd=scene_folder)

            print(f"Execution result (return code: {returncode}):")
            print(f"Output:\n{stdout}")

            if stderr:
                print(f"Error output:\n{stderr}")
            
            # テストスクリプトを削除
            try:
                os.remove(test_script_path)
            except:
                pass
            
            if returncode == 0:
                self.report({'INFO'}, "External Python test succeeded")
            else:
                self.report({'WARNING'}, "External Python test detected issues")
            
            return {'FINISHED'}
        
        except Exception as e:
            error_msg = f"External Python test failed: {str(e)}"
            print(error_msg)
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# アドオンとして読み込まれた場合はここでは自動登録しない
# register()とunregister()は__init__.pyから呼び出される
