# -*- coding: utf-8 -*-
"""
MochiFitter - Advanced Avatar Outfit Retargeting System for Blender
"""

bl_info = {
    "name": "MochiFitter-Kai",
    "author": "Community Fork (Original: MochiFitter Development Team)",
    "version": (0, 2, 14),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > MochiFitter",
    "description": "Community-optimized fork of MochiFitter - Avatar Outfit Retargeting System using RBF interpolation",
    "warning": "非公式版 - This is an unofficial community fork",
    "wiki_url": "https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai",
    "category": "Mesh",
    "support": "COMMUNITY"
}

import bpy
import sys
import os
import importlib

# deps ディレクトリをパスに追加（scipy等の依存関係用）
# 注意: scipy の import は遅延実行する（ファイルロック防止のため）
libs_path = os.path.join(os.path.dirname(__file__), 'deps')
if libs_path not in sys.path:
    sys.path.append(libs_path)

# アドオンのディレクトリパスを取得
addon_dir = os.path.dirname(__file__)

# モジュールの動的インポート
def reload_modules():
    """アドオンのモジュールを再読み込みする"""
    import importlib
    
    # すでにインポートされているモジュールがあれば再読み込み
    if "SaveAndApplyFieldAuto" in locals():
        importlib.reload(SaveAndApplyFieldAuto)
    
    # メインモジュールをインポート
    from . import SaveAndApplyFieldAuto

def register():
    """アドオンを登録する"""
    # メインモジュールを再読み込み
    reload_modules()
    
    # メインモジュールの登録
    from . import SaveAndApplyFieldAuto
    
    try:
        SaveAndApplyFieldAuto.register()
        print("MochiFitter アドオンが正常に登録されました")
    except Exception as e:
        print(f"MochiFitter アドオンの登録中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

def unregister():
    """アドオンの登録を解除する"""
    from . import SaveAndApplyFieldAuto
    
    try:
        SaveAndApplyFieldAuto.unregister()
        print("MochiFitter アドオンの登録が解除されました")
    except Exception as e:
        print(f"MochiFitter アドオンの登録解除中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

# スクリプトとして直接実行された場合の処理
if __name__ == "__main__":
    register() 