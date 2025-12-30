#!/usr/bin/env python3
"""
Extract mesh data from FBX for regression testing.

This script extracts vertex coordinates, normals, and vertex weights from an FBX file
and saves them as JSON/NPZ for comparison after optimization.

Usage (run in Blender):
    blender --background --python tests/extract_mesh_data.py -- input.fbx output_dir
"""

import sys
import os
import json
import numpy as np

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("ERROR: This script must be run in Blender")
    sys.exit(1)


def extract_mesh_data(fbx_path: str, output_dir: str):
    """Extract mesh data from FBX file."""
    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import FBX
    print(f"Importing FBX: {fbx_path}")
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    mesh_data = {}

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data
        mesh_name = obj.name
        print(f"Processing mesh: {mesh_name}")

        # Get vertex coordinates
        num_verts = len(mesh.vertices)
        coords = np.empty(num_verts * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)

        # Get vertex normals
        normals = np.empty(num_verts * 3, dtype=np.float64)
        mesh.vertices.foreach_get("normal", normals)
        normals = normals.reshape(-1, 3)

        # Get vertex groups (weights)
        vertex_groups = {}
        for vg in obj.vertex_groups:
            vg_name = vg.name
            weights = np.zeros(num_verts, dtype=np.float64)

            for i, vert in enumerate(mesh.vertices):
                for g in vert.groups:
                    if g.group == vg.index:
                        weights[i] = g.weight
                        break

            # Only store if there are non-zero weights
            if np.any(weights > 0):
                vertex_groups[vg_name] = weights.tolist()

        mesh_data[mesh_name] = {
            'num_vertices': num_verts,
            'vertex_coords': coords.tolist(),
            'vertex_normals': normals.tolist(),
            'vertex_groups': vertex_groups,
            'num_polygons': len(mesh.polygons),
        }

    # Save as JSON (human-readable, for diff)
    json_path = os.path.join(output_dir, 'mesh_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source_fbx': os.path.basename(fbx_path),
            'meshes': {
                name: {
                    'num_vertices': data['num_vertices'],
                    'num_polygons': data['num_polygons'],
                    'num_vertex_groups': len(data['vertex_groups']),
                    'vertex_groups_names': list(data['vertex_groups'].keys()),
                }
                for name, data in mesh_data.items()
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")

    # Save as NPZ (for numerical comparison)
    npz_path = os.path.join(output_dir, 'mesh_data.npz')
    npz_data = {}
    for name, data in mesh_data.items():
        safe_name = name.replace(' ', '_').replace('.', '_')
        npz_data[f'{safe_name}_coords'] = np.array(data['vertex_coords'], dtype=np.float32)
        npz_data[f'{safe_name}_normals'] = np.array(data['vertex_normals'], dtype=np.float32)

        # Store vertex groups
        for vg_name, weights in data['vertex_groups'].items():
            safe_vg_name = vg_name.replace(' ', '_').replace('.', '_')
            npz_data[f'{safe_name}_vg_{safe_vg_name}'] = np.array(weights, dtype=np.float32)

    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved: {npz_path}")

    # Print summary
    print("\n=== Extraction Summary ===")
    for name, data in mesh_data.items():
        print(f"  {name}: {data['num_vertices']} vertices, {data['num_polygons']} polygons, {len(data['vertex_groups'])} vertex groups")

    return mesh_data


def main():
    # Parse arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("Usage: blender --background --python extract_mesh_data.py -- input.fbx output_dir")
        sys.exit(1)

    if len(argv) < 2:
        print("Usage: blender --background --python extract_mesh_data.py -- input.fbx output_dir")
        sys.exit(1)

    fbx_path = argv[0]
    output_dir = argv[1]

    if not os.path.exists(fbx_path):
        print(f"ERROR: FBX file not found: {fbx_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    extract_mesh_data(fbx_path, output_dir)
    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
