# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MochiFitter is a Blender addon (v4.0+) for avatar outfit retargeting using RBF (Radial Basis Function) interpolation and deformation fields. It transfers mesh deformations between different avatar body types for VRChat/VRM avatars.

## Architecture

### Core Components

**SaveAndApplyFieldAuto.py** - Main addon module containing:
- UI panels (View3D > Sidebar > MochiFitter)
- Operators for pose data save/load, deformation field generation
- RBF interpolation logic for single-threaded processing
- JSON serialization for pose data (posediff, pose_basis files)
- NPZ file generation for deformation fields

**rbf_multithread_processor.py** - Standalone multiprocess script:
- Called externally by Blender addon for heavy RBF computation
- Uses ProcessPoolExecutor for parallel deformation field calculation
- Multi-Quadratic Biharmonic RBF kernel implementation
- Memory monitoring with psutil (optional)

### Key Data Flows

```
Avatar A (Source) ──┐
                    ├──> RBF Interpolation ──> deformation_*.npz
Avatar B (Target) ──┘

Armature Pose ──> save_armature_pose() ──> posediff_*.json / pose_basis_*.json
```

### File Formats Generated

- `posediff_*.json` - Bone transformation differences between avatars (location, rotation, scale, delta_matrix)
- `pose_basis_*.json` - Base pose data for an avatar
- `deformation_*.npz` - NumPy compressed arrays containing RBF interpolation data and vertex displacements
- `avatar_data_*.json` - Avatar bone hierarchy and humanoid bone mappings

## Dependencies

- Blender 4.0+
- NumPy (bundled or system)
- SciPy (for cKDTree, cdist) - addon provides reinstall button if missing

## Important Notes

- `MochFitter-unity-addon/` contains paid Unity addon content - **never commit to public repo**
- The addon dynamically loads scipy from `deps/` subdirectory
- Multithread processor runs as separate Python process to bypass Blender's GIL

## Coordinate Systems

- Blender uses Z-up, right-handed coordinate system
- Unity uses Y-up, left-handed coordinate system
- `world_matrix` in NPZ files is currently always identity matrix (known limitation)
