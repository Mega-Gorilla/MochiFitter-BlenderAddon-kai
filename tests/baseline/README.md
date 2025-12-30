# Baseline Scripts

This directory contains the original MochiFitter scripts before any optimization work.

## Files

### retarget_script2_14_original.py

- **Source**: Original MochiFitter Unity addon Blender script
- **Date**: 2025-12-29 (backup created before optimization work)
- **Version**: Pre-optimization baseline (before Issue #48)

This file represents the unmodified MochiFitter script and serves as the baseline for performance comparisons.

## Benchmark Results (2025-12-31)

### Test Case: Beryl → Template → mao (chain processing)

#### Original MochiFitter (pre-#48)

| Run | Duration |
|-----|----------|
| 1 | 278.87s |
| 2 | 258.75s |
| 3 | 258.34s |
| **Average** | **265.32s** |

#### Phase 1 Complete (P1-0 + P1-1 + P1-2)

| Run | Duration |
|-----|----------|
| 1 | 299.97s |
| 2 | 292.88s |
| 3 | 289.07s |
| **Average** | **293.97s** |

### Analysis

| Version | Average | Difference |
|---------|---------|------------|
| Original (pre-#48) | 265.32s | baseline |
| Phase 1 complete | 293.97s | **+10.8% slower** |

### Important Notes

1. **Data Integrity**: The original script does NOT include Issue #46 fixes (data integrity improvements)
2. **P1-0 Overhead**: The `foreach_get` optimization returns numpy arrays, requiring Vector conversion in `inverse_bone_deform_all_vertices()` which adds overhead
3. **No Numba**: Tests were run WITHOUT Numba installed, so P1-2 uses NumPy fallback (not JIT-compiled)
4. **Quality vs Speed**: The Phase 1 version includes bug fixes and improved data integrity, which may be more important than raw speed

### CLI Differences

The original script uses a different CLI format:

```bash
# Original format
--input (not --source)
--base (required .blend file)
--base-fbx (semicolon-separated)
--config (semicolon-separated)
--target-meshes (semicolon-separated)

# Current format
--source (not --input)
--target (multiple arguments)
--config (multiple arguments)
--target-meshes (comma-separated)
```

## Usage

To run a benchmark with the original script:

```bash
blender --background --python tests/baseline/retarget_script2_14_original.py -- \
    --input="path/to/source.fbx" \
    --output="path/to/output.fbx" \
    --base="path/to/base_project.blend" \
    --base-fbx="Template.fbx;target.fbx" \
    --config="config1.json;config2.json" \
    --init-pose="pose.json" \
    --hips-position=0.0,0.0,0.93 \
    --target-meshes="Mesh1;Mesh2"
```
