#!/usr/bin/env python3
"""
Benchmark: foreach_get vs list comprehension for vertex data extraction

This script benchmarks the performance difference between:
1. List comprehension: [v.co for v in mesh.vertices]
2. foreach_get: mesh.vertices.foreach_get("co", coords)

Also benchmarks matrix transformation patterns:
1. Loop: [matrix @ Vector(v) for v in vertices]
2. NumPy batch: vertices @ matrix.T (after extracting to NumPy array)

Usage (run in Blender):
    blender --background --python tests/foreach_get_benchmark.py

Or from Blender Python console:
    exec(open("tests/foreach_get_benchmark.py").read())
"""

import sys
import time
import numpy as np

# Check if running in Blender
try:
    import bpy
    from mathutils import Vector, Matrix
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("WARNING: Not running in Blender. Only NumPy benchmarks will run.")


def create_test_mesh(num_vertices: int = 10000) -> 'bpy.types.Object':
    """Create a test mesh with specified number of vertices."""
    import bmesh

    # Create new mesh
    mesh = bpy.data.meshes.new("benchmark_mesh")
    obj = bpy.data.objects.new("benchmark_obj", mesh)

    # Link to scene
    bpy.context.collection.objects.link(obj)

    # Create vertices using bmesh
    bm = bmesh.new()

    # Generate random vertices
    np.random.seed(42)
    coords = np.random.randn(num_vertices, 3).astype(np.float32)

    for co in coords:
        bm.verts.new(co)

    bm.to_mesh(mesh)
    bm.free()

    return obj


def benchmark_vertex_extraction(mesh, num_iterations: int = 100):
    """Benchmark vertex coordinate extraction methods."""
    results = {}
    num_vertices = len(mesh.vertices)

    # Method 1: List comprehension (current implementation)
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        coords = [v.co[:] for v in mesh.vertices]
        times.append(time.perf_counter() - start)
    results['list_comprehension'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }

    # Method 2: List comprehension with .copy()
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        coords = [v.co.copy() for v in mesh.vertices]
        times.append(time.perf_counter() - start)
    results['list_comprehension_copy'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }

    # Method 3: foreach_get (optimized)
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        coords = np.empty(num_vertices * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)
        times.append(time.perf_counter() - start)
    results['foreach_get'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }

    # Method 4: foreach_get with float64
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        coords = np.empty(num_vertices * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)
        times.append(time.perf_counter() - start)
    results['foreach_get_float64'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }

    return results


def benchmark_matrix_transformation(num_vertices: int = 10000, num_iterations: int = 100):
    """Benchmark matrix transformation methods."""
    results = {}

    # Generate test data
    np.random.seed(42)
    vertices = np.random.randn(num_vertices, 3).astype(np.float64)

    # Create a random 4x4 transformation matrix
    matrix = Matrix([
        [1.0, 0.1, 0.0, 0.5],
        [0.0, 1.0, 0.1, 0.3],
        [0.0, 0.0, 1.0, 0.2],
        [0.0, 0.0, 0.0, 1.0],
    ])
    matrix_np = np.array(matrix)

    # Method 1: Loop with Vector (current implementation)
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = np.array([matrix @ Vector(v) for v in vertices])
        times.append(time.perf_counter() - start)
    results['loop_vector'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }
    reference_result = result

    # Method 2: NumPy batch transformation (optimized)
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        # Add homogeneous coordinate
        vertices_homo = np.hstack([vertices, np.ones((num_vertices, 1))])
        # Matrix multiplication
        result_homo = vertices_homo @ matrix_np.T
        # Remove homogeneous coordinate
        result = result_homo[:, :3]
        times.append(time.perf_counter() - start)
    results['numpy_batch'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }

    # Verify correctness
    mse = np.mean((result - reference_result) ** 2)
    results['numpy_batch']['mse'] = mse

    # Method 3: NumPy batch (pre-allocated)
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        # Pre-allocate
        vertices_homo = np.empty((num_vertices, 4), dtype=np.float64)
        vertices_homo[:, :3] = vertices
        vertices_homo[:, 3] = 1.0
        # Matrix multiplication
        result = (vertices_homo @ matrix_np.T)[:, :3]
        times.append(time.perf_counter() - start)
    results['numpy_batch_preallocated'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }

    # Verify correctness
    mse = np.mean((result - reference_result) ** 2)
    results['numpy_batch_preallocated']['mse'] = mse

    # Method 4: NumPy 3x3 rotation + translation (if applicable)
    times = []
    rotation = matrix_np[:3, :3]
    translation = matrix_np[:3, 3]
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = vertices @ rotation.T + translation
        times.append(time.perf_counter() - start)
    results['numpy_rotation_translation'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }

    # Verify correctness
    mse = np.mean((result - reference_result) ** 2)
    results['numpy_rotation_translation']['mse'] = mse

    return results


def benchmark_combined_extraction_transform(mesh, num_iterations: int = 50):
    """Benchmark combined extraction + transformation (real-world pattern)."""
    results = {}
    num_vertices = len(mesh.vertices)

    # Create transformation matrix
    matrix = Matrix([
        [1.0, 0.1, 0.0, 0.5],
        [0.0, 1.0, 0.1, 0.3],
        [0.0, 0.0, 1.0, 0.2],
        [0.0, 0.0, 0.0, 1.0],
    ])
    matrix_np = np.array(matrix)
    rotation = matrix_np[:3, :3]
    translation = matrix_np[:3, 3]

    # Method 1: Current implementation pattern
    # np.array([matrix @ v.co for v in mesh.vertices])
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = np.array([matrix @ v.co for v in mesh.vertices])
        times.append(time.perf_counter() - start)
    results['current_pattern'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }
    reference_result = result

    # Method 2: foreach_get + NumPy batch
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        # Extract with foreach_get
        coords = np.empty(num_vertices * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)
        # Transform with NumPy
        result = coords @ rotation.T + translation
        times.append(time.perf_counter() - start)
    results['foreach_get_numpy'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }

    # Verify correctness
    mse = np.mean((result - reference_result) ** 2)
    results['foreach_get_numpy']['mse'] = mse

    return results


def print_results(title: str, results: dict, baseline_key: str = None):
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")

    if baseline_key and baseline_key in results:
        baseline_time = results[baseline_key]['mean']
    else:
        baseline_time = None

    print(f"{'Method':<35} {'Mean (ms)':<12} {'Speedup':<10} {'MSE':<12}")
    print(f"{'-' * 70}")

    for method, data in results.items():
        mean = data['mean']
        if baseline_time:
            speedup = f"{baseline_time / mean:.2f}x"
        else:
            speedup = "-"

        mse = data.get('mse', '-')
        if isinstance(mse, float):
            mse = f"{mse:.2e}"

        print(f"{method:<35} {mean:<12.2f} {speedup:<10} {mse:<12}")


def cleanup_test_objects():
    """Remove test objects from scene."""
    for obj in bpy.data.objects:
        if obj.name.startswith("benchmark_"):
            bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in bpy.data.meshes:
        if mesh.name.startswith("benchmark_"):
            bpy.data.meshes.remove(mesh)


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print(" foreach_get & Matrix Transformation Benchmark")
    print(" Based on retarget_script2_14.py patterns")
    print("=" * 70)

    if not IN_BLENDER:
        print("\nRunning NumPy-only benchmarks (not in Blender)")

        # Run matrix transformation benchmark
        for num_vertices in [10000, 30000, 100000]:
            results = benchmark_matrix_transformation(num_vertices, num_iterations=50)
            print_results(
                f"Matrix Transformation ({num_vertices:,} vertices)",
                results,
                baseline_key='loop_vector'
            )
        return

    # Run in Blender
    vertex_counts = [10000, 30000, 100000]

    for num_vertices in vertex_counts:
        print(f"\n\n{'#' * 70}")
        print(f" Testing with {num_vertices:,} vertices")
        print(f"{'#' * 70}")

        # Create test mesh
        obj = create_test_mesh(num_vertices)
        mesh = obj.data

        # Benchmark 1: Vertex extraction
        results = benchmark_vertex_extraction(mesh, num_iterations=100)
        print_results(
            f"Vertex Extraction ({num_vertices:,} vertices)",
            results,
            baseline_key='list_comprehension'
        )

        # Benchmark 2: Matrix transformation
        results = benchmark_matrix_transformation(num_vertices, num_iterations=50)
        print_results(
            f"Matrix Transformation ({num_vertices:,} vertices)",
            results,
            baseline_key='loop_vector'
        )

        # Benchmark 3: Combined (real-world pattern)
        results = benchmark_combined_extraction_transform(mesh, num_iterations=50)
        print_results(
            f"Combined Extraction + Transform ({num_vertices:,} vertices)",
            results,
            baseline_key='current_pattern'
        )

        # Cleanup
        cleanup_test_objects()

    print("\n" + "=" * 70)
    print(" Benchmark Complete")
    print("=" * 70)

    # Summary
    print("\n## Key Findings:")
    print("- foreach_get is significantly faster than list comprehension for vertex extraction")
    print("- NumPy batch matrix transformation is faster than loop with Vector")
    print("- Combined optimization (foreach_get + NumPy) provides maximum speedup")
    print("\n## Recommended Changes:")
    print("1. Replace [v.co for v in mesh.vertices] with foreach_get")
    print("2. Replace [matrix @ Vector(v) for v in vertices] with NumPy batch")
    print("3. Pre-allocate NumPy arrays where possible")


if __name__ == "__main__":
    main()
