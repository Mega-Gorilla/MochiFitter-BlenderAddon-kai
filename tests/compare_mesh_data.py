#!/usr/bin/env python3
"""
Compare mesh data between baseline and optimized output.

This script compares vertex coordinates, normals, and weights between two NPZ files
to detect any regression after optimization.

Usage:
    python tests/compare_mesh_data.py baseline_dir optimized_dir
"""

import sys
import os
import json
import numpy as np
from pathlib import Path


def load_npz_data(npz_path: str) -> dict:
    """Load NPZ file and return as dictionary."""
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def compare_arrays(name: str, baseline: np.ndarray, optimized: np.ndarray,
                   tolerance: float = 1e-5) -> dict:
    """Compare two arrays and return statistics."""
    if baseline.shape != optimized.shape:
        return {
            'name': name,
            'status': 'SHAPE_MISMATCH',
            'baseline_shape': baseline.shape,
            'optimized_shape': optimized.shape,
        }

    diff = np.abs(baseline - optimized)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    mse = np.mean(diff ** 2)

    passed = max_diff <= tolerance

    return {
        'name': name,
        'status': 'PASS' if passed else 'FAIL',
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
        'mse': float(mse),
        'tolerance': tolerance,
        'num_elements': baseline.size,
        'num_mismatched': int(np.sum(diff > tolerance)),
    }


def compare_mesh_data(baseline_dir: str, optimized_dir: str,
                      tolerance: float = 1e-5) -> dict:
    """Compare mesh data between baseline and optimized."""
    baseline_npz = os.path.join(baseline_dir, 'mesh_data.npz')
    optimized_npz = os.path.join(optimized_dir, 'mesh_data.npz')

    if not os.path.exists(baseline_npz):
        print(f"ERROR: Baseline NPZ not found: {baseline_npz}")
        sys.exit(1)

    if not os.path.exists(optimized_npz):
        print(f"ERROR: Optimized NPZ not found: {optimized_npz}")
        sys.exit(1)

    baseline_data = load_npz_data(baseline_npz)
    optimized_data = load_npz_data(optimized_npz)

    results = {
        'baseline_dir': baseline_dir,
        'optimized_dir': optimized_dir,
        'tolerance': tolerance,
        'comparisons': [],
        'summary': {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'missing_in_optimized': 0,
            'extra_in_optimized': 0,
        }
    }

    # Compare arrays that exist in both
    baseline_keys = set(baseline_data.keys())
    optimized_keys = set(optimized_data.keys())

    common_keys = baseline_keys & optimized_keys
    missing_keys = baseline_keys - optimized_keys
    extra_keys = optimized_keys - baseline_keys

    # Report missing/extra
    if missing_keys:
        print(f"\nWARNING: Missing in optimized: {len(missing_keys)} arrays")
        for key in sorted(missing_keys)[:10]:
            print(f"  - {key}")
        results['summary']['missing_in_optimized'] = len(missing_keys)

    if extra_keys:
        print(f"\nWARNING: Extra in optimized: {len(extra_keys)} arrays")
        for key in sorted(extra_keys)[:10]:
            print(f"  + {key}")
        results['summary']['extra_in_optimized'] = len(extra_keys)

    # Compare common arrays
    print(f"\nComparing {len(common_keys)} arrays...")
    print(f"{'Array Name':<50} {'Status':<10} {'Max Diff':<15} {'MSE':<15}")
    print("-" * 90)

    for key in sorted(common_keys):
        comparison = compare_arrays(
            key,
            baseline_data[key],
            optimized_data[key],
            tolerance
        )
        results['comparisons'].append(comparison)
        results['summary']['total'] += 1

        if comparison['status'] == 'PASS':
            results['summary']['passed'] += 1
            status_str = '\033[92mPASS\033[0m'  # Green
        else:
            results['summary']['failed'] += 1
            status_str = '\033[91mFAIL\033[0m'  # Red

        max_diff = comparison.get('max_diff', 'N/A')
        mse = comparison.get('mse', 'N/A')

        if isinstance(max_diff, float):
            max_diff_str = f"{max_diff:.2e}"
        else:
            max_diff_str = str(max_diff)

        if isinstance(mse, float):
            mse_str = f"{mse:.2e}"
        else:
            mse_str = str(mse)

        print(f"{key:<50} {status_str:<10} {max_diff_str:<15} {mse_str:<15}")

    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_mesh_data.py baseline_dir optimized_dir [tolerance]")
        print("\nExample:")
        print("  python compare_mesh_data.py tests/baseline_data tests/optimized_data")
        sys.exit(1)

    baseline_dir = sys.argv[1]
    optimized_dir = sys.argv[2]
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5

    print("=" * 70)
    print(" Mesh Data Regression Test")
    print("=" * 70)
    print(f"Baseline:  {baseline_dir}")
    print(f"Optimized: {optimized_dir}")
    print(f"Tolerance: {tolerance}")

    results = compare_mesh_data(baseline_dir, optimized_dir, tolerance)

    # Print summary
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    summary = results['summary']
    print(f"Total comparisons: {summary['total']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Failed: {summary['failed']}")

    if summary['missing_in_optimized'] > 0:
        print(f"  Missing in optimized: {summary['missing_in_optimized']}")
    if summary['extra_in_optimized'] > 0:
        print(f"  Extra in optimized: {summary['extra_in_optimized']}")

    # Save results to JSON
    output_path = os.path.join(optimized_dir, 'comparison_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # Exit with error code if any failures
    if summary['failed'] > 0 or summary['missing_in_optimized'] > 0:
        print("\n\033[91mREGRESSION DETECTED!\033[0m")
        sys.exit(1)
    else:
        print("\n\033[92mAll tests passed!\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
