#!/usr/bin/env python3
"""
Memory benchmark for retarget_script optimization.

Measures memory usage during retarget processing to understand the
impact of KDTree caching (P1-1) on memory consumption.

Usage:
    python memory_benchmark.py [--iterations N]
"""

import subprocess
import sys
import time
import os
import json
import argparse

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required. Install with: pip install psutil")
    sys.exit(1)


def get_process_memory(pid):
    """Get memory usage of a process in MB."""
    try:
        process = psutil.Process(pid)
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def monitor_process(proc, interval=0.5):
    """Monitor a subprocess's memory usage."""
    memory_samples = []
    peak_rss = 0
    peak_vms = 0

    while proc.poll() is None:
        mem = get_process_memory(proc.pid)
        if mem:
            memory_samples.append({
                'timestamp': time.time(),
                'rss_mb': mem['rss_mb'],
                'vms_mb': mem['vms_mb'],
            })
            peak_rss = max(peak_rss, mem['rss_mb'])
            peak_vms = max(peak_vms, mem['vms_mb'])
        time.sleep(interval)

    return {
        'samples': memory_samples,
        'peak_rss_mb': peak_rss,
        'peak_vms_mb': peak_vms,
        'sample_count': len(memory_samples),
    }


def run_benchmark(blender_path, script_path, args, description):
    """Run a single benchmark and measure memory."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")

    cmd = [blender_path, '--background', '--python', script_path, '--'] + args

    start_time = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    # Monitor memory while process runs
    memory_data = monitor_process(proc, interval=0.5)

    # Wait for completion
    stdout, _ = proc.communicate()
    duration = time.time() - start_time

    return {
        'description': description,
        'duration_sec': duration,
        'return_code': proc.returncode,
        'peak_rss_mb': memory_data['peak_rss_mb'],
        'peak_vms_mb': memory_data['peak_vms_mb'],
        'sample_count': memory_data['sample_count'],
        'memory_samples': memory_data['samples'],
    }


def main():
    parser = argparse.ArgumentParser(description='Memory benchmark for retarget optimization')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations')
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blender_path = os.path.join(
        base_dir,
        'MochFitter-unity-addon', 'BlenderTools', 'blender-4.0.2-windows-x64', 'blender.exe'
    )
    script_path = os.path.join(
        base_dir,
        'MochFitter-unity-addon', 'BlenderTools', 'blender-4.0.2-windows-x64', 'dev', 'retarget_script2_14.py'
    )

    # Check paths
    if not os.path.exists(blender_path):
        # Try alternative path
        blender_path = r'D:\vrchat\mao_avator - mochifitter\BlenderTools\blender-4.0.2-windows-x64\blender.exe'

    if not os.path.exists(blender_path):
        print(f"ERROR: Blender not found at {blender_path}")
        sys.exit(1)

    print(f"Blender: {blender_path}")
    print(f"Script: {script_path}")

    # Test arguments (beryl_to_mao preset)
    test_args = [
        '--source=D:/Codes/MochiFitter-BlenderAddon-kai/MochFitter-unity-addon/OutfitRetargetingSystem/Editor/SourceOutfits/beryl_costume.fbx',
        '--target=D:/Codes/MochiFitter-BlenderAddon-kai/MochFitter-unity-addon/OutfitRetargetingSystem/Editor/Template.fbx',
        '--target=D:/Codes/MochiFitter-BlenderAddon-kai/MochFitter-unity-addon/OutfitRetargetingSystem/Editor/TargetAvatars/mao.fbx',
        '--config=D:/Codes/MochiFitter-BlenderAddon-kai/MochFitter-unity-addon/OutfitRetargetingSystem/Editor/config_beryl2template.json',
        '--config=D:/Codes/MochiFitter-BlenderAddon-kai/MochFitter-unity-addon/OutfitRetargetingSystem/Editor/config_template2mao.json',
        '--init-pose=D:/Codes/MochiFitter-BlenderAddon-kai/MochFitter-unity-addon/OutfitRetargetingSystem/Outputs/empty_pose.json',
        '--hips-position=0.00000000,0.00955725,0.93028500',
        '--target-meshes=Costume_Body,Costume_Frill_Arm,Costume_Frill_Hip,Costume_Gloves,Costume_Neck,Costume_Socks,HighHeel',
        '--output=D:/Codes/MochiFitter-BlenderAddon-kai/MochFitter-unity-addon/OutfitRetargetingSystem/Outputs/memory_benchmark_test.fbx',
    ]

    results = []

    for i in range(args.iterations):
        print(f"\n>>> Iteration {i+1}/{args.iterations}")
        result = run_benchmark(
            blender_path,
            script_path,
            test_args,
            f"Beryl → Template → mao (iteration {i+1})"
        )
        results.append(result)

        print(f"\nResults for iteration {i+1}:")
        print(f"  Duration: {result['duration_sec']:.2f}s")
        print(f"  Peak RSS: {result['peak_rss_mb']:.1f} MB")
        print(f"  Peak VMS: {result['peak_vms_mb']:.1f} MB")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    avg_duration = sum(r['duration_sec'] for r in results) / len(results)
    avg_peak_rss = sum(r['peak_rss_mb'] for r in results) / len(results)
    avg_peak_vms = sum(r['peak_vms_mb'] for r in results) / len(results)
    max_peak_rss = max(r['peak_rss_mb'] for r in results)
    max_peak_vms = max(r['peak_vms_mb'] for r in results)

    print(f"Iterations: {len(results)}")
    print(f"Average Duration: {avg_duration:.2f}s")
    print(f"Average Peak RSS: {avg_peak_rss:.1f} MB")
    print(f"Average Peak VMS: {avg_peak_vms:.1f} MB")
    print(f"Max Peak RSS: {max_peak_rss:.1f} MB")
    print(f"Max Peak VMS: {max_peak_vms:.1f} MB")

    # Save results
    output_path = os.path.join(base_dir, 'tests', 'memory_benchmark_results.json')
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'iterations': len(results),
        'avg_duration_sec': avg_duration,
        'avg_peak_rss_mb': avg_peak_rss,
        'avg_peak_vms_mb': avg_peak_vms,
        'max_peak_rss_mb': max_peak_rss,
        'max_peak_vms_mb': max_peak_vms,
        'results': [{k: v for k, v in r.items() if k != 'memory_samples'} for r in results],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
