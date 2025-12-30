#!/usr/bin/env python3
"""
Memory benchmark for retarget_script optimization v2.

Uses run_retarget.py wrapper for correct path handling.

Usage:
    python memory_benchmark_v2.py
"""

import subprocess
import sys
import time
import os
import json
import threading

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required. Install with: pip install psutil")
    sys.exit(1)


class MemoryMonitor:
    """Monitor memory usage of a process and its children."""

    def __init__(self, pid, interval=0.5):
        self.pid = pid
        self.interval = interval
        self.samples = []
        self.peak_rss = 0
        self.peak_vms = 0
        self._stop = False
        self._thread = None

    def _get_memory(self):
        """Get total memory usage of process tree."""
        try:
            process = psutil.Process(self.pid)
            children = process.children(recursive=True)

            total_rss = process.memory_info().rss
            total_vms = process.memory_info().vms

            for child in children:
                try:
                    mem = child.memory_info()
                    total_rss += mem.rss
                    total_vms += mem.vms
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            return total_rss / (1024 * 1024), total_vms / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None, None

    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop:
            rss, vms = self._get_memory()
            if rss is not None:
                self.samples.append({
                    'timestamp': time.time(),
                    'rss_mb': rss,
                    'vms_mb': vms,
                })
                self.peak_rss = max(self.peak_rss, rss)
                self.peak_vms = max(self.peak_vms, vms)
            time.sleep(self.interval)

    def start(self):
        """Start monitoring in background thread."""
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_results(self):
        """Get monitoring results."""
        return {
            'peak_rss_mb': self.peak_rss,
            'peak_vms_mb': self.peak_vms,
            'sample_count': len(self.samples),
            'samples': self.samples,
        }


def run_benchmark():
    """Run benchmark using run_retarget.py wrapper."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_retarget = os.path.join(
        base_dir,
        'MochFitter-unity-addon', 'OutfitRetargetingSystem', 'run_retarget.py'
    )

    if not os.path.exists(run_retarget):
        print(f"ERROR: run_retarget.py not found at {run_retarget}")
        sys.exit(1)

    print(f"Using wrapper: {run_retarget}")

    cmd = [sys.executable, run_retarget, '--preset', 'beryl_to_mao']

    print(f"\nStarting benchmark...")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=os.path.dirname(run_retarget)
    )

    # Start memory monitoring
    monitor = MemoryMonitor(proc.pid, interval=0.5)
    monitor.start()

    # Read output in real-time
    output_lines = []
    for line in proc.stdout:
        output_lines.append(line)
        # Print progress indicators
        if 'Duration:' in line or 'Status:' in line or '処理時間' in line:
            print(f"  {line.strip()}")

    proc.wait()
    monitor.stop()

    duration = time.time() - start_time
    mem_results = monitor.get_results()

    return {
        'duration_sec': duration,
        'return_code': proc.returncode,
        'peak_rss_mb': mem_results['peak_rss_mb'],
        'peak_vms_mb': mem_results['peak_vms_mb'],
        'sample_count': mem_results['sample_count'],
    }


def main():
    print("="*60)
    print("Memory Benchmark for P1-1 KDTree Caching")
    print("="*60)

    result = run_benchmark()

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Duration: {result['duration_sec']:.2f}s")
    print(f"Peak RSS (Resident Memory): {result['peak_rss_mb']:.1f} MB")
    print(f"Peak VMS (Virtual Memory): {result['peak_vms_mb']:.1f} MB")
    print(f"Memory Samples: {result['sample_count']}")
    print(f"Return Code: {result['return_code']}")

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'memory_benchmark_results.json'
    )

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'optimization': 'P1-0 + P1-1 (foreach_get + KDTree caching)',
        'duration_sec': result['duration_sec'],
        'peak_rss_mb': result['peak_rss_mb'],
        'peak_vms_mb': result['peak_vms_mb'],
        'sample_count': result['sample_count'],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
