#!/usr/bin/env python3
"""
CLI wrapper script for retarget_script2_14.py

This script simplifies running retarget_script2_14.py from the command line
without requiring Unity. It also records benchmark data (time, memory usage).

Usage:
    python run_retarget.py --preset beryl_to_mao
    python run_retarget.py --config custom_test.json

Requirements:
    - Blender 4.0+ installed
    - Required data files in Editor/ folder (see README.md)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Script directory (OutfitRetargetingSystem/)
SCRIPT_DIR = Path(__file__).parent.resolve()
EDITOR_DIR = SCRIPT_DIR / "Editor"
OUTPUTS_DIR = SCRIPT_DIR / "Outputs"
# BlenderTools/ ディレクトリ（setup_blender.py でインストールしたBlenderが配置される）
BLENDER_TOOLS_DIR = SCRIPT_DIR.parent / "BlenderTools"

# Preset configurations for common test cases
PRESETS = {
    "beryl_to_mao": {
        "description": "Beryl costume → Template → mao (chain processing)",
        "input": "Editor/TestingDatasets/Beryl_Costumes.fbx",
        "output": "Outputs/cli_test_beryl_to_mao.fbx",
        "base": "Editor/base_project.blend",
        "base_fbx": ["Editor/Template.fbx", "Editor/TargetAvatars/mao.fbx"],
        "config": ["Editor/config_beryl2template.json", "Editor/config_template2mao.json"],
        "init_pose": "Outputs/empty_pose.json",
        "hips_position": "0.00000000,0.00955725,0.93028500",
        "target_meshes": [
            "Costume_Body", "Costume_Frill_Arm", "Costume_Frill_Hip",
            "Costume_Gloves", "Costume_Neck", "Costume_Socks", "HighHeel"
        ],
        "blend_shapes": ["Highheel"],
        "blend_shape_values": [1.0],
        "no_subdivision": True,
    },
}


def _get_platform_dir_suffix() -> str:
    """Get platform-specific directory suffix for Blender installation."""
    import platform as pf
    system = pf.system().lower()
    machine = pf.machine().lower()

    if system == "windows":
        return "windows-x64"
    elif system == "linux":
        return "linux-x64"
    elif system == "darwin":
        # macOS: Apple Silicon (arm64) or Intel (x64)
        if machine in ("arm64", "aarch64"):
            return "macos-arm64"
        else:
            return "macos-x64"
    else:
        return "unknown"


def find_blender() -> Path:
    """Find Blender executable.

    Search order:
    1. BLENDER_PATH environment variable
    2. BlenderTools/ directory (setup_blender.py でインストールしたもの)
    3. Common installation paths (Windows/Linux/macOS)
    4. PATH
    """
    import shutil
    import platform as pf

    # Check BLENDER_PATH environment variable first
    env_path = os.environ.get("BLENDER_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Check BlenderTools/ for setup_blender.py installed versions
    # 優先順: デフォルトバージョン (4.0.2) > 最新バージョン
    if BLENDER_TOOLS_DIR.exists():
        system = pf.system().lower()
        platform_suffix = _get_platform_dir_suffix()

        # プラットフォーム別の実行ファイル名とパターン
        if system == "windows":
            exe_name = "blender.exe"
            pattern = "blender-*-windows-x64"
        elif system == "linux":
            exe_name = "blender"
            pattern = "blender-*-linux-x64"
        elif system == "darwin":
            exe_name = "Blender.app/Contents/MacOS/Blender"
            pattern = "blender-*-macos-*"
        else:
            exe_name = "blender"
            pattern = "blender-*"

        # デフォルトバージョン (4.0.2) を優先的にチェック
        default_dir = BLENDER_TOOLS_DIR / f"blender-4.0.2-{platform_suffix}"
        default_exe = default_dir / exe_name
        if default_exe.exists():
            return default_exe

        # BlenderTools/ 内の全バージョンをチェック (バージョン番号降順)
        blender_dirs = sorted(
            BLENDER_TOOLS_DIR.glob(pattern),
            key=lambda p: p.name,
            reverse=True  # 最新バージョン優先
        )
        for blender_dir in blender_dirs:
            exe_path = blender_dir / exe_name
            if exe_path.exists():
                return exe_path

    # Common installation paths by platform
    system = pf.system().lower()

    if system == "windows":
        common_paths = [
            Path(r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"),
            Path(r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"),
            Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
            Path(r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"),
        ]
        for p in common_paths:
            if p.exists():
                return p

    elif system == "linux":
        linux_paths = [
            Path("/usr/bin/blender"),
            Path("/snap/bin/blender"),
            Path.home() / ".local/bin/blender",
        ]
        for p in linux_paths:
            if p.exists():
                return p

    elif system == "darwin":
        # macOS: /Applications/Blender.app
        macos_paths = [
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
        ]
        for p in macos_paths:
            if p.exists():
                return p

    # Check PATH
    blender_path = shutil.which("blender")
    if blender_path:
        return Path(blender_path)

    raise FileNotFoundError(
        "Blender not found. Run 'python scripts/setup_blender.py' to install, "
        "or set BLENDER_PATH environment variable."
    )


def find_retarget_script() -> Path:
    """Find retarget_script2_14.py.

    Search order:
    1. RETARGET_SCRIPT_PATH environment variable
    2. BlenderTools/blender-*/dev/ directory (default 4.0.2 first, platform-aware)
    """
    # Check environment variable first
    env_path = os.environ.get("RETARGET_SCRIPT_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Check BlenderTools/blender-*/dev/ directories
    if BLENDER_TOOLS_DIR.exists():
        # デフォルトバージョン (4.0.2) を優先（プラットフォーム対応）
        platform_suffix = _get_platform_dir_suffix()
        default_script = BLENDER_TOOLS_DIR / f"blender-4.0.2-{platform_suffix}" / "dev" / "retarget_script2_14.py"
        if default_script.exists():
            return default_script

        # 他のバージョンを探索
        for blender_dir in sorted(BLENDER_TOOLS_DIR.glob("blender-*"), reverse=True):
            script_path = blender_dir / "dev" / "retarget_script2_14.py"
            if script_path.exists():
                return script_path

    raise FileNotFoundError(
        f"retarget_script2_14.py not found. "
        f"Set RETARGET_SCRIPT_PATH environment variable or place script in BlenderTools/blender-*/dev/"
    )


def resolve_path(path: str) -> Path:
    """Resolve path relative to SCRIPT_DIR."""
    p = Path(path)
    if p.is_absolute():
        return p
    return SCRIPT_DIR / path


def build_command(config: dict, blender_path: Path, script_path: Path) -> list:
    """Build Blender command line arguments."""
    cmd = [
        str(blender_path),
        "--background",
        "--python", str(script_path),
        "--",
    ]

    # Required arguments
    cmd.append(f"--input={resolve_path(config['input'])}")
    cmd.append(f"--output={resolve_path(config['output'])}")
    cmd.append(f"--base={resolve_path(config['base'])}")

    # Chain processing (semicolon-separated)
    base_fbx_paths = [str(resolve_path(p)) for p in config['base_fbx']]
    cmd.append(f"--base-fbx={';'.join(base_fbx_paths)}")

    config_paths = [str(resolve_path(p)) for p in config['config']]
    cmd.append(f"--config={';'.join(config_paths)}")

    cmd.append(f"--init-pose={resolve_path(config['init_pose'])}")

    # Optional arguments
    if 'hips_position' in config:
        cmd.append(f"--hips-position={config['hips_position']}")

    if 'target_meshes' in config:
        cmd.append(f"--target-meshes={';'.join(config['target_meshes'])}")

    if 'blend_shapes' in config and config['blend_shapes']:
        cmd.append(f"--blend-shapes={';'.join(config['blend_shapes'])}")

    if 'blend_shape_values' in config and config['blend_shape_values']:
        values = [str(v) for v in config['blend_shape_values']]
        cmd.append(f"--blend-shape-values={';'.join(values)}")

    if config.get('no_subdivision', False):
        cmd.append("--no-subdivision")

    if config.get('profile', False):
        cmd.append("--profile")

    return cmd


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def run_retarget(config: dict, verbose: bool = True) -> dict:
    """
    Run retarget_script2_14.py with the given configuration.

    Returns:
        dict: Benchmark results including time, memory, and status
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config.get("description", "custom"),
        "status": "unknown",
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "output_file": None,
        "output_size_bytes": None,
        "error": None,
    }

    try:
        blender_path = find_blender()
        script_path = find_retarget_script()

        if verbose:
            print(f"Blender: {blender_path}")
            print(f"Script: {script_path}")

        cmd = build_command(config, blender_path, script_path)

        if verbose:
            print(f"\nCommand:")
            print(" ".join(f'"{c}"' if " " in c else c for c in cmd[:5]))
            print("  [... arguments truncated ...]")
            print()

        # Ensure output directory exists
        output_path = resolve_path(config['output'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run the command
        results["start_time"] = datetime.now().isoformat()
        start = time.perf_counter()

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        end = time.perf_counter()
        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = round(end - start, 2)

        # Check results
        if process.returncode == 0:
            results["status"] = "success"
            if output_path.exists():
                results["output_file"] = str(output_path)
                results["output_size_bytes"] = output_path.stat().st_size
        else:
            results["status"] = "failed"
            results["error"] = process.stderr[-2000:] if process.stderr else "Unknown error"

        # Save stdout/stderr to log
        log_path = output_path.with_suffix(".log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Retarget CLI Log ===\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Duration: {results['duration_seconds']}s\n")
            f.write(f"Status: {results['status']}\n\n")
            f.write(f"=== STDOUT ===\n{process.stdout}\n")
            f.write(f"=== STDERR ===\n{process.stderr}\n")

        if verbose:
            print(f"Status: {results['status']}")
            print(f"Duration: {results['duration_seconds']}s")
            if results['output_size_bytes']:
                size_mb = results['output_size_bytes'] / 1024 / 1024
                print(f"Output: {output_path.name} ({size_mb:.2f} MB)")
            print(f"Log: {log_path}")

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        if verbose:
            print(f"Error: {e}")

    return results


def save_benchmark(results: dict, benchmark_file: Path = None):
    """Append benchmark results to a JSON file."""
    if benchmark_file is None:
        benchmark_file = OUTPUTS_DIR / "benchmark_results.json"

    # Load existing results
    existing = []
    if benchmark_file.exists():
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    # Append new results
    existing.append(results)

    # Save
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Benchmark saved to: {benchmark_file}")


def list_presets():
    """List available presets."""
    print("Available presets:")
    for name, config in PRESETS.items():
        print(f"  {name}: {config.get('description', 'No description')}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI wrapper for retarget_script2_14.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_retarget.py --preset beryl_to_mao
  python run_retarget.py --preset beryl_to_mao --benchmark
  python run_retarget.py --list-presets
        """
    )

    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        help="Use a predefined test configuration"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to custom configuration JSON file"
    )
    parser.add_argument(
        "--list-presets", "-l",
        action="store_true",
        help="List available presets"
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Save benchmark results to benchmark_results.json"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile profiling (outputs to profile_output/)"
    )

    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return 0

    # Get configuration
    if args.preset:
        config = PRESETS[args.preset].copy()
    elif args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        parser.print_help()
        print("\nError: Please specify --preset or --config")
        return 1

    # Enable profiling if requested
    if args.profile:
        config['profile'] = True
        print("PROFILING MODE ENABLED")

    # Run retarget
    print(f"Running: {config.get('description', 'custom config')}")
    print("=" * 60)

    results = run_retarget(config, verbose=not args.quiet)

    # Save benchmark if requested
    if args.benchmark:
        save_benchmark(results)

    return 0 if results["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
