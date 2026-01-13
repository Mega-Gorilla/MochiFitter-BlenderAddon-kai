#!/usr/bin/env python3
"""
MochiFitter Blender Addon - Release Build Script

This script creates a distributable ZIP file for the Blender addon.
The ZIP can be installed directly in Blender via Edit > Preferences > Add-ons > Install.

Cross-platform notes:
- The bundled psutil binary (.pyd) only works on Windows
- Linux/macOS users should install psutil via Blender's Python if memory monitoring is needed
- Core functionality works without psutil (graceful degradation)

Usage:
    python scripts/build_release.py [--output-dir DIR]

Output:
    MochiFitter-BlenderAddon-vX.Y.Z.zip
"""

import os
import sys
import re
import zipfile
import argparse
import shutil
from pathlib import Path
from datetime import datetime


# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ADDON_DIR = PROJECT_ROOT / "MochiFitter-BlenderAddon"

# Files/folders to include in the addon
INCLUDE_FILES = [
    "__init__.py",
    "SaveAndApplyFieldAuto.py",
    "rbf_multithread_processor.py",
    "LICENSE.txt",
]

# Bundled dependencies to include (from deps/)
BUNDLED_DEPS = [
    "psutil",
    "psutil-7.2.1.dist-info",
]

# Patterns to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "Thumbs.db",
    "*.whl",
]


def get_version_from_init() -> tuple:
    """Extract version tuple from __init__.py"""
    init_file = ADDON_DIR / "__init__.py"

    with open(init_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Match: "version": (0, 2, 15),
    match = re.search(r'"version"\s*:\s*\((\d+),\s*(\d+),\s*(\d+)\)', content)
    if match:
        return tuple(int(x) for x in match.groups())

    raise ValueError("Could not find version in __init__.py")


def format_version(version: tuple) -> str:
    """Format version tuple as string"""
    return f"{version[0]}.{version[1]}.{version[2]}"


def should_exclude(path: Path) -> bool:
    """Check if a path should be excluded"""
    name = path.name

    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern:
            return True

    return False


def add_directory_to_zip(zipf: zipfile.ZipFile, source_dir: Path, archive_base: str):
    """Recursively add a directory to the ZIP file"""
    for item in source_dir.rglob("*"):
        if should_exclude(item):
            continue

        if item.is_file():
            # Calculate archive path
            relative_path = item.relative_to(source_dir.parent)
            archive_path = str(Path(archive_base) / relative_path)

            print(f"  Adding: {archive_path}")
            zipf.write(item, archive_path)


def build_release(output_dir: Path = None) -> Path:
    """Build the release ZIP file"""

    # Get version
    version = get_version_from_init()
    version_str = format_version(version)

    print(f"\n{'='*60}")
    print(f"MochiFitter Blender Addon - Release Build")
    print(f"{'='*60}")
    print(f"Version: {version_str}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Determine output directory
    if output_dir is None:
        output_dir = PROJECT_ROOT / "dist"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ZIP filename
    zip_filename = f"MochiFitter-BlenderAddon-v{version_str}.zip"
    zip_path = output_dir / zip_filename

    # Remove existing ZIP if present
    if zip_path.exists():
        print(f"Removing existing: {zip_filename}")
        zip_path.unlink()

    # Addon folder name in ZIP (must be valid Python package name)
    addon_name = "MochiFitter-BlenderAddon"

    print(f"\nCreating: {zip_filename}")
    print(f"Output: {zip_path}")
    print()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

        # Add main addon files
        print("Adding main files:")
        for filename in INCLUDE_FILES:
            source_path = ADDON_DIR / filename
            if source_path.exists():
                archive_path = f"{addon_name}/{filename}"
                print(f"  Adding: {archive_path}")
                zipf.write(source_path, archive_path)
            else:
                print(f"  WARNING: {filename} not found!")

        # Add bundled dependencies
        print("\nAdding bundled dependencies:")
        deps_dir = ADDON_DIR / "deps"

        for dep_name in BUNDLED_DEPS:
            dep_path = deps_dir / dep_name
            if dep_path.exists() and dep_path.is_dir():
                for item in dep_path.rglob("*"):
                    if should_exclude(item):
                        continue
                    if item.is_file():
                        relative_path = item.relative_to(deps_dir)
                        archive_path = f"{addon_name}/deps/{relative_path}"
                        print(f"  Adding: {archive_path}")
                        zipf.write(item, archive_path)
            else:
                print(f"  WARNING: {dep_name} not found in deps/")

    # Print summary
    zip_size = zip_path.stat().st_size
    print(f"\n{'='*60}")
    print(f"Build Complete!")
    print(f"{'='*60}")
    print(f"Output: {zip_path}")
    print(f"Size: {zip_size / 1024:.1f} KB ({zip_size:,} bytes)")
    print()
    print("Installation:")
    print("  1. Open Blender")
    print("  2. Edit > Preferences > Add-ons")
    print("  3. Click 'Install...' and select the ZIP file")
    print("  4. Enable 'MochiFitter-Kai' addon")
    print()

    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Build MochiFitter Blender Addon release ZIP"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for ZIP file (default: dist/)"
    )

    args = parser.parse_args()

    try:
        zip_path = build_release(args.output_dir)
        print(f"Success! ZIP created at: {zip_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
