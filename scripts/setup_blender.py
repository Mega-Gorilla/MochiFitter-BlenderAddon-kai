#!/usr/bin/env python3
"""
setup_blender.py - Blender自動セットアップスクリプト

ローカル開発環境でのBlenderバージョン管理・互換性テストを容易にします。

使用例:
    python scripts/setup_blender.py                    # デフォルト (4.0.2) をインストール
    python scripts/setup_blender.py --version 4.2.0    # 特定バージョンをインストール
    python scripts/setup_blender.py --list-versions    # 利用可能バージョン一覧
    python scripts/setup_blender.py --clean            # キャッシュ削除

配置先:
    MochFitter-unity-addon/BlenderTools/blender-{version}-{platform}/
"""

import argparse
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

# =============================================================================
# Constants
# =============================================================================

# デフォルトバージョン (もちふぃった～公式推奨)
DEFAULT_VERSION = "4.0.2"

# スクリプトのルートディレクトリ
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Blender配置先ディレクトリ
BLENDER_TOOLS_DIR = PROJECT_ROOT / "MochFitter-unity-addon" / "BlenderTools"

# サポートするBlenderバージョン一覧
SUPPORTED_VERSIONS = [
    "4.0.2",
    "4.1.0",
    "4.1.1",
    "4.2.0",
    "4.2.1",
    "4.2.2",
    "4.2.3",
    "4.3.0",
    "4.3.1",
    "4.3.2",
]

# ダウンロードURLベース
BLENDER_DOWNLOAD_BASE = "https://download.blender.org/release"


# =============================================================================
# Platform Detection
# =============================================================================

def get_platform_info() -> Tuple[str, str, str]:
    """
    現在のプラットフォーム情報を取得する。

    Returns:
        Tuple[str, str, str]: (platform_name, arch, extension)
            - platform_name: "windows", "linux", "macos"
            - arch: "x64", "arm64"
            - extension: "zip", "tar.xz", "dmg"
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        return ("windows", "x64", "zip")
    elif system == "linux":
        return ("linux", "x64", "tar.xz")
    elif system == "darwin":
        # macOS: Apple Silicon (arm64) or Intel (x64)
        if machine in ("arm64", "aarch64"):
            return ("macos", "arm64", "dmg")
        else:
            return ("macos", "x64", "dmg")
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def get_download_url(version: str) -> str:
    """
    Blenderダウンロード URL を生成する。

    Args:
        version: Blenderバージョン (例: "4.0.2")

    Returns:
        str: ダウンロード URL
    """
    platform_name, arch, ext = get_platform_info()

    # バージョンのメジャー.マイナー部分を取得 (例: "4.0.2" -> "4.0")
    major_minor = ".".join(version.split(".")[:2])

    # プラットフォーム別のファイル名
    if platform_name == "windows":
        filename = f"blender-{version}-windows-x64.zip"
    elif platform_name == "linux":
        filename = f"blender-{version}-linux-x64.tar.xz"
    elif platform_name == "macos":
        filename = f"blender-{version}-macos-{arch}.dmg"
    else:
        raise RuntimeError(f"Unsupported platform: {platform_name}")

    return f"{BLENDER_DOWNLOAD_BASE}/Blender{major_minor}/{filename}"


def get_install_dir(version: str) -> Path:
    """
    Blenderインストール先ディレクトリを取得する。

    Args:
        version: Blenderバージョン

    Returns:
        Path: インストール先ディレクトリ
    """
    platform_name, arch, _ = get_platform_info()

    if platform_name == "macos":
        dir_name = f"blender-{version}-macos-{arch}"
    else:
        dir_name = f"blender-{version}-{platform_name}-x64"

    return BLENDER_TOOLS_DIR / dir_name


def get_blender_executable(install_dir: Path) -> Path:
    """
    Blender実行ファイルのパスを取得する。

    Args:
        install_dir: インストール先ディレクトリ

    Returns:
        Path: Blender実行ファイルのパス
    """
    platform_name, _, _ = get_platform_info()

    if platform_name == "windows":
        return install_dir / "blender.exe"
    elif platform_name == "linux":
        return install_dir / "blender"
    elif platform_name == "macos":
        return install_dir / "Blender.app" / "Contents" / "MacOS" / "Blender"
    else:
        raise RuntimeError(f"Unsupported platform: {platform_name}")


# =============================================================================
# Download & Extract
# =============================================================================

def download_file(url: str, dest_path: Path, show_progress: bool = True) -> None:
    """
    URLからファイルをダウンロードする。

    Args:
        url: ダウンロード URL
        dest_path: 保存先パス
        show_progress: 進捗表示するか
    """
    print(f"Downloading: {url}")

    def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        if show_progress and total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        if show_progress:
            print()  # 改行
        print(f"Downloaded: {dest_path}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Download failed: {e.code} {e.reason}\n  URL: {url}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Download failed: {e.reason}\n  URL: {url}")


def extract_zip(archive_path: Path, extract_dir: Path) -> Path:
    """
    ZIPファイルを解凍する。

    Args:
        archive_path: アーカイブファイルパス
        extract_dir: 解凍先ディレクトリ

    Returns:
        Path: 解凍されたBlenderディレクトリ
    """
    print(f"Extracting: {archive_path}")

    with zipfile.ZipFile(archive_path, 'r') as zf:
        # 最初のエントリからトップレベルディレクトリ名を取得
        top_dir = zf.namelist()[0].split('/')[0]
        zf.extractall(extract_dir)

    extracted_path = extract_dir / top_dir
    print(f"Extracted to: {extracted_path}")
    return extracted_path


def extract_tarxz(archive_path: Path, extract_dir: Path) -> Path:
    """
    tar.xzファイルを解凍する。

    Args:
        archive_path: アーカイブファイルパス
        extract_dir: 解凍先ディレクトリ

    Returns:
        Path: 解凍されたBlenderディレクトリ
    """
    print(f"Extracting: {archive_path}")

    with tarfile.open(archive_path, 'r:xz') as tf:
        # 最初のメンバーからトップレベルディレクトリ名を取得
        top_dir = tf.getnames()[0].split('/')[0]
        tf.extractall(extract_dir)

    extracted_path = extract_dir / top_dir
    print(f"Extracted to: {extracted_path}")
    return extracted_path


def extract_dmg(archive_path: Path, extract_dir: Path) -> Path:
    """
    DMGファイルを解凍する (macOS)。

    Args:
        archive_path: アーカイブファイルパス
        extract_dir: 解凍先ディレクトリ

    Returns:
        Path: 解凍されたBlenderディレクトリ
    """
    import subprocess

    print(f"Mounting: {archive_path}")

    # DMGをマウント
    mount_point = Path(tempfile.mkdtemp())
    try:
        subprocess.run(
            ["hdiutil", "attach", str(archive_path), "-mountpoint", str(mount_point), "-quiet"],
            check=True
        )

        # Blender.app をコピー
        app_path = mount_point / "Blender.app"
        if not app_path.exists():
            # 別のディレクトリ構造の場合を探す
            for item in mount_point.iterdir():
                if item.name.endswith(".app"):
                    app_path = item
                    break

        dest_app = extract_dir / "Blender.app"
        shutil.copytree(app_path, dest_app)
        print(f"Copied to: {dest_app}")

    finally:
        # アンマウント
        subprocess.run(["hdiutil", "detach", str(mount_point), "-quiet"], check=False)
        shutil.rmtree(mount_point, ignore_errors=True)

    return extract_dir


# =============================================================================
# Main Operations
# =============================================================================

def install_blender(version: str, force: bool = False) -> Path:
    """
    指定バージョンのBlenderをインストールする。

    Args:
        version: Blenderバージョン
        force: 既存インストールを上書きするか

    Returns:
        Path: インストールされたBlender実行ファイルのパス
    """
    install_dir = get_install_dir(version)
    executable = get_blender_executable(install_dir)

    # 既にインストール済みかチェック
    if executable.exists() and not force:
        print(f"Blender {version} is already installed at: {install_dir}")
        print(f"  Executable: {executable}")
        print("  Use --force to reinstall.")
        return executable

    # BlenderToolsディレクトリを作成
    BLENDER_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    # ダウンロードURL取得
    url = get_download_url(version)
    platform_name, _, ext = get_platform_info()

    # 一時ディレクトリでダウンロード・解凍
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_name = f"blender-{version}.{ext}"
        archive_path = temp_path / archive_name

        # ダウンロード
        download_file(url, archive_path)

        # 解凍
        if ext == "zip":
            extracted_dir = extract_zip(archive_path, temp_path)
        elif ext == "tar.xz":
            extracted_dir = extract_tarxz(archive_path, temp_path)
        elif ext == "dmg":
            extracted_dir = extract_dmg(archive_path, temp_path)
        else:
            raise RuntimeError(f"Unsupported archive format: {ext}")

        # インストール先に移動
        if install_dir.exists():
            print(f"Removing existing installation: {install_dir}")
            shutil.rmtree(install_dir)

        print(f"Installing to: {install_dir}")
        shutil.move(str(extracted_dir), str(install_dir))

    # インストール確認
    if not executable.exists():
        raise RuntimeError(f"Installation failed: executable not found at {executable}")

    print()
    print("=" * 60)
    print(f"Blender {version} installed successfully!")
    print(f"  Location: {install_dir}")
    print(f"  Executable: {executable}")
    print()
    print("To use this Blender with run_retarget.py:")
    print(f'  set BLENDER_PATH="{executable}"')
    print("  python run_retarget.py --preset beryl_to_mao")
    print("=" * 60)

    return executable


def list_versions() -> None:
    """利用可能なBlenderバージョン一覧を表示する。"""
    print("Supported Blender versions:")
    print()

    for version in SUPPORTED_VERSIONS:
        install_dir = get_install_dir(version)
        executable = get_blender_executable(install_dir)

        if executable.exists():
            status = "[installed]"
        else:
            status = ""

        default_marker = "(default)" if version == DEFAULT_VERSION else ""
        print(f"  {version} {default_marker} {status}")

    print()
    print(f"Default version: {DEFAULT_VERSION} (もちふぃった～公式推奨)")
    print()
    print("Usage:")
    print(f"  python {Path(__file__).name} --version {DEFAULT_VERSION}")


def list_installed() -> None:
    """インストール済みBlender一覧を表示する。"""
    print("Installed Blender versions:")
    print()

    if not BLENDER_TOOLS_DIR.exists():
        print("  (none)")
        return

    found = False
    for item in sorted(BLENDER_TOOLS_DIR.iterdir()):
        if item.is_dir() and item.name.startswith("blender-"):
            executable = get_blender_executable(item)
            if executable.exists():
                found = True
                print(f"  {item.name}")
                print(f"    Executable: {executable}")

    if not found:
        print("  (none)")


def clean_cache() -> None:
    """BlenderToolsディレクトリを削除する。"""
    if not BLENDER_TOOLS_DIR.exists():
        print("BlenderTools directory does not exist. Nothing to clean.")
        return

    print(f"Removing: {BLENDER_TOOLS_DIR}")

    # 確認
    response = input("Are you sure you want to delete all installed Blenders? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    shutil.rmtree(BLENDER_TOOLS_DIR)
    print("Cleaned successfully.")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """メイン関数。"""
    parser = argparse.ArgumentParser(
        description="Blender自動セットアップスクリプト (もちふぃった～開発用)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
使用例:
  python {Path(__file__).name}                      デフォルト ({DEFAULT_VERSION}) をインストール
  python {Path(__file__).name} --version 4.2.0      特定バージョンをインストール
  python {Path(__file__).name} --list-versions      利用可能バージョン一覧
  python {Path(__file__).name} --list-installed     インストール済み一覧
  python {Path(__file__).name} --clean              全てのインストールを削除

配置先:
  {BLENDER_TOOLS_DIR}/
        """
    )

    parser.add_argument(
        "--version", "-v",
        type=str,
        default=DEFAULT_VERSION,
        help=f"インストールするBlenderバージョン (デフォルト: {DEFAULT_VERSION})"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="既存インストールを上書き"
    )
    parser.add_argument(
        "--list-versions", "-l",
        action="store_true",
        help="利用可能バージョン一覧を表示"
    )
    parser.add_argument(
        "--list-installed",
        action="store_true",
        help="インストール済みBlender一覧を表示"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="BlenderToolsディレクトリを削除"
    )

    args = parser.parse_args()

    try:
        if args.list_versions:
            list_versions()
            return 0

        if args.list_installed:
            list_installed()
            return 0

        if args.clean:
            clean_cache()
            return 0

        # バージョン検証
        if args.version not in SUPPORTED_VERSIONS:
            print(f"Error: Unsupported version '{args.version}'")
            print(f"Supported versions: {', '.join(SUPPORTED_VERSIONS)}")
            return 1

        # インストール実行
        install_blender(args.version, force=args.force)
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
