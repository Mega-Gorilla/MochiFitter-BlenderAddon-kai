#!/usr/bin/env python3
"""
generate_patcher.py - smoothing_processor.py 最適化パッチDLL生成スクリプト

このスクリプトは以下を自動で行います:
1. オリジナル版と最適化版の smoothing_processor.py を差分分析
2. パッチ適用ロジックを含む C# ソースコードを生成
3. csc.exe を使用して DLL をビルド

使用方法:
  python generate_patcher.py --original <original_file> --optimized <optimized_file>
  python generate_patcher.py --use-manual-patches  # 手動定義パッチを使用

出力:
  - build/MochiFitterPatcher.cs
  - dist/MochiFitterPatcher.dll
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# バージョン情報
VERSION = "1.0.0"
DLL_NAME = "MochiFitterPatcher"


@dataclass
class PatchDefinition:
    """パッチ定義"""
    patch_id: str
    description: str
    original_code: str
    optimized_code: str
    line_hint: Optional[int] = None  # 検索開始行のヒント


# 手動定義パッチ（差分分析が困難な場合のフォールバック）
# 注意: インデントはオリジナルファイルと完全に一致させる必要がある
MANUAL_PATCHES: List[PatchDefinition] = [
    PatchDefinition(
        patch_id="blend_weights_vectorize_1",
        description="ウェイト合成ループのベクトル化 (1箇所目, line 402-406)",
        # 28スペース + コード
        # 注意: 403行目の空行には28スペースのトレーリングホワイトスペースがある
        original_code="                            final_weights = np.zeros(len(vertex_coords), dtype=np.float32)\n                            \n                            for i in range(len(vertex_coords)):\n                                blend_factor = mask_weights[i]\n                                final_weights[i] = original_weights[i] * (1.0 - blend_factor) + smoothed_weights[i] * blend_factor",
        optimized_code="                            # MochiFitter-Kai Optimized: NumPy vectorized blending (2544x faster)\n                            final_weights = original_weights * (1.0 - mask_weights) + smoothed_weights * mask_weights",
        line_hint=402
    ),
    PatchDefinition(
        patch_id="blend_weights_vectorize_2",
        description="ウェイト合成ループのベクトル化 (2箇所目, line 477-480)",
        # 24スペース + コード
        original_code="                        final_weights = np.zeros(len(vertex_coords), dtype=np.float32)\n                        for i in range(len(vertex_coords)):\n                            blend_factor = mask_weights[i]\n                            final_weights[i] = original_weights[i] * (1.0 - blend_factor) + smoothed_weights[i] * blend_factor",
        optimized_code="                        # MochiFitter-Kai Optimized: NumPy vectorized blending (2544x faster)\n                        final_weights = original_weights * (1.0 - mask_weights) + smoothed_weights * mask_weights",
        line_hint=477
    ),
]


def find_csc_exe() -> Optional[Path]:
    """
    .NET Framework の csc.exe を検索

    Returns:
        csc.exe のパス、見つからない場合は None
    """
    # .NET Framework のパス候補
    framework_paths = [
        Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
        Path(r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe"),
    ]

    for path in framework_paths:
        if path.exists():
            return path

    # PATH から検索
    try:
        result = subprocess.run(
            ["where", "csc.exe"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return Path(result.stdout.strip().split('\n')[0])
    except Exception:
        pass

    return None


def escape_csharp_string(s: str) -> str:
    """C# の文字列リテラル用にエスケープ"""
    # バックスラッシュを先にエスケープ
    s = s.replace("\\", "\\\\")
    # ダブルクォートをエスケープ
    s = s.replace('"', '\\"')
    # 改行をエスケープ
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s


def analyze_diff(original_path: Path, optimized_path: Path) -> List[PatchDefinition]:
    """
    オリジナル版と最適化版の差分を分析してパッチ定義を生成

    現状は単純な行ベース比較。将来的にはAST比較などの高度な手法を検討。
    """
    patches = []

    with open(original_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    with open(optimized_path, 'r', encoding='utf-8') as f:
        optimized_lines = f.readlines()

    # 単純な差分検出（行単位）
    # TODO: より高度な差分分析を実装

    # 現時点では、既知のパターン（forループのベクトル化）を検出
    in_diff_block = False
    diff_start = -1
    original_block = []

    for i, line in enumerate(original_lines):
        # forループによるウェイト合成パターンを検出
        if "for i in range(len(vertex_coords)):" in line:
            # 次の数行を確認
            if i + 2 < len(original_lines):
                next_lines = "".join(original_lines[i:i+3])
                if "blend_factor = mask_weights[i]" in next_lines:
                    # パターン検出成功
                    # 既にMANUAL_PATCHESに定義済みなのでスキップ
                    pass

    # 差分が検出できなかった場合は空リストを返す
    return patches


def generate_csharp_source(patches: List[PatchDefinition], output_path: Path) -> None:
    """
    パッチ定義から C# ソースコードを生成
    """

    # パッチ配列の生成
    patch_entries = []
    for p in patches:
        entry = f'''            new PatchEntry {{
                PatchId = "{p.patch_id}",
                Description = "{escape_csharp_string(p.description)}",
                OriginalCode = "{escape_csharp_string(p.original_code)}",
                OptimizedCode = "{escape_csharp_string(p.optimized_code)}",
                LineHint = {p.line_hint if p.line_hint else -1}
            }}'''
        patch_entries.append(entry)

    patches_array = ",\n".join(patch_entries)

    # パッチ数を取得
    total_patches = len(patches)

    # .NET Framework 4.0互換のC#コードを生成（文字列補間を使用しない）
    cs_source = '''// ============================================
// MochiFitter-Kai Patcher DLL
// Auto-generated by generate_patcher.py
// Version: ''' + VERSION + '''
// ============================================
//
// このDLLはsmoothing_processor.pyに最適化パッチを適用します。
//
// 使用方法:
//   MochiFitterPatcher.ApplyPatches(filePath);   // パッチ適用
//   MochiFitterPatcher.RemovePatches(filePath);  // パッチ削除
//   MochiFitterPatcher.IsPatched(filePath);      // 適用状態確認
// ============================================

using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace MochiFitterKai
{
    /// <summary>
    /// パッチエントリ定義
    /// </summary>
    public class PatchEntry
    {
        public string PatchId { get; set; }
        public string Description { get; set; }
        public string OriginalCode { get; set; }
        public string OptimizedCode { get; set; }
        public int LineHint { get; set; }
    }

    /// <summary>
    /// パッチ適用結果
    /// </summary>
    public class PatchResult
    {
        public bool Success { get; set; }
        public int PatchesApplied { get; set; }
        public int TotalPatches { get; set; }
        public List<string> AppliedPatchIds { get; set; }
        public List<string> SkippedPatchIds { get; set; }
        public List<string> Messages { get; set; }
        public string Error { get; set; }

        public PatchResult()
        {
            Messages = new List<string>();
            AppliedPatchIds = new List<string>();
            SkippedPatchIds = new List<string>();
        }
    }

    /// <summary>
    /// smoothing_processor.py 最適化パッチャー
    /// </summary>
    public static class MochiFitterPatcher
    {
        /// <summary>
        /// パッチャーバージョン
        /// </summary>
        public const string Version = "''' + VERSION + '''";

        /// <summary>
        /// 総パッチ数
        /// </summary>
        public const int TotalPatchCount = ''' + str(total_patches) + ''';

        /// <summary>
        /// 最適化マーカー（ファイルヘッダー）
        /// </summary>
        private const string OptimizationMarker = "# MochiFitter-Kai Optimized";

        /// <summary>
        /// 完全適用マーカー
        /// </summary>
        private const string FullyAppliedMarker = "# Patches Applied: ''' + str(total_patches) + '''/''' + str(total_patches) + '''";

        /// <summary>
        /// パッチ定義リスト
        /// </summary>
        private static readonly PatchEntry[] Patches = new PatchEntry[]
        {
''' + patches_array + '''
        };

        /// <summary>
        /// 行末空白を正規化（各行の末尾の空白を削除）
        /// </summary>
        private static string NormalizeTrailingWhitespace(string content)
        {
            string[] lines = content.Split(new[] { "\\n" }, StringSplitOptions.None);
            for (int i = 0; i < lines.Length; i++)
            {
                lines[i] = lines[i].TrimEnd();
            }
            return string.Join("\\n", lines);
        }

        /// <summary>
        /// ファイルがパッチ適用済みかどうかを確認
        /// </summary>
        /// <param name="filePath">smoothing_processor.py のパス</param>
        /// <returns>パッチ適用済みの場合 true</returns>
        public static bool IsPatched(string filePath)
        {
            if (!File.Exists(filePath))
                return false;

            try
            {
                string content = File.ReadAllText(filePath, Encoding.UTF8);
                return content.Contains(OptimizationMarker);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// ファイルが完全にパッチ適用済みかどうかを確認
        /// </summary>
        /// <param name="filePath">smoothing_processor.py のパス</param>
        /// <returns>全パッチ適用済みの場合 true</returns>
        public static bool IsFullyPatched(string filePath)
        {
            if (!File.Exists(filePath))
                return false;

            try
            {
                string content = File.ReadAllText(filePath, Encoding.UTF8);
                return content.Contains(FullyAppliedMarker);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// パッチを適用
        /// </summary>
        /// <param name="filePath">smoothing_processor.py のパス</param>
        /// <returns>適用結果</returns>
        public static PatchResult ApplyPatches(string filePath)
        {
            var result = new PatchResult();
            result.TotalPatches = TotalPatchCount;

            if (!File.Exists(filePath))
            {
                result.Success = false;
                result.Error = "ファイルが見つかりません: " + filePath;
                return result;
            }

            try
            {
                string content = File.ReadAllText(filePath, Encoding.UTF8);

                // 改行コードを正規化（CRLF → LF）
                content = content.Replace("\\r\\n", "\\n").Replace("\\r", "\\n");

                // 既に完全にパッチ適用済みの場合
                if (content.Contains(FullyAppliedMarker))
                {
                    result.Success = true;
                    result.PatchesApplied = TotalPatchCount;
                    result.Messages.Add("既に全パッチ適用済みです");
                    return result;
                }

                // 部分的に適用済みの場合は警告（ヘッダーがあるが完全適用ではない）
                if (content.Contains(OptimizationMarker) && !content.Contains(FullyAppliedMarker))
                {
                    result.Messages.Add("警告: 部分的にパッチが適用されています。再適用を試みます...");
                    // ヘッダーを削除して再適用を試みる
                    int headerEnd = content.IndexOf("# ============================================\\n",
                        content.IndexOf(OptimizationMarker));
                    if (headerEnd > 0)
                    {
                        headerEnd = content.IndexOf("\\n", headerEnd) + 1;
                        content = content.Substring(headerEnd);
                    }
                }

                // バックアップを作成（まだ存在しない場合のみ）
                string backupPath = filePath + ".bak";
                if (!File.Exists(backupPath))
                {
                    // オリジナルファイルを読み込んでバックアップ
                    string originalContent = File.ReadAllText(filePath, Encoding.UTF8);
                    File.WriteAllText(backupPath, originalContent, Encoding.UTF8);
                    result.Messages.Add("バックアップを作成: " + backupPath);
                }

                // 行末空白を正規化したコンテンツを作成（マッチング用）
                string normalizedContent = NormalizeTrailingWhitespace(content);

                // パッチを適用
                string modifiedContent = content;
                int appliedCount = 0;

                foreach (var patch in Patches)
                {
                    // 正規化されたパターンでマッチング
                    string normalizedOriginal = NormalizeTrailingWhitespace(patch.OriginalCode);

                    if (normalizedContent.Contains(normalizedOriginal))
                    {
                        // 元のコンテンツで置換を試みる
                        if (modifiedContent.Contains(patch.OriginalCode))
                        {
                            modifiedContent = modifiedContent.Replace(
                                patch.OriginalCode,
                                patch.OptimizedCode
                            );
                        }
                        else
                        {
                            // 正規化されたバージョンで置換
                            modifiedContent = NormalizeTrailingWhitespace(modifiedContent);
                            modifiedContent = modifiedContent.Replace(
                                normalizedOriginal,
                                patch.OptimizedCode
                            );
                        }
                        result.Messages.Add("パッチ適用: " + patch.Description);
                        result.AppliedPatchIds.Add(patch.PatchId);
                        appliedCount++;

                        // normalizedContentも更新
                        normalizedContent = NormalizeTrailingWhitespace(modifiedContent);
                    }
                    else
                    {
                        result.Messages.Add("パッチスキップ（コード不一致）: " + patch.Description);
                        result.SkippedPatchIds.Add(patch.PatchId);
                    }
                }

                // ヘッダーマーカーを追加（全パッチ適用時のみ完全適用マーカー）
                if (appliedCount > 0)
                {
                    string patchStatus = appliedCount.ToString() + "/" + TotalPatchCount.ToString();
                    string header = "# ============================================" + Environment.NewLine +
                        "# MochiFitter-Kai Optimized" + Environment.NewLine +
                        "# Version: " + Version + Environment.NewLine +
                        "# Patches Applied: " + patchStatus + Environment.NewLine +
                        "# DO NOT REMOVE THIS HEADER" + Environment.NewLine +
                        "# ============================================" + Environment.NewLine;
                    modifiedContent = header + modifiedContent;
                }

                // ファイルを書き込み
                File.WriteAllText(filePath, modifiedContent, Encoding.UTF8);

                result.Success = appliedCount > 0;
                result.PatchesApplied = appliedCount;

                if (appliedCount == 0)
                {
                    result.Error = "適用可能なパッチがありませんでした（コードが異なる可能性があります）";
                }
                else if (appliedCount < TotalPatchCount)
                {
                    result.Messages.Add("警告: 一部のパッチのみ適用されました (" + appliedCount.ToString() + "/" + TotalPatchCount.ToString() + ")");
                }
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Error = "パッチ適用中にエラー: " + ex.Message;
            }

            return result;
        }

        /// <summary>
        /// パッチを削除（バックアップから復元）
        /// </summary>
        /// <param name="filePath">smoothing_processor.py のパス</param>
        /// <returns>削除結果</returns>
        public static PatchResult RemovePatches(string filePath)
        {
            var result = new PatchResult();
            result.TotalPatches = TotalPatchCount;

            string backupPath = filePath + ".bak";

            if (!File.Exists(backupPath))
            {
                result.Success = false;
                result.Error = "バックアップファイルが見つかりません: " + backupPath;
                return result;
            }

            try
            {
                // バックアップから復元
                File.Copy(backupPath, filePath, true);
                result.Success = true;
                result.Messages.Add("バックアップから復元しました");
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Error = "復元中にエラー: " + ex.Message;
            }

            return result;
        }

        /// <summary>
        /// パッチ情報を取得
        /// </summary>
        /// <returns>パッチ定義のリスト</returns>
        public static PatchEntry[] GetPatchDefinitions()
        {
            return Patches;
        }
    }
}
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cs_source)

    print(f"[OK] C# ソースを生成: {output_path}")


def build_dll(cs_path: Path, output_path: Path) -> bool:
    """
    csc.exe を使用して DLL をビルド
    """
    csc = find_csc_exe()
    if csc is None:
        print("[ERROR] csc.exe が見つかりません")
        print("  .NET Framework 4.0+ がインストールされていることを確認してください")
        return False

    print(f"[INFO] Using csc.exe: {csc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # コンパイル実行
    cmd = [
        str(csc),
        "/target:library",
        f"/out:{output_path}",
        "/optimize+",
        "/nologo",
        str(cs_path)
    ]

    print(f"[INFO] Building: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            print(f"[ERROR] ビルド失敗:")
            print(result.stdout)
            print(result.stderr)
            return False

        print(f"[OK] DLL を生成: {output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] ビルド中にエラー: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="smoothing_processor.py 最適化パッチ DLL 生成スクリプト"
    )
    parser.add_argument(
        "--original",
        type=Path,
        help="オリジナル smoothing_processor.py のパス"
    )
    parser.add_argument(
        "--optimized",
        type=Path,
        help="最適化版 smoothing_processor.py のパス"
    )
    parser.add_argument(
        "--use-manual-patches",
        action="store_true",
        help="手動定義パッチを使用（差分分析をスキップ）"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="出力ディレクトリ（デフォルト: スクリプトと同じディレクトリ）"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="DLL ビルドをスキップ（C# ソースのみ生成）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MochiFitter-Kai Patcher Generator")
    print(f"Version: {VERSION}")
    print("=" * 60)

    # パッチ定義を取得
    if args.use_manual_patches:
        print("\n[INFO] 手動定義パッチを使用")
        patches = MANUAL_PATCHES
    elif args.original and args.optimized:
        print(f"\n[INFO] 差分分析: {args.original} vs {args.optimized}")
        if not args.original.exists():
            print(f"[ERROR] オリジナルファイルが見つかりません: {args.original}")
            sys.exit(1)
        if not args.optimized.exists():
            print(f"[ERROR] 最適化版ファイルが見つかりません: {args.optimized}")
            sys.exit(1)

        patches = analyze_diff(args.original, args.optimized)

        if not patches:
            print("[WARN] 差分分析でパッチを検出できませんでした")
            print("[INFO] 手動定義パッチにフォールバック")
            patches = MANUAL_PATCHES
    else:
        print("\n[INFO] パッチソース未指定、手動定義パッチを使用")
        patches = MANUAL_PATCHES

    print(f"\n[INFO] パッチ数: {len(patches)}")
    for p in patches:
        print(f"  - {p.patch_id}: {p.description}")

    # 出力パス
    build_dir = args.output_dir / "build"
    dist_dir = args.output_dir / "dist"

    cs_path = build_dir / f"{DLL_NAME}.cs"
    dll_path = dist_dir / f"{DLL_NAME}.dll"

    # C# ソース生成
    print(f"\n[INFO] C# ソースを生成中...")
    generate_csharp_source(patches, cs_path)

    # DLL ビルド
    if args.skip_build:
        print("\n[INFO] DLL ビルドをスキップ")
    else:
        print(f"\n[INFO] DLL をビルド中...")
        if build_dll(cs_path, dll_path):
            print("\n" + "=" * 60)
            print("生成完了!")
            print(f"  C# ソース: {cs_path}")
            print(f"  DLL:       {dll_path}")
            print("=" * 60)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
