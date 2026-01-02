# ============================================
# MochiFitter-Kai Patcher Test Script
# パッチ適用の検証テスト
# ============================================

param(
    [switch]$KeepTestDir,  # テストディレクトリを削除しない
    [switch]$Verbose       # 詳細出力
)

$ErrorActionPreference = "Stop"

# パス設定
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$DllPath = Join-Path $ScriptDir "dist\MochiFitterPatcher.dll"
$OriginalFile = Join-Path $RepoRoot "original_v34r\Editor\smoothing_processor.py"
$TestDir = Join-Path $ScriptDir "test_output"
$TestFile = Join-Path $TestDir "smoothing_processor.py"

# テスト結果
$TestResults = @{
    Passed = 0
    Failed = 0
    Tests = @()
}

# ============================================
# ヘルパー関数
# ============================================

function Write-TestHeader {
    param([string]$Title)
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host " $Title" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-TestResult {
    param(
        [string]$TestName,
        [bool]$Passed,
        [string]$Message = ""
    )

    $TestResults.Tests += @{
        Name = $TestName
        Passed = $Passed
        Message = $Message
    }

    if ($Passed) {
        $TestResults.Passed++
        Write-Host "[PASS] " -NoNewline -ForegroundColor Green
        Write-Host $TestName
    } else {
        $TestResults.Failed++
        Write-Host "[FAIL] " -NoNewline -ForegroundColor Red
        Write-Host $TestName
        if ($Message) {
            Write-Host "       $Message" -ForegroundColor Yellow
        }
    }
}

function Write-VerboseInfo {
    param([string]$Message)
    if ($Verbose) {
        Write-Host "  [DEBUG] $Message" -ForegroundColor Gray
    }
}

# ============================================
# テスト関数
# ============================================

function Test-Prerequisites {
    Write-TestHeader "Prerequisites Check"

    # DLLの存在確認
    $dllExists = Test-Path $DllPath
    Write-TestResult "DLL exists at expected path" $dllExists "Path: $DllPath"

    # オリジナルファイルの存在確認
    $originalExists = Test-Path $OriginalFile
    Write-TestResult "Original file exists" $originalExists "Path: $OriginalFile"

    return $dllExists -and $originalExists
}

function Initialize-TestEnvironment {
    Write-TestHeader "Test Environment Setup"

    # テストディレクトリ作成
    if (Test-Path $TestDir) {
        Remove-Item $TestDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $TestDir -Force | Out-Null
    Write-VerboseInfo "Created test directory: $TestDir"

    # オリジナルファイルをコピー
    Copy-Item $OriginalFile $TestFile -Force
    Write-VerboseInfo "Copied original file to test location"

    $copySuccess = Test-Path $TestFile
    Write-TestResult "Test file created" $copySuccess

    return $copySuccess
}

function Test-OriginalFileContent {
    Write-TestHeader "Original File Validation"

    $content = Get-Content $TestFile -Raw

    # オリジナルファイルにパッチ対象のコードが含まれているか確認
    $pattern1 = "for i in range(len(vertex_coords)):"
    $pattern2 = "blend_factor = mask_weights[i]"
    $pattern3 = "final_weights[i] = original_weights[i]"

    $hasPattern1 = $content.Contains($pattern1)
    $hasPattern2 = $content.Contains($pattern2)
    $hasPattern3 = $content.Contains($pattern3)

    Write-TestResult "Contains for-loop pattern" $hasPattern1
    Write-TestResult "Contains blend_factor assignment" $hasPattern2
    Write-TestResult "Contains element-wise assignment" $hasPattern3

    # 最適化マーカーがないことを確認
    $marker = "# MochiFitter-Kai Optimized"
    $hasMarker = $content.Contains($marker)
    Write-TestResult "Does NOT contain optimization marker" (-not $hasMarker) "File already has marker"

    # forループの出現回数をカウント
    $forLoopCount = ([regex]::Matches($content, "for i in range\(len\(vertex_coords\)\):")).Count
    Write-VerboseInfo "Found $forLoopCount for-loop instances"

    $blendFactorCount = ([regex]::Matches($content, "blend_factor = mask_weights\[i\]")).Count
    Write-VerboseInfo "Found $blendFactorCount blend_factor assignments"

    return $hasPattern1 -and $hasPattern2 -and $hasPattern3 -and (-not $hasMarker)
}

function Test-ApplyPatches {
    Write-TestHeader "Patch Application"

    try {
        # DLLを読み込み
        Add-Type -Path $DllPath
        Write-TestResult "DLL loaded successfully" $true
    }
    catch {
        Write-TestResult "DLL loaded successfully" $false $_.Exception.Message
        return $false
    }

    # パッチを適用
    try {
        $result = [MochiFitterKai.MochiFitterPatcher]::ApplyPatches($TestFile)

        Write-VerboseInfo "ApplyPatches returned:"
        Write-VerboseInfo "  Success: $($result.Success)"
        Write-VerboseInfo "  PatchesApplied: $($result.PatchesApplied)"
        Write-VerboseInfo "  Error: $($result.Error)"

        foreach ($msg in $result.Messages) {
            Write-VerboseInfo "  Message: $msg"
        }

        Write-TestResult "ApplyPatches succeeded" $result.Success $result.Error
        Write-TestResult "Patches applied count > 0" ($result.PatchesApplied -gt 0) "Applied: $($result.PatchesApplied)"

        return $result.Success
    }
    catch {
        Write-TestResult "ApplyPatches executed without exception" $false $_.Exception.Message
        return $false
    }
}

function Test-PatchedFileContent {
    Write-TestHeader "Patched File Validation"

    $content = Get-Content $TestFile -Raw

    # 最適化マーカーが追加されているか
    $marker = "# MochiFitter-Kai Optimized"
    $hasMarker = $content.Contains($marker)
    Write-TestResult "Contains optimization marker" $hasMarker

    # ベクトル化されたコードが含まれているか
    $vectorizedPattern = "original_weights * (1.0 - mask_weights) + smoothed_weights * mask_weights"
    $hasVectorized = $content.Contains($vectorizedPattern)
    Write-TestResult "Contains vectorized code" $hasVectorized

    # 最適化コメントが含まれているか
    $optimizationComment = "MochiFitter-Kai Optimized: NumPy vectorized blending"
    $hasComment = $content.Contains($optimizationComment)
    Write-TestResult "Contains optimization comment" $hasComment

    # 元のforループパターンがまだ残っているか確認（他の場所で使われている可能性）
    $originalPattern = "for i in range(len(vertex_coords)):"
    $remainingForLoops = ([regex]::Matches($content, [regex]::Escape($originalPattern))).Count

    # 元のblend_factorパターンの確認
    $blendFactorPattern = "blend_factor = mask_weights[i]"
    $remainingBlendFactor = ([regex]::Matches($content, [regex]::Escape($blendFactorPattern))).Count

    Write-VerboseInfo "Remaining for-loops: $remainingForLoops"
    Write-VerboseInfo "Remaining blend_factor patterns: $remainingBlendFactor"

    # パッチ対象の2箇所が置換されていることを確認
    # 元は2箇所あったはずなので、0になっているべき
    Write-TestResult "Removed blend_factor patterns (was 2, now 0)" ($remainingBlendFactor -eq 0) "Remaining: $remainingBlendFactor"

    return $hasMarker -and $hasVectorized
}

function Test-BackupCreated {
    Write-TestHeader "Backup Verification"

    $backupPath = "$TestFile.bak"
    $backupExists = Test-Path $backupPath

    Write-TestResult "Backup file created" $backupExists "Path: $backupPath"

    if ($backupExists) {
        # バックアップがオリジナルと同じ内容か確認
        $originalContent = Get-Content $OriginalFile -Raw
        $backupContent = Get-Content $backupPath -Raw

        $contentsMatch = $originalContent -eq $backupContent
        Write-TestResult "Backup matches original content" $contentsMatch
    }

    return $backupExists
}

function Test-IsPatchedMethod {
    Write-TestHeader "IsPatched Method Verification"

    try {
        # パッチ適用済みファイルをチェック
        $isPatched = [MochiFitterKai.MochiFitterPatcher]::IsPatched($TestFile)
        Write-TestResult "IsPatched returns true for patched file" $isPatched

        # オリジナルファイルをチェック
        $isOriginalPatched = [MochiFitterKai.MochiFitterPatcher]::IsPatched($OriginalFile)
        Write-TestResult "IsPatched returns false for original file" (-not $isOriginalPatched)

        # 存在しないファイルをチェック
        $nonExistent = [MochiFitterKai.MochiFitterPatcher]::IsPatched("C:\nonexistent\file.py")
        Write-TestResult "IsPatched returns false for non-existent file" (-not $nonExistent)

        return $isPatched -and (-not $isOriginalPatched)
    }
    catch {
        Write-TestResult "IsPatched method works" $false $_.Exception.Message
        return $false
    }
}

function Test-RemovePatches {
    Write-TestHeader "Patch Removal (Uninstall)"

    try {
        $result = [MochiFitterKai.MochiFitterPatcher]::RemovePatches($TestFile)

        Write-VerboseInfo "RemovePatches returned:"
        Write-VerboseInfo "  Success: $($result.Success)"
        Write-VerboseInfo "  Error: $($result.Error)"

        Write-TestResult "RemovePatches succeeded" $result.Success $result.Error

        if ($result.Success) {
            # ファイルがオリジナルに戻ったか確認
            $restoredContent = Get-Content $TestFile -Raw
            $originalContent = Get-Content $OriginalFile -Raw

            $contentsMatch = $restoredContent -eq $originalContent
            Write-TestResult "File restored to original content" $contentsMatch

            # IsPatchedがfalseを返すか
            $isPatched = [MochiFitterKai.MochiFitterPatcher]::IsPatched($TestFile)
            Write-TestResult "IsPatched returns false after removal" (-not $isPatched)
        }

        return $result.Success
    }
    catch {
        Write-TestResult "RemovePatches executed without exception" $false $_.Exception.Message
        return $false
    }
}

function Test-ReApplyPatches {
    Write-TestHeader "Re-apply Patches (After Uninstall)"

    try {
        $result = [MochiFitterKai.MochiFitterPatcher]::ApplyPatches($TestFile)

        Write-TestResult "Re-apply patches succeeded" $result.Success $result.Error
        Write-TestResult "Patches applied on re-apply" ($result.PatchesApplied -gt 0) "Applied: $($result.PatchesApplied)"

        return $result.Success
    }
    catch {
        Write-TestResult "Re-apply patches executed without exception" $false $_.Exception.Message
        return $false
    }
}

function Test-DoubleApply {
    Write-TestHeader "Double Apply Prevention"

    try {
        # 既にパッチ適用済みのファイルに再適用
        $result = [MochiFitterKai.MochiFitterPatcher]::ApplyPatches($TestFile)

        Write-VerboseInfo "Double apply result:"
        Write-VerboseInfo "  Success: $($result.Success)"
        Write-VerboseInfo "  PatchesApplied: $($result.PatchesApplied)"

        # 既に適用済みの場合、Successはtrueだが適用数は0
        $expectedBehavior = $result.Success -and ($result.PatchesApplied -eq 0 -or $result.Messages -contains "既にパッチ適用済みです")

        Write-TestResult "Handles double apply gracefully" $expectedBehavior

        return $expectedBehavior
    }
    catch {
        Write-TestResult "Double apply handled without exception" $false $_.Exception.Message
        return $false
    }
}

function Show-TestSummary {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host " Test Summary" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""

    $totalTests = $TestResults.Passed + $TestResults.Failed
    $passRate = if ($totalTests -gt 0) { [math]::Round(($TestResults.Passed / $totalTests) * 100, 1) } else { 0 }

    Write-Host "Total Tests: $totalTests"
    Write-Host "Passed: " -NoNewline
    Write-Host "$($TestResults.Passed)" -ForegroundColor Green
    Write-Host "Failed: " -NoNewline
    Write-Host "$($TestResults.Failed)" -ForegroundColor $(if ($TestResults.Failed -gt 0) { "Red" } else { "Green" })
    Write-Host "Pass Rate: $passRate%"
    Write-Host ""

    if ($TestResults.Failed -gt 0) {
        Write-Host "Failed Tests:" -ForegroundColor Red
        foreach ($test in $TestResults.Tests) {
            if (-not $test.Passed) {
                Write-Host "  - $($test.Name)" -ForegroundColor Red
                if ($test.Message) {
                    Write-Host "    $($test.Message)" -ForegroundColor Yellow
                }
            }
        }
        Write-Host ""
    }

    return $TestResults.Failed -eq 0
}

function Cleanup-TestEnvironment {
    if (-not $KeepTestDir) {
        if (Test-Path $TestDir) {
            Remove-Item $TestDir -Recurse -Force
            Write-Host "[INFO] Test directory cleaned up" -ForegroundColor Gray
        }
    } else {
        Write-Host "[INFO] Test directory preserved at: $TestDir" -ForegroundColor Yellow
    }
}

# ============================================
# メイン処理
# ============================================

Write-Host ""
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "#  MochiFitter-Kai Patcher Test Suite     #" -ForegroundColor Magenta
Write-Host "############################################" -ForegroundColor Magenta
Write-Host ""
Write-Host "Repository Root: $RepoRoot"
Write-Host "DLL Path: $DllPath"
Write-Host "Original File: $OriginalFile"
Write-Host "Test Directory: $TestDir"
Write-Host ""

try {
    # 1. 前提条件チェック
    if (-not (Test-Prerequisites)) {
        Write-Host ""
        Write-Host "[ABORT] Prerequisites not met. Cannot continue." -ForegroundColor Red
        exit 1
    }

    # 2. テスト環境セットアップ
    if (-not (Initialize-TestEnvironment)) {
        Write-Host ""
        Write-Host "[ABORT] Failed to initialize test environment." -ForegroundColor Red
        exit 1
    }

    # 3. オリジナルファイル検証
    Test-OriginalFileContent | Out-Null

    # 4. パッチ適用テスト
    Test-ApplyPatches | Out-Null

    # 5. パッチ適用後の検証
    Test-PatchedFileContent | Out-Null

    # 6. バックアップ検証
    Test-BackupCreated | Out-Null

    # 7. IsPatched メソッド検証
    Test-IsPatchedMethod | Out-Null

    # 8. 二重適用テスト
    Test-DoubleApply | Out-Null

    # 9. パッチ削除テスト
    Test-RemovePatches | Out-Null

    # 10. 再適用テスト
    Test-ReApplyPatches | Out-Null

    # サマリー表示
    $allPassed = Show-TestSummary

    # クリーンアップ
    Cleanup-TestEnvironment

    if ($allPassed) {
        Write-Host "[SUCCESS] All tests passed!" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "[FAILURE] Some tests failed." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host ""
    Write-Host "[ERROR] Unexpected error: $_" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    Cleanup-TestEnvironment
    exit 1
}

