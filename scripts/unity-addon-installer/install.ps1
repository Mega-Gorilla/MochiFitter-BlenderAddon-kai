# ============================================
# MochiFitter-Kai Optimization Installer
# PowerShell Installation Script
# ============================================

param(
    [switch]$Uninstall,
    [switch]$Help
)

# 設定
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DllPath = Join-Path $ScriptDir "dist\MochiFitterPatcher.dll"
$RetargetScriptSourcePath = Join-Path $ScriptDir "files\retarget_script2_14.py"

# VCC 設定ファイルパス
$VccSettingsPath = Join-Path $env:LOCALAPPDATA "VRChatCreatorCompanion\settings.json"

# 対象ファイルの相対パス
$SmoothingProcessorRelPath = "Assets\OutfitRetargetingSystem\Editor\smoothing_processor.py"
$RetargetScriptRelPath = "BlenderTools\blender-4.0.2-windows-x64\dev\retarget_script2_14.py"

# 最適化マーカー
$OptimizationMarker = "# MochiFitter-Kai Optimized"

# ============================================
# ヘルパー関数
# ============================================

function Write-Header {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host " MochiFitter-Kai Optimization Installer" -ForegroundColor Cyan
    Write-Host " Version 1.0.0" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-SuccessMessage {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-ErrorMessage {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-InfoMessage {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Yellow
}

function Write-WarningMessage {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor DarkYellow
}

function Show-Help {
    Write-Host @"
MochiFitter-Kai Optimization Installer

使用方法:
  install.bat                インストールモード（対話的）
  install.bat -Uninstall     アンインストールモード

オプション:
  -Help         このヘルプを表示
  -Uninstall    最適化パッチをアンインストール

説明:
  このインストーラーは MochiFitter Unity アドオンの smoothing_processor.py に
  最適化パッチを適用します。パッチ適用により、スムージング処理が約2500倍高速化されます。

必要条件:
  - VRChat Creator Companion (VCC) がインストールされていること
  - MochiFitter Unity アドオンが VCC プロジェクトにインストールされていること

"@
}

function Get-VccProjects {
    <#
    .SYNOPSIS
    VCC の settings.json から userProjects を取得
    #>

    if (-not (Test-Path $VccSettingsPath)) {
        Write-ErrorMessage "VCC settings.json が見つかりません: $VccSettingsPath"
        Write-InfoMessage "VRChat Creator Companion をインストールしてください"
        return @()
    }

    try {
        $settings = Get-Content $VccSettingsPath -Raw | ConvertFrom-Json
        if ($settings.userProjects) {
            return $settings.userProjects
        } else {
            Write-WarningMessage "userProjects が見つかりません"
            return @()
        }
    }
    catch {
        Write-ErrorMessage "settings.json の解析に失敗: $_"
        return @()
    }
}

function Get-MochiFitterStatus {
    <#
    .SYNOPSIS
    プロジェクト内の MochiFitter インストール状態を確認
    #>
    param([string]$ProjectPath)

    $result = @{
        HasMochiFitter = $false
        SmoothingProcessorPath = $null
        SmoothingProcessorOptimized = $false
        RetargetScriptPath = $null
        RetargetScriptOptimized = $false
    }

    # smoothing_processor.py を検索
    $smoothingPath = Join-Path $ProjectPath $SmoothingProcessorRelPath
    if (Test-Path $smoothingPath) {
        $result.HasMochiFitter = $true
        $result.SmoothingProcessorPath = $smoothingPath

        # 最適化マーカーを確認
        $content = Get-Content $smoothingPath -Raw -ErrorAction SilentlyContinue
        if ($content -and $content.Contains($OptimizationMarker)) {
            $result.SmoothingProcessorOptimized = $true
        }
    }

    # retarget_script2_14.py を検索（BlenderTools 内）
    # 複数のバージョンパスをチェック
    $blenderVersions = @("4.0.2", "4.1.0", "4.2.0", "4.3.0", "4.3.2")
    foreach ($ver in $blenderVersions) {
        $retargetPath = Join-Path $ProjectPath "BlenderTools\blender-$ver-windows-x64\dev\retarget_script2_14.py"
        if (Test-Path $retargetPath) {
            $result.RetargetScriptPath = $retargetPath

            $content = Get-Content $retargetPath -Raw -ErrorAction SilentlyContinue
            if ($content -and $content.Contains($OptimizationMarker)) {
                $result.RetargetScriptOptimized = $true
            }
            break
        }
    }

    return $result
}

function Show-BlenderWarning {
    <#
    .SYNOPSIS
    Blender 未設定警告を表示
    #>
    param([string]$ProjectPath)

    Write-Host ""
    Write-Host "┌─────────────────────────────────────────────────────────────┐" -ForegroundColor Yellow
    Write-Host "│ [警告] $ProjectPath" -ForegroundColor Yellow
    Write-Host "│                                                             │" -ForegroundColor Yellow
    Write-Host "│ MochiFitter は検出されましたが、Blender 連携が未設定です。 │" -ForegroundColor Yellow
    Write-Host "│                                                             │" -ForegroundColor Yellow
    Write-Host "│ 以下の手順で設定を完了してください:                        │" -ForegroundColor Yellow
    Write-Host "│ 1. Unity で該当プロジェクトを開く                          │" -ForegroundColor Yellow
    Write-Host "│ 2. メニュー: Tools → MochiFitter                           │" -ForegroundColor Yellow
    Write-Host "│ 3. MochiFitter ウィンドウで Blender Status が              │" -ForegroundColor Yellow
    Write-Host "│    `"Installed`" になっていることを確認                      │" -ForegroundColor Yellow
    Write-Host "│    (Not Installed の場合は Install ボタンをクリック)       │" -ForegroundColor Yellow
    Write-Host "│                                                             │" -ForegroundColor Yellow
    Write-Host "│ ※ Blender Status が Installed でない状態で最適化を適用     │" -ForegroundColor Yellow
    Write-Host "│   すると、後から Blender をインストールした際に            │" -ForegroundColor Yellow
    Write-Host "│   最適化が上書きされてしまいます。                         │" -ForegroundColor Yellow
    Write-Host "└─────────────────────────────────────────────────────────────┘" -ForegroundColor Yellow
    Write-Host ""
}

function Show-ProjectList {
    <#
    .SYNOPSIS
    プロジェクト一覧を表示し、選択させる
    #>
    param([array]$Projects)

    Write-Host ""
    Write-Host "検出されたプロジェクト:" -ForegroundColor White
    Write-Host "------------------------" -ForegroundColor Gray

    $validProjects = @()
    $warningProjects = @()
    $index = 0

    foreach ($project in $Projects) {
        if (-not (Test-Path $project)) {
            continue
        }

        $status = Get-MochiFitterStatus -ProjectPath $project
        $projectName = Split-Path $project -Leaf

        if ($status.HasMochiFitter) {
            # Blender 未設定警告の判定（OutfitRetargetingSystem あり、retarget_script なし）
            if (-not $status.RetargetScriptPath) {
                $warningProjects += $project
            }

            $validProjects += @{
                Index = $index
                Path = $project
                Status = $status
            }

            # 両ファイルが最適化済みの場合のみ「インストール済み」
            $isFullyOptimized = $status.SmoothingProcessorOptimized -and $status.RetargetScriptOptimized

            $statusText = if ($isFullyOptimized) {
                "[インストール済み]"
            } elseif ($status.SmoothingProcessorOptimized -or $status.RetargetScriptOptimized) {
                "[部分的に最適化]"
            } else {
                "[未インストール]"
            }

            $statusColor = if ($isFullyOptimized) { "Green" } elseif ($status.SmoothingProcessorOptimized -or $status.RetargetScriptOptimized) { "DarkYellow" } else { "Yellow" }

            Write-Host "  [$index] " -NoNewline -ForegroundColor Cyan
            Write-Host "$projectName " -NoNewline -ForegroundColor White
            Write-Host $statusText -ForegroundColor $statusColor
            Write-Host "      $project" -ForegroundColor Gray

            # Blender 未設定の場合は警告マーク
            if (-not $status.RetargetScriptPath) {
                Write-Host "      ⚠ Blender 未設定" -ForegroundColor DarkYellow
            }

            $index++
        }
    }

    # Blender 未設定警告を表示
    foreach ($warnProject in $warningProjects) {
        Show-BlenderWarning -ProjectPath $warnProject
    }

    if ($validProjects.Count -eq 0) {
        Write-WarningMessage "MochiFitter がインストールされているプロジェクトが見つかりませんでした"
        return $null
    }

    Write-Host ""
    Write-Host "  [q] 終了" -ForegroundColor DarkGray
    Write-Host ""

    do {
        $selection = Read-Host "プロジェクトを選択してください (0-$($validProjects.Count - 1))"

        if ($selection -eq 'q') {
            return $null
        }

        $selIndex = -1
        if ([int]::TryParse($selection, [ref]$selIndex)) {
            if ($selIndex -ge 0 -and $selIndex -lt $validProjects.Count) {
                return $validProjects[$selIndex]
            }
        }

        Write-Host "無効な選択です。もう一度入力してください。" -ForegroundColor Red
    } while ($true)
}

function Install-Optimization {
    <#
    .SYNOPSIS
    最適化パッチをインストール
    #>
    param([string]$ProjectPath, $Status)

    Write-Host ""
    Write-InfoMessage "最適化パッチをインストールしています..."

    $smoothingSuccess = $false
    $retargetSuccess = $false

    # ========================================
    # 1. smoothing_processor.py のパッチ適用
    # ========================================
    $smoothingPath = $Status.SmoothingProcessorPath

    if (-not $smoothingPath -or -not (Test-Path $smoothingPath)) {
        Write-ErrorMessage "smoothing_processor.py が見つかりません"
    } else {
        # DLL を読み込み
        if (-not (Test-Path $DllPath)) {
            Write-ErrorMessage "パッチャー DLL が見つかりません: $DllPath"
        } else {
            try {
                Add-Type -Path $DllPath -ErrorAction SilentlyContinue

                $result = [MochiFitterKai.MochiFitterPatcher]::ApplyPatches($smoothingPath)

                foreach ($msg in $result.Messages) {
                    Write-InfoMessage $msg
                }

                if ($result.Success) {
                    Write-SuccessMessage "smoothing_processor.py: パッチ適用完了"
                    $smoothingSuccess = $true
                } else {
                    Write-ErrorMessage "smoothing_processor.py: $($result.Error)"
                }
            }
            catch {
                Write-ErrorMessage "smoothing_processor.py パッチ適用中にエラー: $_"
            }
        }
    }

    # ========================================
    # 2. retarget_script2_14.py の完全置換
    # ========================================
    $retargetPath = $Status.RetargetScriptPath

    if (-not $retargetPath) {
        Write-WarningMessage "retarget_script2_14.py が見つかりません（Blender 未設定）"
        Write-InfoMessage "Blender 設定完了後に再度インストールしてください"
    } elseif (-not (Test-Path $RetargetScriptSourcePath)) {
        Write-ErrorMessage "最適化版 retarget_script2_14.py が見つかりません: $RetargetScriptSourcePath"
    } else {
        try {
            # バックアップ作成
            $backupPath = "$retargetPath.bak"
            if (-not (Test-Path $backupPath)) {
                Copy-Item -Path $retargetPath -Destination $backupPath -Force
                Write-InfoMessage "retarget_script2_14.py: バックアップ作成 → $backupPath"
            }

            # 最適化版で置換
            Copy-Item -Path $RetargetScriptSourcePath -Destination $retargetPath -Force
            Write-SuccessMessage "retarget_script2_14.py: 最適化版に置換完了"
            $retargetSuccess = $true
        }
        catch {
            Write-ErrorMessage "retarget_script2_14.py 置換中にエラー: $_"
        }
    }

    # 結果判定
    if ($smoothingSuccess -or $retargetSuccess) {
        return $true
    } else {
        return $false
    }
}

function Uninstall-Optimization {
    <#
    .SYNOPSIS
    最適化パッチをアンインストール
    #>
    param([string]$ProjectPath, $Status)

    Write-Host ""
    Write-InfoMessage "最適化パッチをアンインストールしています..."

    $smoothingSuccess = $false
    $retargetSuccess = $false

    # ========================================
    # 1. smoothing_processor.py のパッチ削除
    # ========================================
    $smoothingPath = $Status.SmoothingProcessorPath

    if (-not $smoothingPath -or -not (Test-Path $smoothingPath)) {
        Write-WarningMessage "smoothing_processor.py が見つかりません"
    } else {
        # DLL を読み込み
        if (-not (Test-Path $DllPath)) {
            Write-ErrorMessage "パッチャー DLL が見つかりません: $DllPath"
        } else {
            try {
                Add-Type -Path $DllPath -ErrorAction SilentlyContinue

                $result = [MochiFitterKai.MochiFitterPatcher]::RemovePatches($smoothingPath)

                foreach ($msg in $result.Messages) {
                    Write-InfoMessage $msg
                }

                if ($result.Success) {
                    Write-SuccessMessage "smoothing_processor.py: 復元完了"
                    $smoothingSuccess = $true
                } else {
                    Write-ErrorMessage "smoothing_processor.py: $($result.Error)"
                }
            }
            catch {
                Write-ErrorMessage "smoothing_processor.py 復元中にエラー: $_"
            }
        }
    }

    # ========================================
    # 2. retarget_script2_14.py の復元
    # ========================================
    $retargetPath = $Status.RetargetScriptPath

    if (-not $retargetPath) {
        Write-WarningMessage "retarget_script2_14.py が見つかりません（Blender 未設定）"
    } else {
        $backupPath = "$retargetPath.bak"

        if (-not (Test-Path $backupPath)) {
            Write-WarningMessage "retarget_script2_14.py: バックアップが見つかりません"
        } else {
            try {
                # バックアップから復元
                Copy-Item -Path $backupPath -Destination $retargetPath -Force
                Write-SuccessMessage "retarget_script2_14.py: バックアップから復元完了"
                $retargetSuccess = $true
            }
            catch {
                Write-ErrorMessage "retarget_script2_14.py 復元中にエラー: $_"
            }
        }
    }

    # 結果判定
    if ($smoothingSuccess -or $retargetSuccess) {
        return $true
    } else {
        return $false
    }
}

# ============================================
# メイン処理
# ============================================

Write-Header

if ($Help) {
    Show-Help
    exit 0
}

# VCC プロジェクト一覧を取得
$projects = Get-VccProjects
if ($projects.Count -eq 0) {
    Write-ErrorMessage "プロジェクトが見つかりませんでした"
    exit 1
}

Write-InfoMessage "VCC から $($projects.Count) 個のプロジェクトを検出しました"

# プロジェクト選択
$selected = Show-ProjectList -Projects $projects
if (-not $selected) {
    Write-InfoMessage "終了します"
    exit 0
}

Write-Host ""
Write-Host "選択されたプロジェクト:" -ForegroundColor White
Write-Host "  $($selected.Path)" -ForegroundColor Cyan
Write-Host ""

# インストール状態の判定
$isFullyOptimized = $selected.Status.SmoothingProcessorOptimized -and $selected.Status.RetargetScriptOptimized
$isPartiallyOptimized = $selected.Status.SmoothingProcessorOptimized -or $selected.Status.RetargetScriptOptimized

# インストール/アンインストール実行
if ($Uninstall) {
    if (-not $isPartiallyOptimized) {
        Write-WarningMessage "このプロジェクトには最適化パッチが適用されていません"
        exit 0
    }

    $confirm = Read-Host "最適化パッチをアンインストールしますか？ (y/N)"
    if ($confirm -ne 'y' -and $confirm -ne 'Y') {
        Write-InfoMessage "キャンセルしました"
        exit 0
    }

    $success = Uninstall-Optimization -ProjectPath $selected.Path -Status $selected.Status
} else {
    if ($isFullyOptimized) {
        Write-WarningMessage "このプロジェクトは既に最適化済みです"

        $confirm = Read-Host "再インストールしますか？ (y/N)"
        if ($confirm -ne 'y' -and $confirm -ne 'Y') {
            Write-InfoMessage "終了します"
            exit 0
        }
    } elseif ($isPartiallyOptimized) {
        Write-WarningMessage "このプロジェクトは部分的に最適化されています"
        Write-InfoMessage "すべてのファイルに最適化を適用します"
    }

    $success = Install-Optimization -ProjectPath $selected.Path -Status $selected.Status
}

Write-Host ""
if ($success) {
    Write-SuccessMessage "処理が完了しました"
    exit 0
} else {
    Write-ErrorMessage "処理に失敗しました"
    exit 1
}

