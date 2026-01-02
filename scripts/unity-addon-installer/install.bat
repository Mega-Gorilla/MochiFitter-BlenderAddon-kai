@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: ============================================
:: MochiFitter-Kai Optimization Installer
:: Unity Addon Optimization Patch Installer
:: ============================================

title MochiFitter-Kai Optimization Installer

:: PowerShell スクリプトを実行
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*

:: 終了コードを保持
set "EXIT_CODE=%ERRORLEVEL%"

:: 一時停止（ユーザーが結果を確認できるように）
echo.
pause

exit /b %EXIT_CODE%
