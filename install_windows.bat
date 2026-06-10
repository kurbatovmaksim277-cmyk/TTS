@echo off
setlocal
cd /d "%~dp0"

echo === Omnix TTS Windows installer ===
echo This will create .venv and install dependencies.
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*

if errorlevel 1 (
    echo.
    echo Installation failed.
    pause
    exit /b 1
)

echo.
echo Installation finished.
pause
