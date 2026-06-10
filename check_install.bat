@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run install_windows.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import torch, sys; print('Python:', sys.version); print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); print('Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NO GPU')"

pause
