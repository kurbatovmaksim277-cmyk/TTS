@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run install_windows.bat first.
    pause
    exit /b 1
)

rem Default stable settings. You can edit these.
set TTS_SEED=12345
set TTS_DETERMINISTIC=1
set MAX_INPUT_CHARS=2000
set MAX_CHUNK_CHARS=220
set SILENCE_BETWEEN_CHUNKS_MS=80
rem Uncomment this if you want to force 1.7B model:
rem set QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base

".venv\Scripts\python.exe" "tts_simple_local.py"

pause
