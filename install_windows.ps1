param(
    [string]$PythonVersion = "3.12",
    [string]$TorchMode = "cu128",
    [switch]$SkipSoX,
    [switch]$ForceRecreateVenv
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host ""
Write-Host "=== Omnix TTS local installer ===" -ForegroundColor Cyan
Write-Host "Project dir: $PSScriptRoot"
Write-Host "Python target: $PythonVersion"
Write-Host "Torch mode: $TorchMode"
Write-Host ""

function Test-CommandExists {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Invoke-Step {
    param([string]$Title, [scriptblock]$Block)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Yellow
    & $Block
}

function Get-PythonLauncherCommand {
    param([string]$Version)

    if (Test-CommandExists "py") {
        try {
            & py "-$Version" --version | Out-Host
            return "py -$Version"
        } catch {
            Write-Host "Python $Version not found through py launcher." -ForegroundColor DarkYellow
        }
    }

    if (Test-CommandExists "python") {
        try {
            & python --version | Out-Host
            return "python"
        } catch {}
    }

    return $null
}

Invoke-Step "Checking Python" {
    $script:PythonCmd = Get-PythonLauncherCommand -Version $PythonVersion

    if (-not $script:PythonCmd) {
        Write-Host "Python $PythonVersion was not found." -ForegroundColor Red

        if (Test-CommandExists "winget") {
            Write-Host "Trying to install Python $PythonVersion with winget..."
            winget install -e --id "Python.Python.$PythonVersion"
            Write-Host ""
            Write-Host "Python was installed. Close this terminal, open it again, and run install_windows.bat once more." -ForegroundColor Green
            exit 0
        }

        throw "Python $PythonVersion not found and winget is unavailable. Install Python $PythonVersion manually."
    }
}

Invoke-Step "Creating virtual environment" {
    if ($ForceRecreateVenv -and (Test-Path ".venv")) {
        Write-Host "Removing existing .venv..."
        Remove-Item -Recurse -Force ".venv"
    }

    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        Write-Host "Creating .venv..."
        Invoke-Expression "$script:PythonCmd -m venv .venv"
    } else {
        Write-Host ".venv already exists."
    }

    & ".\.venv\Scripts\python.exe" --version
}

Invoke-Step "Upgrading pip tooling" {
    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel --default-timeout=1000
}

Invoke-Step "Installing PyTorch" {
    & ".\.venv\Scripts\python.exe" -m pip uninstall -y torch torchvision torchaudio | Out-Host

    if ($TorchMode -eq "cpu") {
        Write-Host "Installing CPU PyTorch. This may be too slow for TTS."
        & ".\.venv\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --default-timeout=1000
    }
    elseif ($TorchMode -eq "nightly-cu128") {
        Write-Host "Installing PyTorch nightly CUDA 12.8..."
        & ".\.venv\Scripts\python.exe" -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --default-timeout=1000
    }
    else {
        Write-Host "Installing PyTorch CUDA 12.8 stable..."
        try {
            & ".\.venv\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --default-timeout=1000
        }
        catch {
            Write-Host "Stable cu128 install failed. Trying nightly cu128..." -ForegroundColor DarkYellow
            & ".\.venv\Scripts\python.exe" -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --default-timeout=1000
        }
    }
}

Invoke-Step "Installing TTS dependencies" {
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements_local.txt --default-timeout=1000
}

Invoke-Step "Installing SoX optional dependency" {
    if ($SkipSoX) {
        Write-Host "Skipped SoX installation."
    }
    elseif (Test-CommandExists "sox") {
        sox --version
    }
    elseif (Test-CommandExists "winget") {
        Write-Host "Installing SoX with winget..."
        try {
            winget install -e --id ChrisBagwell.SoX
            Write-Host "SoX installed. If 'sox' is still not recognized, close this terminal and open it again." -ForegroundColor Green
        }
        catch {
            Write-Host "SoX installation failed. TTS may still work, but warnings may appear." -ForegroundColor DarkYellow
        }
    }
    else {
        Write-Host "winget not found. Install SoX manually if needed." -ForegroundColor DarkYellow
    }
}

Invoke-Step "Creating folders" {
    New-Item -ItemType Directory -Force -Path "voices" | Out-Null
    New-Item -ItemType Directory -Force -Path "outputs" | Out-Null

    if (-not (Test-Path "voices\README.txt")) {
        @"
Put voice reference WAV files here.

Example:
    voices\1.wav
    voices\1.txt

The .txt file must contain the exact transcript of the WAV file.
Recommended reference length: 5-20 seconds.
"@ | Set-Content -Path "voices\README.txt" -Encoding UTF8
    }
}

Invoke-Step "Checking installation" {
    $checkCode = @'
import sys
print("Python:", sys.version)

try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("Torch check failed:", repr(e))

try:
    import faster_qwen3_tts
    print("faster_qwen3_tts: OK")
except Exception as e:
    print("faster_qwen3_tts check failed:", repr(e))

try:
    import soundfile, sounddevice, numpy
    print("audio deps: OK")
except Exception as e:
    print("audio deps check failed:", repr(e))
'@
    & ".\.venv\Scripts\python.exe" -c $checkCode
}

Write-Host ""
Write-Host "=== Installation complete ===" -ForegroundColor Green
Write-Host "Add voice samples into: voices\"
Write-Host "Then run: run_tts.bat"
Write-Host ""
