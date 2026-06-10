# Omnix TTS colleague package




cd C:\Users\maksim\Desktop\omnix_tts_extract
Remove-Item -Recurse -Force .venv

py -0p

py -3.12 -m venv .venv

.\install_windows.bat







Это набор для запуска локального TTS на Windows.

## Что внутри

```text
install_windows.bat       # простой установщик для коллеги
install_windows.ps1       # основной PowerShell-установщик
run_tts.bat               # запуск после установки
check_install.bat         # проверка PyTorch/CUDA
requirements_local.txt    # зависимости TTS без PyTorch
tts_simple_local.py       # основной скрипт
stress_words_ru.json      # словарь ударений/замен
voices/                   # сюда положить WAV-голоса
outputs/                  # сюда сохраняются WAV-результаты
```

## Быстрый запуск на компьютере коллеги

1. Распаковать папку.
2. Запустить:

```text
install_windows.bat
```

3. Положить голос:

```text
voices\1.wav
voices\1.txt
```

`1.txt` должен содержать точную расшифровку того, что произнесено в `1.wav`.

Рекомендуемый reference: 5-20 секунд чистой речи.

4. Запустить:

```text
run_tts.bat
```

## Важно про Python

Установщик пытается использовать Python 3.12. Если Python 3.12 не найден, он попробует поставить его через winget.

После установки Python через winget нужно закрыть терминал, открыть заново и снова запустить:

```text
install_windows.bat
```

## Важно про RTX 50 / RTX 5070

По умолчанию ставится PyTorch CUDA 12.8:

```text
https://download.pytorch.org/whl/cu128
```

Если stable-сборка не установится, установщик попробует nightly CUDA 12.8.

## Ручные варианты

Stable CUDA 12.8:

```powershell
install_windows.bat
```

Nightly CUDA 12.8 принудительно:

```powershell
install_windows.bat -TorchMode nightly-cu128
```

CPU-режим, если нет NVIDIA GPU:

```powershell
install_windows.bat -TorchMode cpu
```

CPU-режим может быть очень медленным и не рекомендуется.

## Проверка

```text
check_install.bat
```

Нужно увидеть:

```text
CUDA available: True
GPU: NVIDIA ...
```

## Запуск 1.7B модели

Открой `run_tts.bat` и раскомментируй строку:

```bat
set QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
```
