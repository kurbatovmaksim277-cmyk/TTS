# Omnix TTS Extract

Готовый минимальный набор файлов для использования TTS из проекта Omnix отдельно.

Есть два варианта:

1. `omnix_tts_client.py` — использовать уже запущенный Omnix как TTS-сервер.
2. `tts_simple_local.py` — использовать `faster-qwen3-tts` напрямую в отдельном Python-файле.

---

## Вариант 1. Клиент к уже запущенному Omnix

Сначала запусти Omnix обычным способом, например:

```powershell
python app.py
```

Потом в отдельном терминале:

```powershell
python -m venv .venv
.\.venv\Scripts\activate

python -m pip install -r requirements_client.txt

python omnix_tts_client.py
```

Скрипт получит список голосов через:

```text
GET /api/tts/speakers
```

И будет отправлять текст через:

```text
POST /api/tts
```

---

## Вариант 2. Локальный TTS без запуска Omnix

Этот вариант использует библиотеку `faster-qwen3-tts`.

Создай окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
```

Установи PyTorch CUDA. Пример для CUDA 12.4:

```powershell
python -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

Потом установи зависимости:

```powershell
python -m pip install -r requirements_local.txt
```

Положи WAV-файлы голосов в папку:

```text
voices/
```

Например:

```text
voices/max.wav
voices/girl.wav
voices/robot.wav
```

Запусти:

```powershell
python tts_simple_local.py
```

---

## Как выбрать голос

В локальном варианте имя голоса = имя WAV-файла без расширения.

Например:

```text
voices/max.wav
```

В программе выбирай:

```text
max
```

---

## Важно про reference text

Для лучшего качества желательно создать рядом с WAV текстовый файл с точным текстом, который произнесён в образце.

Пример:

```text
voices/max.wav
voices/max.txt
```

В `max.txt` напиши точную фразу из `max.wav`.

Если `.txt` нет, скрипт всё равно попробует запустить TTS, но качество клонирования может быть хуже.

---

## Рекомендуемый WAV для голоса

Лучше использовать:

- 5–15 секунд речи;
- без музыки;
- без фонового шума;
- один говорящий;
- формат WAV;
- желательно 16 kHz или 24 kHz mono, но чаще библиотека сама справляется.

---

## Что внутри

```text
omnix_tts_client.py       # клиент к запущенному Omnix
tts_simple_local.py       # автономный TTS через faster-qwen3-tts
requirements_client.txt   # зависимости для клиента Omnix
requirements_local.txt    # зависимости для локального TTS
voices/                   # сюда положить WAV-файлы голосов
```

---

## Если появляется ошибка CUDA

Проверь:

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Если `torch.cuda.is_available()` показывает `False`, значит PyTorch установлен без CUDA или драйвер/версия CUDA не подходят.

---

## Если Omnix TTS работает, а локальный нет

Используй `omnix_tts_client.py`.

Это самый безопасный вариант, потому что он не ломает окружение Omnix и использует уже настроенный TTS-провайдер проекта.
