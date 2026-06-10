# Deterministic / stable TTS pack

Эта версия уменьшает случайные изменения интонации и длительности.

Что добавлено:

- фиксированный seed `TTS_SEED`;
- повторная установка seed перед каждым chunk;
- настройки `torch.backends.cudnn.deterministic`;
- предупреждение, если `voices/<voice>.txt` слишком длинный;
- сохранён режим длинного текста до 2000 символов с нарезкой и склейкой.

## Установка

Скопируй в папку проекта:

```text
tts_simple_local.py
stress_words_ru.json
```

## Запуск

```powershell
cd C:\Users\maksim\Desktop\omnix_tts_extract
.\.venv\Scripts\python.exe tts_simple_local.py
```

## Фиксация seed

По умолчанию:

```text
TTS_SEED=12345
```

Можно менять:

```powershell
$env:TTS_SEED="12345"
.\.venv\Scripts\python.exe tts_simple_local.py
```

Для другого варианта интонации:

```powershell
$env:TTS_SEED="777"
.\.venv\Scripts\python.exe tts_simple_local.py
```

Если нужен максимально стабильный режим:

```powershell
$env:TTS_DETERMINISTIC="1"
$env:TTS_SEED="12345"
.\.venv\Scripts\python.exe tts_simple_local.py
```

## Важно

Это уменьшает разброс, но не гарантирует 100% одинаковый WAV на каждом запуске, потому что CUDA и сама TTS-библиотека могут использовать недетерминированные операции.

Если один и тот же текст всё равно звучит сильно по-разному, главная причина обычно в reference voice:

```text
voices\3.wav
voices\3.txt
```

Лучше использовать короткий reference:

- 5–20 секунд речи;
- чистый звук без музыки;
- один голос;
- `3.txt` должен быть точной расшифровкой именно `3.wav`, не всей длинной речи.
