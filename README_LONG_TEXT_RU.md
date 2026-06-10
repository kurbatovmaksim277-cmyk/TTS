# Long text TTS pack

Эта версия `tts_simple_local.py` рассчитана на текст до 2000 символов.

Что изменено:

- длинный текст автоматически делится на короткие куски;
- каждый кусок генерируется отдельно;
- аудио автоматически склеивается в один WAV;
- между кусками добавляется короткая пауза;
- добавлен ввод текста из `.txt` файла через `file:path.txt`;
- исправлено прерывание воспроизведения через Ctrl+C без длинного traceback.

## Установка

Скопируй в папку проекта:

```text
tts_simple_local.py
stress_words_ru.json
```

Папка должна выглядеть так:

```text
omnix_tts_extract/
├─ tts_simple_local.py
├─ stress_words_ru.json
├─ voices/
│  ├─ 1.wav
│  └─ 1.txt
```

## Запуск

```powershell
.\.venv\Scripts\python.exe tts_simple_local.py
```

## Ввод длинного текста

Можно вставлять текст прямо в строку `Text:`.

Для очень длинного/многострочного текста лучше сохранить его в файл, например:

```text
input.txt
```

И в программе ввести:

```text
file:input.txt
```

## Настройки через переменные окружения

По умолчанию:

```text
MAX_INPUT_CHARS=2000
MAX_CHUNK_CHARS=220
SILENCE_BETWEEN_CHUNKS_MS=250
```

Можно временно поменять так:

```powershell
$env:MAX_INPUT_CHARS="2000"
$env:MAX_CHUNK_CHARS="180"
$env:SILENCE_BETWEEN_CHUNKS_MS="300"
.\.venv\Scripts\python.exe tts_simple_local.py
```

Если модель всё равно обрывает фразу, уменьши `MAX_CHUNK_CHARS` до `150` или `120`.
