# Stress modes pack

Эта версия нужна, если TTS плохо произносит текст после словаря ударений.

В `stress_words_ru.json` есть параметр:

```json
"stress_output_mode": "unicode"
```

Режимы:

- `unicode` — `каталО'г` отправляется как `катало́г`
- `apostrophe` — отправляется как `каталО'г`
- `uppercase` — отправляется как `каталОг`
- `none` — отправляется как `каталог`

Если произношение стало хуже или модель странно читает ударения, сначала попробуй:

```json
"stress_output_mode": "none"
```

Потом:

```json
"stress_output_mode": "unicode"
```

И только потом:

```json
"stress_output_mode": "uppercase"
```

`apostrophe` я бы оставлял только если конкретно на твоём голосе он звучит лучше.

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

Запуск:

```powershell
.\.venv\Scripts\python.exe tts_simple_local.py
```

Смотри строку `Prepared text:`. Если там стоят странные символы или апострофы, поменяй `stress_output_mode`.
