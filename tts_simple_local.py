"""
Quality-first local Qwen3-TTS voice cloning with style instructions.

Основные улучшения:
    - backend official (лучшее соответствие исходной модели) или faster (скорость)
    - модель 1.7B по умолчанию
    - проверка CUDA/RTX 50xx до загрузки модели
    - полный ICL voice clone при наличии точного voices/<name>.txt
    - non_streaming_mode=True: весь текст фрагмента подаётся модели заранее
    - текст до 2000 символов, деление только по естественным границам
    - автоматический повтор/дробление подозрительно короткого результата
    - фиксированный seed и профили quality/stable/expressive
    - нормализация русских чисел, годов и словарь ударений
    - плавные края и естественные паузы между фрагментами
    - защита от длинных артефактных хвостов: динамический token budget и fallback

Запуск:
    python tts_qwen_quality.py
"""


from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import torch
import soundfile as sf


MODEL_ID = os.environ.get(
    "QWEN_TTS_MODEL",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
)
BACKEND = os.environ.get("TTS_BACKEND", "faster").strip().lower()
PROFILE = os.environ.get("TTS_PROFILE", "quality").strip().lower()

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "voices"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))
STRESS_DICT_PATH = Path(os.environ.get("STRESS_DICT_PATH", "stress_words_ru.json"))

DEFAULT_STYLE_INSTRUCT = os.environ.get("TTS_STYLE_INSTRUCT", "").strip()
PRESERVE_VOICE_PREFIX = os.environ.get(
    "TTS_PRESERVE_VOICE_PREFIX",
    (
        "Максимально сохраняй тембр, высоту голоса, артикуляцию, ритм, "
        "характер пауз и индивидуальную манеру говорящего из референсного аудио. "
        "Не меняй личность, возраст и основной характер голоса."
    ),
).strip()

MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "2000"))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "240"))
MAX_RETRY_DEPTH = int(os.environ.get("MAX_RETRY_DEPTH", "2"))

TTS_SEED = int(os.environ.get("TTS_SEED", "12345"))
TTS_DETERMINISTIC = os.environ.get("TTS_DETERMINISTIC", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
MAX_REF_TEXT_CHARS_WARNING = int(os.environ.get("MAX_REF_TEXT_CHARS_WARNING", "500"))

# Паузы используются только в местах, где текст действительно был разделён.
PAUSE_COMMA_MS = int(os.environ.get("PAUSE_COMMA_MS", "30"))
PAUSE_SEMICOLON_MS = int(os.environ.get("PAUSE_SEMICOLON_MS", "55"))
PAUSE_SENTENCE_MS = int(os.environ.get("PAUSE_SENTENCE_MS", "80"))
EDGE_FADE_MS = int(os.environ.get("EDGE_FADE_MS", "5"))
FINAL_SILENCE_MS = int(os.environ.get("FINAL_SILENCE_MS", "350"))

# Normalize hidden silence that the model may generate at chunk boundaries.
# Without this, a chunk can already end with 0.5-2.0 seconds of silence and
# the script then adds another explicit pause.
TRIM_CHUNK_BOUNDARY_SILENCE = os.environ.get(
    "TRIM_CHUNK_BOUNDARY_SILENCE", "1"
).strip().lower() not in {"0", "false", "no", "off"}
BOUNDARY_FRAME_MS = int(os.environ.get("BOUNDARY_FRAME_MS", "10"))
MAX_INTERNAL_LEADING_SILENCE_MS = int(
    os.environ.get("MAX_INTERNAL_LEADING_SILENCE_MS", "110")
)
MAX_INTERNAL_TRAILING_SILENCE_MS = int(
    os.environ.get("MAX_INTERNAL_TRAILING_SILENCE_MS", "170")
)
KEEP_INTERNAL_LEADING_SILENCE_MS = int(
    os.environ.get("KEEP_INTERNAL_LEADING_SILENCE_MS", "15")
)
KEEP_INTERNAL_TRAILING_SILENCE_MS = int(
    os.environ.get("KEEP_INTERNAL_TRAILING_SILENCE_MS", "55")
)
FINAL_CHUNK_ATTEMPTS = max(1, int(os.environ.get("FINAL_CHUNK_ATTEMPTS", "2")))
MIN_TOKEN_CHAR_FACTOR = float(os.environ.get("MIN_TOKEN_CHAR_FACTOR", "0.72"))

# Prevent long-form generation from continuing into repetitions/noise after
# the requested text has already been spoken.
MAX_TOKEN_CHAR_FACTOR = float(os.environ.get("MAX_TOKEN_CHAR_FACTOR", "1.35"))
MAX_TOKEN_EXTRA = int(os.environ.get("MAX_TOKEN_EXTRA", "48"))
ABSOLUTE_MAX_NEW_TOKENS = int(os.environ.get("ABSOLUTE_MAX_NEW_TOKENS", "1024"))

# A natural Russian passage normally remains inside these conservative bounds.
MAX_SECONDS_PER_WORD = float(os.environ.get("MAX_SECONDS_PER_WORD", "0.86"))
MAX_SECONDS_PER_LETTER = float(os.environ.get("MAX_SECONDS_PER_LETTER", "0.135"))
MAX_DURATION_EXTRA_SEC = float(os.environ.get("MAX_DURATION_EXTRA_SEC", "6.0"))

# Experimental clone-instruct becomes less stable when it is much longer than
# the synthesis text. Keep only the most useful first sentences.
MAX_STYLE_INSTRUCTION_CHARS = int(
    os.environ.get("MAX_STYLE_INSTRUCTION_CHARS", "420")
)

# Long-form generation uses a calmer sampling profile even when the selected
# profile is "quality".
LONG_FORM_TEMPERATURE_CAP = float(
    os.environ.get("LONG_FORM_TEMPERATURE_CAP", "0.68")
)
LONG_FORM_TOP_K_CAP = int(os.environ.get("LONG_FORM_TOP_K_CAP", "25"))
LONG_FORM_TOP_P_CAP = float(os.environ.get("LONG_FORM_TOP_P_CAP", "0.90"))
LONG_FORM_REPETITION_PENALTY = float(
    os.environ.get("LONG_FORM_REPETITION_PENALTY", "1.08")
)

# Voice consistency controls.
# The best way to avoid a timbre jump is to generate the whole passage in one
# model call. If that result looks truncated, the script falls back to chunks.
WHOLE_TEXT_FIRST = os.environ.get(
    "WHOLE_TEXT_FIRST", "1"
).strip().lower() not in {"0", "false", "no", "off"}
WHOLE_TEXT_MAX_CHARS = int(os.environ.get("WHOLE_TEXT_MAX_CHARS", "650"))

# All fallback chunks start from the same stochastic state. Previously every
# chunk used a text-derived seed, which encouraged a new voice realization.
CONSISTENT_CHUNK_SEED = os.environ.get(
    "CONSISTENT_CHUNK_SEED", "1"
).strip().lower() not in {"0", "false", "no", "off"}

# Do not create several valid final candidates and select the longest one.
# Retrying a final chunk is now done only when truncation is suspected.
RETRY_ONLY_IF_TRUNCATED = os.environ.get(
    "RETRY_ONLY_IF_TRUNCATED", "1"
).strip().lower() not in {"0", "false", "no", "off"}

MATCH_CHUNK_LOUDNESS = os.environ.get(
    "MATCH_CHUNK_LOUDNESS", "1"
).strip().lower() not in {"0", "false", "no", "off"}
LOUDNESS_GAIN_MIN = float(os.environ.get("LOUDNESS_GAIN_MIN", "0.82"))
LOUDNESS_GAIN_MAX = float(os.environ.get("LOUDNESS_GAIN_MAX", "1.20"))

CONTINUITY_INSTRUCTION = os.environ.get(
    "TTS_CONTINUITY_INSTRUCTION",
    (
        "Сохраняй один и тот же тембр, высоту, речевой регистр, громкость, "
        "темп и эмоциональное состояние на протяжении всей записи."
    ),
).strip()

GENERATION_PROFILES = {
    "quality": {
        "temperature": 0.80,
        "top_k": 40,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "do_sample": True,
    },
    "stable": {
        "temperature": 0.65,
        "top_k": 25,
        "top_p": 0.90,
        "repetition_penalty": 1.06,
        "do_sample": True,
    },
    "expressive": {
        "temperature": 0.95,
        "top_k": 60,
        "top_p": 1.00,
        "repetition_penalty": 1.04,
        "do_sample": True,
    },
}



def find_voices(voices_dir: Path = VOICES_DIR) -> dict[str, Path]:
    voices_dir.mkdir(parents=True, exist_ok=True)
    voices: dict[str, Path] = {}
    for wav in sorted(voices_dir.glob("*.wav")):
        voices[wav.stem] = wav
    return voices


def read_reference_text(wav_path: Path) -> str:
    txt_path = wav_path.with_suffix(".txt")
    if not txt_path.exists():
        return ""
    return txt_path.read_text(encoding="utf-8").strip()


def read_style_instruction(wav_path: Path) -> str:
    """Read optional voices/<voice>.style.txt."""
    style_path = wav_path.with_name(f"{wav_path.stem}.style.txt")
    if style_path.exists():
        return style_path.read_text(encoding="utf-8").strip()
    return DEFAULT_STYLE_INSTRUCT


def _shorten_style_instruction(text: str, limit: int) -> str:
    """Keep complete leading sentences and avoid cutting a word in half."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned

    candidate = cleaned[:limit]
    punctuation = max(
        candidate.rfind("."),
        candidate.rfind("!"),
        candidate.rfind("?"),
        candidate.rfind(";"),
    )

    if punctuation >= int(limit * 0.55):
        candidate = candidate[: punctuation + 1]
    else:
        space = candidate.rfind(" ")
        if space > 0:
            candidate = candidate[:space]

    return candidate.strip()


def build_clone_instruction(user_instruction: str) -> str:
    """
    Preserve speaker identity while adding optional prosody guidance.

    Empty instruction is intentional: it lets ICL copy the reference manner
    without additional steering.
    """
    user_instruction = re.sub(r"\s+", " ", user_instruction).strip()
    if not user_instruction:
        return ""

    shortened = _shorten_style_instruction(
        user_instruction,
        MAX_STYLE_INSTRUCTION_CHARS,
    )
    if len(shortened) < len(user_instruction):
        print(
            f"Style instruction shortened: "
            f"{len(user_instruction)} -> {len(shortened)} chars"
        )

    return (
        f"{PRESERVE_VOICE_PREFIX} "
        f"{CONTINUITY_INSTRUCTION} "
        f"Дополнительная манера: {shortened}"
    )


# ----------------------------
# Reproducibility helpers
# ----------------------------

def set_tts_seed(seed: int = TTS_SEED) -> None:
    """
    Make TTS generation more repeatable.

    Notes:
      - Some CUDA kernels may still be slightly nondeterministic.
      - If the TTS library uses its own internal RNG, global seeding may not fully control it.
      - Still, this usually reduces random changes in duration and intonation.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if TTS_DETERMINISTIC:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Make matmul behavior more stable on recent NVIDIA GPUs.
        # This can slightly reduce speed but improves repeatability.
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def stable_chunk_seed(base_seed: int, chunk_index: int, text: str) -> int:
    """
    Build a stable per-chunk seed without Python's randomized hash().
    Same text + same chunk index + same base seed -> same seed.
    """
    value = base_seed + chunk_index * 1009

    for ch in text:
        value = (value * 131 + ord(ch)) % 2_147_483_647

    return value


def generation_seed(
    *,
    text: str,
    chunk_index: int,
    attempt: int,
    depth: int,
) -> int:
    """
    Keep the first attempt identical across all chunks.

    A retry gets another deterministic seed, but only after the previous
    candidate was classified as truncated.
    """
    retry_base = TTS_SEED + depth * 100_003 + attempt * 10_007

    if CONSISTENT_CHUNK_SEED:
        return retry_base

    return stable_chunk_seed(retry_base, chunk_index, text)


# Seed once before model initialization.
set_tts_seed(TTS_SEED)


# ----------------------------
# Russian number normalization
# ----------------------------

ONES_MASC = {
    0: "ноль", 1: "один", 2: "два", 3: "три", 4: "четыре",
    5: "пять", 6: "шесть", 7: "семь", 8: "восемь", 9: "девять",
    10: "десять", 11: "одиннадцать", 12: "двенадцать", 13: "тринадцать",
    14: "четырнадцать", 15: "пятнадцать", 16: "шестнадцать",
    17: "семнадцать", 18: "восемнадцать", 19: "девятнадцать",
}

TENS = {
    20: "двадцать", 30: "тридцать", 40: "сорок", 50: "пятьдесят",
    60: "шестьдесят", 70: "семьдесят", 80: "восемьдесят", 90: "девяносто",
}

HUNDREDS = {
    100: "сто", 200: "двести", 300: "триста", 400: "четыреста",
    500: "пятьсот", 600: "шестьсот", 700: "семьсот", 800: "восемьсот", 900: "девятьсот",
}

ORDINAL_0_99_NOM = {
    0: "нулевой", 1: "первый", 2: "второй", 3: "третий", 4: "четвёртый",
    5: "пятый", 6: "шестой", 7: "седьмой", 8: "восьмой", 9: "девятый",
    10: "десятый", 11: "одиннадцатый", 12: "двенадцатый", 13: "тринадцатый",
    14: "четырнадцатый", 15: "пятнадцатый", 16: "шестнадцатый",
    17: "семнадцатый", 18: "восемнадцатый", 19: "девятнадцатый",
    20: "двадцатый", 30: "тридцатый", 40: "сороковой",
    50: "пятидесятый", 60: "шестидесятый", 70: "семидесятый",
    80: "восьмидесятый", 90: "девяностый",
}

ORDINAL_LAST_WORD_FORMS = {
    "nom": {
        "нулевой": "нулевой", "первый": "первый", "второй": "второй", "третий": "третий",
        "четвёртый": "четвёртый", "четвертый": "четвертый", "пятый": "пятый",
        "шестой": "шестой", "седьмой": "седьмой", "восьмой": "восьмой", "девятый": "девятый",
        "десятый": "десятый", "одиннадцатый": "одиннадцатый", "двенадцатый": "двенадцатый",
        "тринадцатый": "тринадцатый", "четырнадцатый": "четырнадцатый",
        "пятнадцатый": "пятнадцатый", "шестнадцатый": "шестнадцатый",
        "семнадцатый": "семнадцатый", "восемнадцатый": "восемнадцатый",
        "девятнадцатый": "девятнадцатый", "двадцатый": "двадцатый",
        "тридцатый": "тридцатый", "сороковой": "сороковой", "пятидесятый": "пятидесятый",
        "шестидесятый": "шестидесятый", "семидесятый": "семидесятый",
        "восьмидесятый": "восьмидесятый", "девяностый": "девяностый",
    },
    "gen": {
        "нулевой": "нулевого", "первый": "первого", "второй": "второго", "третий": "третьего",
        "четвёртый": "четвёртого", "четвертый": "четвертого", "пятый": "пятого",
        "шестой": "шестого", "седьмой": "седьмого", "восьмой": "восьмого", "девятый": "девятого",
        "десятый": "десятого", "одиннадцатый": "одиннадцатого", "двенадцатый": "двенадцатого",
        "тринадцатый": "тринадцатого", "четырнадцатый": "четырнадцатого",
        "пятнадцатый": "пятнадцатого", "шестнадцатый": "шестнадцатого",
        "семнадцатый": "семнадцатого", "восемнадцатый": "восемнадцатого",
        "девятнадцатый": "девятнадцатого", "двадцатый": "двадцатого",
        "тридцатый": "тридцатого", "сороковой": "сорокового", "пятидесятый": "пятидесятого",
        "шестидесятый": "шестидесятого", "семидесятый": "семидесятого",
        "восьмидесятый": "восьмидесятого", "девяностый": "девяностого",
    },
    "prep": {
        "нулевой": "нулевом", "первый": "первом", "второй": "втором", "третий": "третьем",
        "четвёртый": "четвёртом", "четвертый": "четвертом", "пятый": "пятом",
        "шестой": "шестом", "седьмой": "седьмом", "восьмой": "восьмом", "девятый": "девятом",
        "десятый": "десятом", "одиннадцатый": "одиннадцатом", "двенадцатый": "двенадцатом",
        "тринадцатый": "тринадцатом", "четырнадцатый": "четырнадцатом",
        "пятнадцатый": "пятнадцатом", "шестнадцатый": "шестнадцатом",
        "семнадцатый": "семнадцатом", "восемнадцатый": "восемнадцатом",
        "девятнадцатый": "девятнадцатом", "двадцатый": "двадцатом",
        "тридцатый": "тридцатом", "сороковой": "сороковом", "пятидесятый": "пятидесятом",
        "шестидесятый": "шестидесятом", "семидесятый": "семидесятом",
        "восьмидесятый": "восьмидесятом", "девяностый": "девяностом",
    },
    "inst": {
        "нулевой": "нулевым", "первый": "первым", "второй": "вторым", "третий": "третьим",
        "четвёртый": "четвёртым", "четвертый": "четвертым", "пятый": "пятым",
        "шестой": "шестым", "седьмой": "седьмым", "восьмой": "восьмым", "девятый": "девятым",
        "десятый": "десятым", "одиннадцатый": "одиннадцатым", "двенадцатый": "двенадцатым",
        "тринадцатый": "тринадцатым", "четырнадцатый": "четырнадцатым",
        "пятнадцатый": "пятнадцатым", "шестнадцатый": "шестнадцатым",
        "семнадцатый": "семнадцатым", "восемнадцатый": "восемнадцатым",
        "девятнадцатый": "девятнадцатым", "двадцатый": "двадцатым",
        "тридцатый": "тридцатым", "сороковой": "сороковым", "пятидесятый": "пятидесятым",
        "шестидесятый": "шестидесятым", "семидесятый": "семидесятым",
        "восьмидесятый": "восьмидесятым", "девяностый": "девяностым",
    },
}


def ru_0_99(n: int) -> str:
    if n < 0 or n > 99:
        return str(n)
    if n < 20:
        return ONES_MASC[n]
    tens_part = (n // 10) * 10
    ones_part = n % 10
    if ones_part == 0:
        return TENS[tens_part]
    return f"{TENS[tens_part]} {ONES_MASC[ones_part]}"


def ru_number_cardinal(n: int) -> str:
    if n < 0:
        return "минус " + ru_number_cardinal(abs(n))
    if n <= 99:
        return ru_0_99(n)
    if n <= 999:
        hundreds_part = (n // 100) * 100
        rest = n % 100
        if rest == 0:
            return HUNDREDS[hundreds_part]
        return f"{HUNDREDS[hundreds_part]} {ru_0_99(rest)}"
    if n <= 9999:
        thousands_part = n // 1000
        rest = n % 1000
        if thousands_part == 1:
            prefix = "одна тысяча"
        elif thousands_part == 2:
            prefix = "две тысячи"
        elif 3 <= thousands_part <= 4:
            prefix = f"{ONES_MASC[thousands_part]} тысячи"
        else:
            prefix = f"{ru_0_99(thousands_part)} тысяч"
        if rest == 0:
            return prefix
        return f"{prefix} {ru_number_cardinal(rest)}"
    return str(n)


def _year_prefix_and_last(year: int) -> tuple[str, int] | None:
    if 1900 <= year <= 1999:
        return "тысяча девятьсот", year - 1900
    if 2000 <= year <= 2099:
        return "две тысячи", year - 2000
    if 2100 <= year <= 2199:
        return "две тысячи сто", year - 2100
    return None


def ru_year_nom(year: int) -> str:
    result = _year_prefix_and_last(year)
    if result is None:
        return ru_number_cardinal(year)
    prefix, last = result
    if last == 0:
        if year == 2000:
            return "двухтысячный"
        if year == 1900:
            return "тысяча девятисотый"
        if year == 2100:
            return "две тысячи сотый"
        return f"{prefix} нулевой"
    if last in ORDINAL_0_99_NOM:
        return f"{prefix} {ORDINAL_0_99_NOM[last]}"
    tens_part = (last // 10) * 10
    ones_part = last % 10
    return f"{prefix} {TENS[tens_part]} {ORDINAL_0_99_NOM[ones_part]}"


def ru_year_case(year: int, case: str) -> str:
    text = ru_year_nom(year)
    words = text.split()
    if not words:
        return text
    last = words[-1]
    forms = ORDINAL_LAST_WORD_FORMS.get(case, {})
    if last in forms:
        words[-1] = forms[last]
    return " ".join(words)


def normalize_text_for_tts(text: str, language: str = "Russian") -> str:
    lang = language.lower().strip()
    if lang not in {"russian", "ru", "русский"}:
        return text

    def replace_year_with_word(match: re.Match) -> str:
        year = int(match.group(1))
        word = match.group(2).lower()
        if word == "год":
            return f"{ru_year_case(year, 'nom')} год"
        if word == "года":
            return f"{ru_year_case(year, 'gen')} года"
        if word == "году":
            return f"{ru_year_case(year, 'prep')} году"
        if word == "годом":
            return f"{ru_year_case(year, 'inst')} годом"
        return match.group(0)

    text = re.sub(
        r"\b(19\d{2}|20\d{2}|21\d{2})\s+(год|года|году|годом)\b",
        replace_year_with_word,
        text,
        flags=re.IGNORECASE,
    )

    # Modern abbreviated year: "в 26 году" -> "в две тысячи двадцать шестом году".
    # This is intentionally limited to the construction "в NN году".
    def replace_short_year_in_prep(match: re.Match) -> str:
        short_year = int(match.group(1))
        full_year = 2000 + short_year
        return f"в {ru_year_case(full_year, 'prep')} году"

    text = re.sub(
        r"\bв\s+(\d{1,2})\s+году\b",
        replace_short_year_in_prep,
        text,
        flags=re.IGNORECASE,
    )

    def replace_bare_year(match: re.Match) -> str:
        year = int(match.group(1))
        return f"{ru_year_case(year, 'nom')} год"

    text = re.sub(r"\b(19\d{2}|20\d{2}|21\d{2})\b", replace_bare_year, text)

    def replace_percent(match: re.Match) -> str:
        number = int(match.group(1))
        if 0 <= number <= 9999:
            return f"{ru_number_cardinal(number)} процентов"
        return match.group(0)

    text = re.sub(r"\b(\d{1,4})\s*%", replace_percent, text)

    def replace_number(match: re.Match) -> str:
        number = int(match.group(0))
        if 0 <= number <= 9999:
            return ru_number_cardinal(number)
        return match.group(0)

    text = re.sub(r"\b\d{1,4}\b", replace_number, text)

    return text


# ----------------------------
# External stress dictionary
# ----------------------------

CYR_LAT_WORD_CHARS = "A-Za-zА-Яа-яЁё0-9_"
STRESSED_VOWELS = {
    "А": "а", "Е": "е", "Ё": "ё", "И": "и", "О": "о", "У": "у", "Ы": "ы", "Э": "э", "Ю": "ю", "Я": "я",
    "A": "a", "E": "e", "O": "o", "Y": "y",
}
COMBINING_ACUTE = "\u0301"

DEFAULT_STRESS_DICT: dict[str, Any] = {
    "enabled": True,
    "stress_output_mode": "unicode",
    "debug_print_prepared_text": True,
    "description": "Russian TTS stress dictionary.",
    "yo_words": {},
    "phrases": {},
    "words": {},
    "regex": [],
}


def load_stress_dictionary(path: Path = STRESS_DICT_PATH) -> dict[str, Any]:
    if not path.exists():
        return DEFAULT_STRESS_DICT.copy()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: could not load stress dictionary {path}: {exc}")
        return DEFAULT_STRESS_DICT.copy()

    if not isinstance(data, dict):
        print(f"Warning: stress dictionary {path} is not a JSON object.")
        return DEFAULT_STRESS_DICT.copy()

    result = DEFAULT_STRESS_DICT.copy()
    result.update(data)
    return result


def convert_stress_notation(value: str, mode: str) -> str:
    """
    Converts convenient dictionary notation to what is sent to TTS.

    Input notation:
        каталО'г
        нО'вый гО'д

    Modes:
        unicode    -> катало́г
        apostrophe -> каталО'г
        uppercase  -> каталОг
        none       -> каталог
    """
    if mode == "apostrophe":
        return value

    result: list[str] = []
    i = 0

    while i < len(value):
        ch = value[i]
        next_ch = value[i + 1] if i + 1 < len(value) else ""

        if ch in STRESSED_VOWELS and next_ch == "'":
            lower = STRESSED_VOWELS[ch]

            if mode == "unicode":
                result.append(lower + COMBINING_ACUTE)
            elif mode == "uppercase":
                result.append(ch)
            elif mode == "none":
                result.append(lower)
            else:
                # Unknown mode: safest is no artificial stress.
                result.append(lower)

            i += 2
            continue

        # Also strip a bare apostrophe if mode is not apostrophe.
        if ch == "'" and mode != "apostrophe":
            i += 1
            continue

        if mode in {"unicode", "none"} and ch in STRESSED_VOWELS:
            # A capital vowel without apostrophe may be accidental from older dictionaries.
            result.append(STRESSED_VOWELS[ch])
        else:
            result.append(ch)

        i += 1

    return "".join(result)


def convert_mapping_values(mapping: dict[str, str], mode: str) -> dict[str, str]:
    return {src: convert_stress_notation(dst, mode) for src, dst in mapping.items()}


def _literal_replace(text: str, mapping: dict[str, str], word_mode: bool) -> str:
    if not mapping:
        return text

    items = sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True)

    for src, dst in items:
        if not src:
            continue

        if word_mode:
            pattern = rf"(?<![{CYR_LAT_WORD_CHARS}]){re.escape(src)}(?![{CYR_LAT_WORD_CHARS}])"
        else:
            pattern = re.escape(src)

        text = re.sub(pattern, dst, text, flags=re.IGNORECASE)

    return text


def apply_stress_dictionary(text: str, language: str = "Russian") -> str:
    lang = language.lower().strip()

    if lang not in {"russian", "ru", "русский"}:
        return text

    dictionary = load_stress_dictionary()

    if not dictionary.get("enabled", True):
        return text

    stress_mode = str(dictionary.get("stress_output_mode", "unicode")).lower().strip()

    yo_words = dictionary.get("yo_words", {})
    if isinstance(yo_words, dict):
        text = _literal_replace(text, {str(k): str(v) for k, v in yo_words.items()}, word_mode=True)

    phrases = dictionary.get("phrases", {})
    if isinstance(phrases, dict):
        phrases = convert_mapping_values({str(k): str(v) for k, v in phrases.items()}, stress_mode)
        text = _literal_replace(text, phrases, word_mode=False)

    words = dictionary.get("words", {})
    if isinstance(words, dict):
        words = convert_mapping_values({str(k): str(v) for k, v in words.items()}, stress_mode)
        text = _literal_replace(text, words, word_mode=True)

    regex_rules = dictionary.get("regex", [])
    if isinstance(regex_rules, list):
        for rule in regex_rules:
            if not isinstance(rule, dict):
                continue
            pattern = rule.get("pattern")
            replacement = rule.get("replacement")
            if not pattern or replacement is None:
                continue
            replacement = convert_stress_notation(str(replacement), stress_mode)
            try:
                text = re.sub(str(pattern), replacement, text, flags=re.IGNORECASE)
            except re.error as exc:
                print(f"Warning: bad regex in stress dictionary: {pattern!r}: {exc}")

    return text


# ----------------------------
# Long text splitting / audio joining
# ----------------------------

def clean_text_before_tts(text: str) -> str:
    """
    Makes punctuation safer for TTS.

    This reduces early stops caused by unusual quotes, em-dashes,
    glued punctuation like "друзья!Буквально", and common roman numerals.
    """
    replacements = {
        "«": "",
        "»": "",
        "“": "",
        "”": "",
        "„": "",
        "—": ", ",
        "–": ", ",
        "−": "-",
        "\u00a0": " ",
        "…": ".",
        "XXI": "двадцать первого",
        "xxi": "двадцать первого",
        "XX": "двадцатого",
        "xx": "двадцатого",
        "XIX": "девятнадцатого",
        "xix": "девятнадцатого",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Add a missing space after punctuation: "друзья!Буквально" -> "друзья! Буквально"
    text = re.sub(r"([.!?])(?=[А-ЯЁA-Z])", r"\1 ", text)

    # A lone closing parenthesis is often used as an emoticon:
    # "возможности) Добавьте" -> "возможности. Добавьте".
    text = re.sub(r"(?<=[А-Яа-яЁё0-9])\)\s+(?=[А-ЯЁA-Z])", ". ", text)

    # Normalize repeated spaces.
    text = re.sub(r"\s+", " ", text).strip()

    # A complete terminal sentence is less likely to be stopped before the last word.
    # A comma/semicolon/colon at the very end tells the model that speech continues.
    if text:
        text = re.sub(r"[,;:]+\s*$", ".", text)
        if text[-1] not in ".!?":
            text += "."

    return text


def split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Splits text into TTS-safe chunks.

    Priority:
      1. sentence boundaries
      2. comma/semicolon/colon boundaries
      3. word boundaries

    Each chunk is kept short because long one-shot generation may stop early.
    """
    text = clean_text_before_tts(text)

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # Split into sentence-like pieces while preserving punctuation.
    sentence_parts = re.findall(r"[^.!?]+[.!?]*", text)
    sentence_parts = [p.strip() for p in sentence_parts if p.strip()]

    chunks: list[str] = []

    def push_piece(piece: str) -> None:
        piece = piece.strip()
        if not piece:
            return

        if len(piece) <= max_chars:
            chunks.append(piece)
            return

        # Split very long sentence by softer punctuation.
        soft_parts = re.split(r"([,;:])", piece)
        rebuilt: list[str] = []
        for i in range(0, len(soft_parts), 2):
            part = soft_parts[i].strip()
            punct = soft_parts[i + 1] if i + 1 < len(soft_parts) else ""
            if part:
                rebuilt.append((part + punct).strip())

        current = ""
        for part in rebuilt:
            if not current:
                current = part
            elif len(current) + 1 + len(part) <= max_chars:
                current += " " + part
            else:
                if current:
                    chunks.append(current.strip())
                current = part

            # If even comma-part is too long, split by words.
            while len(current) > max_chars:
                words = current.split()
                small = ""
                rest_words = []
                for w in words:
                    if not small:
                        small = w
                    elif len(small) + 1 + len(w) <= max_chars:
                        small += " " + w
                    else:
                        rest_words.append(w)

                chunks.append(small.strip())
                current = " ".join(rest_words).strip()
                if not rest_words:
                    current = ""
                    break

        if current:
            chunks.append(current.strip())

    current_sentence_group = ""

    for part in sentence_parts:
        if not current_sentence_group:
            current_sentence_group = part
        elif len(current_sentence_group) + 1 + len(part) <= max_chars:
            current_sentence_group += " " + part
        else:
            push_piece(current_sentence_group)
            current_sentence_group = part

    if current_sentence_group:
        push_piece(current_sentence_group)

    # Final cleanup: ensure no empty chunks and sentence-like ending.
    cleaned_chunks: list[str] = []
    for c in chunks:
        c = re.sub(r"\s+", " ", c).strip()
        if not c:
            continue
        if c[-1] not in ".!?;:,":
            c += "."
        cleaned_chunks.append(c)

    if cleaned_chunks:
        # Internal chunks may end with soft punctuation, but the final chunk must
        # look like a finished sentence to discourage premature EOS.
        cleaned_chunks[-1] = re.sub(r"[,;:]+\s*$", ".", cleaned_chunks[-1]).strip()
        if cleaned_chunks[-1][-1] not in ".!?":
            cleaned_chunks[-1] += "."

    return cleaned_chunks



def check_cuda_compatibility() -> None:
    """Проверяет, что текущий PyTorch действительно умеет запускаться на GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна. Установлена CPU-сборка PyTorch либо отсутствует "
            "рабочий драйвер NVIDIA."
        )

    gpu = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    arch = f"sm_{capability[0]}{capability[1]}"
    supported = set(torch.cuda.get_arch_list())

    print(f"GPU: {gpu}")
    print(f"PyTorch: {torch.__version__} | CUDA runtime: {torch.version.cuda}")
    print(f"GPU capability: {capability} | required arch: {arch}")

    # Для Blackwell отсутствие sm_120 почти наверняка закончится
    # ошибкой "no kernel image is available".
    if capability[0] >= 12 and arch not in supported:
        raise RuntimeError(
            f"Текущий PyTorch не содержит поддержку {arch}. "
            "Для RTX 50xx установите согласованную cu128-сборку PyTorch. "
            f"Поддерживаемые архитектуры: {sorted(supported)}"
        )

    # Реальный запуск небольшого CUDA-ядра выявляет повреждённую установку раньше модели.
    try:
        test = torch.arange(8, device="cuda")
        _ = (test + 1).cpu()
    except Exception as exc:
        raise RuntimeError(f"Проверка CUDA не пройдена: {exc}") from exc


def validate_reference_audio(path: Path) -> tuple[float, int]:
    """Проверяет длительность и уровень reference WAV."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Не удалось прочитать reference audio {path}: {exc}") from exc

    arr = np.asarray(audio)
    if arr.size == 0:
        raise ValueError(f"Reference audio пустой: {path}")

    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    duration = float(len(arr) / sr)
    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))

    print(
        f"Reference: {duration:.2f} s, {sr} Hz, "
        f"peak={peak:.4f}, RMS={rms:.4f}"
    )

    if peak < 1e-4 or rms < 1e-5:
        raise ValueError("Reference audio почти полностью беззвучный.")

    if duration < 3.0:
        print("Warning: reference короче 3 секунд; похожесть голоса может снизиться.")
    elif duration > 25.0:
        print(
            "Warning: reference длиннее 25 секунд. Для стабильной работы лучше "
            "чистый фрагмент 5–20 секунд и точная расшифровка."
        )

    return duration, int(sr)


def _fade_edges(
    audio: np.ndarray,
    sample_rate: int,
    fade_ms: int = EDGE_FADE_MS,
    *,
    fade_in_enabled: bool = True,
    fade_out_enabled: bool = True,
) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    n = min(int(sample_rate * fade_ms / 1000), len(arr) // 2)
    if n <= 1:
        return arr

    if fade_in_enabled:
        fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
        arr[:n] *= fade_in

    # Never fade out the final segment: even an 8 ms fade can weaken the last
    # consonant and make a complete word sound clipped.
    if fade_out_enabled:
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        arr[-n:] *= fade_out

    return arr


def _pause_after_text(text: str) -> int:
    stripped = text.rstrip()
    if not stripped:
        return PAUSE_SENTENCE_MS
    if stripped.endswith((",",)):
        return PAUSE_COMMA_MS
    if stripped.endswith((";", ":")):
        return PAUSE_SEMICOLON_MS
    return PAUSE_SENTENCE_MS


def _frame_rms(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)

    frame_samples = max(1, int(frame_samples))
    frame_count = (arr.size + frame_samples - 1) // frame_samples
    padded_size = frame_count * frame_samples

    if padded_size != arr.size:
        arr = np.pad(arr, (0, padded_size - arr.size))

    frames = arr.reshape(frame_count, frame_samples)
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)


def _trim_internal_boundary_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    trim_leading: bool,
    trim_trailing: bool,
) -> tuple[np.ndarray, float, float]:
    """
    Conservatively remove excessive silence generated around internal chunks.

    It does not trim the final end of the complete utterance. A small safety
    margin is retained around speech to protect weak consonants and breaths.
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if (
        not TRIM_CHUNK_BOUNDARY_SILENCE
        or arr.size == 0
        or sample_rate <= 0
    ):
        return arr, 0.0, 0.0

    frame_samples = max(1, int(sample_rate * BOUNDARY_FRAME_MS / 1000))
    rms = _frame_rms(arr, frame_samples)

    if rms.size == 0:
        return arr, 0.0, 0.0

    global_rms = float(np.sqrt(np.mean(np.square(arr)) + 1e-12))
    peak = float(np.max(np.abs(arr)))

    # Low enough to retain quiet consonants, high enough to reject typical
    # digital/background silence. The cap prevents over-trimming loud voices.
    threshold = max(1e-4, min(0.0025, global_rms * 0.07, peak * 0.012))

    raw_active = rms > threshold

    # Reject isolated noise spikes: require activity in at least 2 of 5 frames.
    neighborhood = np.convolve(
        raw_active.astype(np.int16),
        np.ones(5, dtype=np.int16),
        mode="same",
    )
    active = neighborhood >= 2

    if not np.any(active):
        return arr, 0.0, 0.0

    active_indices = np.flatnonzero(active)
    first_active_sample = int(active_indices[0] * frame_samples)
    last_active_sample = min(
        arr.size,
        int((active_indices[-1] + 1) * frame_samples),
    )

    original_size = arr.size
    start_sample = 0
    end_sample = original_size

    leading_ms = first_active_sample * 1000.0 / sample_rate
    trailing_ms = (original_size - last_active_sample) * 1000.0 / sample_rate

    removed_leading_ms = 0.0
    removed_trailing_ms = 0.0

    if trim_leading and leading_ms > MAX_INTERNAL_LEADING_SILENCE_MS:
        keep = int(sample_rate * KEEP_INTERNAL_LEADING_SILENCE_MS / 1000)
        start_sample = max(0, first_active_sample - keep)
        removed_leading_ms = start_sample * 1000.0 / sample_rate

    if trim_trailing and trailing_ms > MAX_INTERNAL_TRAILING_SILENCE_MS:
        keep = int(sample_rate * KEEP_INTERNAL_TRAILING_SILENCE_MS / 1000)
        end_sample = min(original_size, last_active_sample + keep)
        removed_trailing_ms = (
            original_size - end_sample
        ) * 1000.0 / sample_rate

    if end_sample <= start_sample:
        return arr, 0.0, 0.0

    return arr[start_sample:end_sample], removed_leading_ms, removed_trailing_ms


def _active_speech_rms(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate RMS from active speech and ignore long silent tails."""
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0 or sample_rate <= 0:
        return 0.0

    frame_samples = max(1, int(sample_rate * 0.020))
    rms = _frame_rms(arr, frame_samples)
    if rms.size == 0:
        return 0.0

    floor = max(1e-4, float(np.percentile(rms, 30)) * 1.8)
    active = rms[rms > floor]

    if active.size == 0:
        return float(np.sqrt(np.mean(np.square(arr)) + 1e-12))

    # Median is resistant to breaths, plosives, and isolated peaks.
    return float(np.median(active))


def _match_loudness_to_target(
    audio: np.ndarray,
    sample_rate: int,
    target_rms: float,
) -> tuple[np.ndarray, float]:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)

    if not MATCH_CHUNK_LOUDNESS or target_rms <= 0.0:
        return arr, 1.0

    current_rms = _active_speech_rms(arr, sample_rate)
    if current_rms <= 1e-8:
        return arr, 1.0

    gain = target_rms / current_rms
    gain = float(np.clip(gain, LOUDNESS_GAIN_MIN, LOUDNESS_GAIN_MAX))

    if abs(gain - 1.0) < 0.015:
        return arr, 1.0

    matched = arr * gain
    peak = float(np.max(np.abs(matched))) if matched.size else 0.0
    if peak > 0.98:
        matched *= 0.98 / peak

    return matched.astype(np.float32), gain


def join_audio_segments(
    segments: list[tuple[np.ndarray, str]],
    sample_rate: int,
) -> np.ndarray:
    if not segments:
        raise RuntimeError("Нет аудиофрагментов для объединения.")

    prepared: list[np.ndarray] = []
    last_index = len(segments) - 1
    target_speech_rms: float | None = None

    for index, (audio, source_text) in enumerate(segments):
        is_first = index == 0
        is_last = index == last_index

        arr, removed_leading_ms, removed_trailing_ms = (
            _trim_internal_boundary_silence(
                audio,
                sample_rate,
                trim_leading=not is_first,
                trim_trailing=not is_last,
            )
        )

        if removed_leading_ms > 1.0 or removed_trailing_ms > 1.0:
            print(
                f"  Boundary trim chunk {index + 1}: "
                f"leading -{removed_leading_ms:.0f} ms, "
                f"trailing -{removed_trailing_ms:.0f} ms"
            )

        if target_speech_rms is None:
            target_speech_rms = _active_speech_rms(arr, sample_rate)
        else:
            arr, gain = _match_loudness_to_target(
                arr,
                sample_rate,
                target_speech_rms,
            )
            if abs(gain - 1.0) >= 0.015:
                print(
                    f"  Loudness match chunk {index + 1}: gain={gain:.3f}"
                )

        arr = _fade_edges(
            arr,
            sample_rate,
            fade_out_enabled=not is_last,
        )
        if arr.size == 0:
            raise RuntimeError(f"Пустой аудиофрагмент №{index + 1}")
        prepared.append(arr)

        if not is_last:
            pause_ms = _pause_after_text(source_text)
            prepared.append(
                np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32)
            )

    # Some players and audio APIs make an abrupt file ending sound clipped.
    # A real silent tail also preserves the last consonant during playback.
    if FINAL_SILENCE_MS > 0:
        prepared.append(
            np.zeros(int(sample_rate * FINAL_SILENCE_MS / 1000), dtype=np.float32)
        )

    return np.concatenate(prepared)


def reasonable_max_duration(text: str) -> float:
    """
    Conservative upper duration bound for Russian speech.

    It intentionally allows slow delivery, but rejects outputs that are
    obviously much longer than the requested passage and likely contain loops,
    noise, humming, or repeated phonemes.
    """
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text)
    letters = re.findall(r"[A-Za-zА-Яа-яЁё0-9]", text)

    by_words = len(words) * MAX_SECONDS_PER_WORD
    by_letters = len(letters) * MAX_SECONDS_PER_LETTER

    return max(
        8.0,
        by_words,
        by_letters,
    ) + MAX_DURATION_EXTRA_SEC


def looks_runaway(audio: np.ndarray, sample_rate: int, text: str) -> bool:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0 or sample_rate <= 0:
        return True

    duration = len(arr) / sample_rate
    maximum = reasonable_max_duration(text)
    return duration > maximum


def looks_truncated(audio: np.ndarray, sample_rate: int, text: str) -> bool:
    """Detect likely early EOS or an abruptly cut final phoneme.

    This is heuristic: we combine word rate, character rate, and whether the
    waveform is still active at the exact end.
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0 or sample_rate <= 0:
        return True

    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text)
    letters = len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]", text))
    duration = len(arr) / sample_rate

    # Conservative lower bounds for Russian speech. These catch a missing final
    # word more often than the previous 6.5 words/sec-only rule.
    minimum_by_words = len(words) / 5.2 if words else 0.0
    minimum_by_chars = letters / 18.0 if letters else 0.0
    too_short = len(words) >= 4 and duration < max(0.55, minimum_by_words, minimum_by_chars)

    # If the final 60 ms still contains substantial speech energy, the waveform
    # was probably cut before the final consonant or natural release.
    tail_n = max(1, int(sample_rate * 0.060))
    body_n = max(tail_n + 1, int(sample_rate * 0.250))
    tail = arr[-tail_n:]
    body = arr[-body_n:-tail_n] if len(arr) > body_n else arr[:-tail_n]
    tail_rms = float(np.sqrt(np.mean(np.square(tail, dtype=np.float64))))
    body_rms = float(np.sqrt(np.mean(np.square(body, dtype=np.float64)))) if body.size else 0.0
    active_at_end = tail_rms > max(0.010, body_rms * 0.45)

    return too_short or active_at_end


def split_chunk_in_half(text: str) -> list[str]:
    """Делит проблемный фрагмент около середины, предпочитая пунктуацию."""
    text = text.strip()
    if len(text) < 2:
        return [text]

    middle = len(text) // 2
    candidates = [
        pos
        for pos, char in enumerate(text)
        if char in ",;:" and 20 <= pos <= len(text) - 20
    ]

    if candidates:
        cut = min(candidates, key=lambda pos: abs(pos - middle)) + 1
        left, right = text[:cut].strip(), text[cut:].strip()
    else:
        spaces = [
            pos
            for pos, char in enumerate(text)
            if char.isspace() and 20 <= pos <= len(text) - 20
        ]
        if not spaces:
            return [text]
        cut = min(spaces, key=lambda pos: abs(pos - middle))
        left, right = text[:cut].strip(), text[cut:].strip()

    if left and left[-1] not in ".!?;:,":
        left += ";"
    if right and right[-1] not in ".!?;:,":
        right += "."

    return [part for part in (left, right) if part]


class LocalQwenTTS:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        backend: str = BACKEND,
        profile: str = PROFILE,
    ):
        check_cuda_compatibility()
        set_tts_seed(TTS_SEED)

        if backend not in {"official", "faster"}:
            raise ValueError("TTS_BACKEND должен быть official или faster.")
        if profile not in GENERATION_PROFILES:
            raise ValueError(
                f"Неизвестный профиль {profile!r}. "
                f"Доступно: {', '.join(GENERATION_PROFILES)}"
            )

        self.backend = backend
        self.profile_name = profile
        self.profile = GENERATION_PROFILES[profile]
        self.model_id = model_id

        print(f"Loading model: {model_id}")
        print(
            f"Backend: {backend} | profile: {profile} | "
            f"seed: {TTS_SEED} | deterministic: {TTS_DETERMINISTIC}"
        )

        if backend == "official":
            from qwen_tts import Qwen3TTSModel

            # На Windows используем SDPA: FlashAttention не обязателен.
            self.model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        else:
            from faster_qwen3_tts import FasterQwen3TTS

            self.model = FasterQwen3TTS.from_pretrained(
                model_id,
                device="cuda",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )

        print("Model loaded.")

    def _create_official_prompt(
        self,
        ref_audio: Path,
        ref_text: str,
    ):
        if self.backend != "official":
            return None

        xvec_only = not bool(ref_text)
        if xvec_only:
            print(
                "Warning: нет ref_text; используется x-vector-only. "
                "Для максимального сходства добавьте точный voices/<voice>.txt."
            )

        return self.model.create_voice_clone_prompt(
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            x_vector_only_mode=xvec_only,
        )

    def _generate_once(
        self,
        text: str,
        language: str,
        ref_audio: Path,
        ref_text: str,
        voice_prompt,
        seed: int,
        style_instruction: str = "",
        long_form: bool = False,
    ) -> tuple[np.ndarray, int]:
        set_tts_seed(seed)

        # The old fixed max_new_tokens=2048 allowed 60-170 seconds of unwanted
        # continuation after a 20-40 second passage. Use a length-aware window.
        min_new_tokens = max(
            16,
            min(768, int(len(text) * MIN_TOKEN_CHAR_FACTOR)),
        )
        max_new_tokens = max(
            min_new_tokens + 40,
            int(len(text) * MAX_TOKEN_CHAR_FACTOR) + MAX_TOKEN_EXTRA,
        )
        max_new_tokens = min(ABSOLUTE_MAX_NEW_TOKENS, max_new_tokens)

        temperature = self.profile["temperature"]
        top_k = self.profile["top_k"]
        top_p = self.profile["top_p"]
        repetition_penalty = self.profile["repetition_penalty"]

        if long_form:
            temperature = min(temperature, LONG_FORM_TEMPERATURE_CAP)
            top_k = min(top_k, LONG_FORM_TOP_K_CAP)
            top_p = min(top_p, LONG_FORM_TOP_P_CAP)
            repetition_penalty = max(
                repetition_penalty,
                LONG_FORM_REPETITION_PENALTY,
            )

        print(
            f"  token window: min={min_new_tokens}, "
            f"max={max_new_tokens}, long_form={long_form}"
        )

        common = dict(
            text=text,
            language=language,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=self.profile["do_sample"],
            repetition_penalty=repetition_penalty,
            non_streaming_mode=True,
        )

        if self.backend == "official":
            # Отдельные параметры subtalker дают согласованный режим sampling.
            common.update(
                subtalker_dosample=self.profile["do_sample"],
                subtalker_top_k=self.profile["top_k"],
                subtalker_top_p=self.profile["top_p"],
                subtalker_temperature=self.profile["temperature"],
            )
            wavs, sr = self.model.generate_voice_clone(
                voice_clone_prompt=voice_prompt,
                **common,
            )
        else:
            faster_kwargs = dict(
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                xvec_only=not bool(ref_text),
                append_silence=True,
                **common,
            )
            if style_instruction:
                faster_kwargs["instruct"] = style_instruction

            wavs, sr = self.model.generate_voice_clone(**faster_kwargs)

        if not wavs:
            raise RuntimeError("Модель вернула пустой список аудио.")

        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if audio.size <= 1:
            raise RuntimeError("Модель вернула пустой аудиосигнал.")

        return audio, int(sr)

    def _generate_with_recovery(
        self,
        text: str,
        language: str,
        ref_audio: Path,
        ref_text: str,
        voice_prompt,
        chunk_index: int,
        style_instruction: str = "",
        is_final_chunk: bool = False,
        depth: int = 0,
    ) -> list[tuple[np.ndarray, str, int]]:
        last_audio: np.ndarray | None = None
        last_sr: int | None = None

        # Start with one candidate. Additional attempts are allowed only when
        # the tail detector sees a likely truncation.
        max_attempts = FINAL_CHUNK_ATTEMPTS if is_final_chunk else 2

        for attempt in range(max_attempts):
            seed = generation_seed(
                text=text,
                chunk_index=chunk_index,
                attempt=attempt,
                depth=depth,
            )
            print(
                f"Generating chunk {chunk_index}, attempt {attempt + 1}, "
                f"depth={depth}, seed={seed}: {text}"
            )

            audio, sr = self._generate_once(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                voice_prompt=voice_prompt,
                seed=seed,
                style_instruction=style_instruction,
                long_form=False,
            )
            last_audio, last_sr = audio, sr

            duration = len(audio) / sr
            maximum_duration = reasonable_max_duration(text)
            truncated = looks_truncated(audio, sr, text)
            runaway = looks_runaway(audio, sr, text)

            if runaway:
                state = (
                    f" | runaway/artefact tail "
                    f"(max expected {maximum_duration:.2f} sec)"
                )
            elif truncated:
                state = " | possible truncation"
            else:
                state = " | tail OK"

            print(
                f"  generated duration: {duration:.2f} sec" + state
            )

            # The first complete and reasonably sized candidate is kept.
            if not truncated and not runaway:
                return [(audio, text, sr)]

            print(
                "  Warning: результат оборван либо содержит слишком длинный "
                "артефактный хвост; повторяем генерацию."
            )

            if not RETRY_ONLY_IF_TRUNCATED:
                continue

        if (
            depth < MAX_RETRY_DEPTH
            and len(text) >= 80
            and len(split_chunk_in_half(text)) > 1
        ):
            print("  Автоматически делю проблемный фрагмент на меньшие части.")
            result: list[tuple[np.ndarray, str, int]] = []
            sub_chunks = split_chunk_in_half(text)

            for sub_index, sub_text in enumerate(sub_chunks, start=1):
                result.extend(
                    self._generate_with_recovery(
                        text=sub_text,
                        language=language,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        voice_prompt=voice_prompt,
                        chunk_index=chunk_index * 10 + sub_index,
                        style_instruction=style_instruction,
                        is_final_chunk=(
                            is_final_chunk and sub_index == len(sub_chunks)
                        ),
                        depth=depth + 1,
                    )
                )

            return result

        if last_audio is None or last_sr is None:
            raise RuntimeError("Не удалось получить аудио.")

        print("  Warning: сохранить удалось, но фрагмент может быть укорочен.")
        return [(last_audio, text, last_sr)]

    def speak(
        self,
        text: str,
        voice_name: str,
        language: str = "Russian",
        out_file: str | Path = "answer.wav",
        play_audio: bool = True,
        instruct: str = "",
    ) -> Path:
        voices = find_voices()
        if voice_name not in voices:
            available = ", ".join(voices) if voices else "голоса не найдены"
            raise ValueError(
                f"Неизвестный голос {voice_name!r}. Доступно: {available}"
            )

        ref_audio = voices[voice_name]
        ref_text = read_reference_text(ref_audio)

        # Priority: explicit argument -> voices/<voice>.style.txt -> env.
        if not instruct.strip():
            instruct = read_style_instruction(ref_audio)
        style_instruction = build_clone_instruction(instruct)

        validate_reference_audio(ref_audio)

        print(f"Voice: {voice_name}")
        print(f"Reference audio: {ref_audio}")
        if ref_text:
            print(f"Reference text ({len(ref_text)} chars): {ref_text}")
            if len(ref_text) > MAX_REF_TEXT_CHARS_WARNING:
                print(
                    "Warning: reference text слишком длинный. Для стабильного "
                    "клонирования лучше 5–20 секунд чистой речи и точная расшифровка."
                )
        else:
            print(
                "Warning: ref_text отсутствует. Будет использован менее точный "
                "x-vector-only режим."
            )

        if style_instruction:
            print(f"Style instruction: {style_instruction}")
            if self.backend == "official":
                print(
                    "Warning: официальный Base API не документирует надёжное "
                    "управление instruct при voice clone. Инструкция будет "
                    "проигнорирована. Используйте TTS_BACKEND=faster."
                )
                style_instruction = ""
        else:
            print(
                "Style instruction: empty — дополнительное управление отключено; "
                "манера берётся из reference audio."
            )

        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(
                f"Текст слишком длинный: {len(text)} символов. "
                f"Лимит: {MAX_INPUT_CHARS}."
            )

        prepared = clean_text_before_tts(text)
        prepared = normalize_text_for_tts(prepared, language)
        prepared = apply_stress_dictionary(prepared, language)
        prepared = clean_text_before_tts(prepared)

        dictionary = load_stress_dictionary()
        if dictionary.get("debug_print_prepared_text", True):
            print(f"Prepared text ({len(prepared)} chars): {prepared}")

        voice_prompt = self._create_official_prompt(ref_audio, ref_text)

        generated: list[tuple[np.ndarray, str, int]] = []
        expected_sr: int | None = None
        whole_text_succeeded = False

        if WHOLE_TEXT_FIRST and len(prepared) <= WHOLE_TEXT_MAX_CHARS:
            print(
                "Continuity mode: trying the whole text in one generation "
                f"({len(prepared)} chars, limit={WHOLE_TEXT_MAX_CHARS})."
            )

            whole_seed = generation_seed(
                text=prepared,
                chunk_index=1,
                attempt=0,
                depth=0,
            )

            whole_audio, whole_sr = self._generate_once(
                text=prepared,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                voice_prompt=voice_prompt,
                seed=whole_seed,
                style_instruction=style_instruction,
                long_form=True,
            )

            whole_duration = len(whole_audio) / whole_sr
            maximum_duration = reasonable_max_duration(prepared)
            whole_truncated = looks_truncated(
                whole_audio,
                whole_sr,
                prepared,
            )
            whole_runaway = looks_runaway(
                whole_audio,
                whole_sr,
                prepared,
            )

            if whole_runaway:
                status = (
                    " | duration is implausibly long "
                    f"(max expected {maximum_duration:.2f} sec), "
                    "fallback to chunks"
                )
            elif whole_truncated:
                status = " | possible truncation, fallback to chunks"
            else:
                status = " | tail OK, no chunk transition"

            print(
                f"  whole-text duration: {whole_duration:.2f} sec"
                + status
            )

            if not whole_truncated and not whole_runaway:
                generated.append((whole_audio, prepared, whole_sr))
                expected_sr = whole_sr
                whole_text_succeeded = True

        if not whole_text_succeeded:
            chunks = split_long_text(prepared, MAX_CHUNK_CHARS)
            if not chunks:
                raise RuntimeError("После подготовки не осталось текста.")

            print(
                f"Text split into {len(chunks)} chunk(s), "
                f"max={MAX_CHUNK_CHARS} chars."
            )
            for i, chunk in enumerate(chunks, start=1):
                print(f"  [{i}/{len(chunks)}] {len(chunk)} chars: {chunk}")

            for index, chunk in enumerate(chunks, start=1):
                parts = self._generate_with_recovery(
                    text=chunk,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    voice_prompt=voice_prompt,
                    chunk_index=index,
                    style_instruction=style_instruction,
                    is_final_chunk=(index == len(chunks)),
                )
                for audio, source_text, sr in parts:
                    if expected_sr is None:
                        expected_sr = sr
                    elif sr != expected_sr:
                        raise RuntimeError(
                            f"Несовпадение sample rate: {expected_sr} и {sr}"
                        )
                    generated.append((audio, source_text, sr))

        if expected_sr is None:
            raise RuntimeError("Модель не вернула sample rate.")

        joined = join_audio_segments(
            [(audio, source_text) for audio, source_text, _ in generated],
            expected_sr,
        )

        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, joined, expected_sr)

        print(f"Saved audio: {out_path.resolve()}")
        print(f"Duration: {len(joined) / expected_sr:.2f} sec")

        if play_audio:
            try:
                sd.play(joined, expected_sr)
                sd.wait()
            except KeyboardInterrupt:
                sd.stop()
                print("\nВоспроизведение остановлено.")

        return out_path


def main() -> None:
    voices = find_voices()
    if not voices:
        print("WAV-голоса не найдены.")
        print(f"Положите reference в: {VOICES_DIR.resolve()}")
        print("Пример: voices/1.wav и voices/1.txt")
        return

    print("Available voices:")
    for name, path in voices.items():
        has_text = path.with_suffix(".txt").exists()
        print(
            f"  - {name} | {path} | "
            f"ref_text={'yes' if has_text else 'no'}"
        )

    try:
        voice = input("\nChoose voice: ").strip()
        if not voice:
            voice = next(iter(voices))

        language = (
            input(
                "Language, for example Russian/English/Chinese [Russian]: "
            ).strip()
            or "Russian"
        )

        saved_style = read_style_instruction(voices[voice])
        if saved_style:
            print(f"Saved voice style: {saved_style}")

        instruct = input(
            "Style instruction [Enter = reference manner / saved style]: "
        ).strip()
        if not instruct:
            instruct = saved_style

        tts = LocalQwenTTS()

        print(f"\nВведите текст. Лимит: {MAX_INPUT_CHARS} символов.")
        print("Для файла используйте: file:input.txt")
        print("Для выхода: q")

        counter = 1
        while True:
            try:
                text = input("\nText: ").strip()
            except KeyboardInterrupt:
                print("\nОстановлено пользователем.")
                break

            if text.lower() in {"q", "quit", "exit"}:
                break
            if not text:
                continue

            if text.lower().startswith("file:"):
                input_path = Path(text[5:].strip().strip('"'))
                try:
                    text = input_path.read_text(encoding="utf-8").strip()
                    print(
                        f"Loaded: {input_path} ({len(text)} chars)"
                    )
                except Exception as exc:
                    print(f"Не удалось прочитать файл: {exc}")
                    continue

            out_file = OUTPUT_DIR / f"{voice}_{counter:03d}.wav"
            counter += 1

            try:
                tts.speak(
                    text=text,
                    voice_name=voice,
                    language=language,
                    out_file=out_file,
                    play_audio=True,
                    instruct=instruct,
                )
            except Exception as exc:
                print(f"TTS error: {type(exc).__name__}: {exc}")

    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")




if __name__ == "__main__":
    main()
