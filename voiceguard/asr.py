"""Розпізнавання мовлення (голос → текст) моделлю Whisper small.

Використовується faster-whisper (CTranslate2) з 8-бітним квантуванням: на
процесорі він у ~5 разів швидший за звичайну реалізацію за тієї самої якості
(30 с українського мовлення ≈ 8 с на ноутбуці без відеокарти). Модель
завантажується один раз (~480 МБ) і кешується.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODEL_SIZE = "small"
_MODEL = None


@dataclass
class Segment:
    start: float
    end: float
    text: str


def model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel

        _MODEL = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
    return _MODEL


def transcribe(x: np.ndarray, language: str | None = "uk") -> tuple[list[Segment], str]:
    """Сегменти з часом і мова. language=None — визначити автоматично (уk/ru/en…)."""
    segs, info = model().transcribe(np.asarray(x, dtype=np.float32), language=language, beam_size=1,
                                    vad_filter=True, condition_on_previous_text=False)
    out = [Segment(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in segs if s.text.strip()]
    return out, info.language
