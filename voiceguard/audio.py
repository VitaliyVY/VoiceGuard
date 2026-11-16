"""Читання й підготовка мовного сигналу."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal

SAMPLE_RATE = 16000


def load(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Моно-сигнал float32 у діапазоні [-1, 1] з частотою sr (WAV, FLAC, OGG)."""
    import soundfile as sf

    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return resample(x, file_sr, sr)


def save(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Передискретизація поліфазним фільтром (раціональне відношення частот)."""
    if sr_from == sr_to:
        return x.astype(np.float32)
    g = gcd(sr_from, sr_to)
    return signal.resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Пікове нормування: ознаки не мають залежати від гучності запису."""
    m = float(np.abs(x).max())
    return (x * (peak / m)).astype(np.float32) if m > 0 else x


def frame_energy_db(x: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    """Енергія кадрів, дБ (для виявлення тиші)."""
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    e = (x[idx] ** 2).mean(axis=1)
    return 10.0 * np.log10(e + 1e-12)


def trim_silence(x: np.ndarray, threshold_db: float = 35.0, frame: int = 400, hop: int = 160,
                 margin: int = 2) -> np.ndarray:
    """Відкидає тишу на початку й у кінці запису.

    Кадр вважається мовою, якщо його енергія не нижча за максимум мінус
    threshold_db. Важливо для ASVspoof 2019 LA: тривалість тиші на краях
    відрізняється у справжніх і синтезованих записах, і класифікатор може
    навчитися розпізнавати тишу замість голосу (Müller et al., 2021).
    """
    e = frame_energy_db(x, frame, hop)
    voiced = np.flatnonzero(e >= e.max() - threshold_db)
    if len(voiced) == 0:
        return x
    start = max(voiced[0] - margin, 0) * hop
    stop = min((voiced[-1] + margin) * hop + frame, len(x))
    return x[start:stop]
