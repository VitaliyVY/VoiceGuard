"""Ознаки самонавченої моделі мовлення WavLM Base+ (Chen et al., 2022).

Модель попередньо навчена без розмітки на 94 тис. годин мовлення (з
доданим шумом і накладанням голосів), тож її внутрішні подання описують
голос значно багатше, ніж LFCC. Модель не донавчається (на процесорі це
неможливо): з кожного з 12 трансформерних шарів береться середнє й
стандартне відхилення за часом, а класифікатор навчається поверх цих
ознак. Нижні й середні шари зберігають акустичні деталі (корисні для
виявлення синтезу), верхні — переважно зміст мовлення.
"""

from __future__ import annotations

import numpy as np

MAX_SECONDS = 4.0
_MODEL = None


def model(threads: int = 3):
    global _MODEL
    if _MODEL is None:
        import torch
        import torchaudio

        torch.set_num_threads(threads)
        _MODEL = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model().eval()
    return _MODEL


FRAME_RATE = 50          # кадрів WavLM за секунду (крок 20 мс)


def frame_states(x: np.ndarray, layers: list[int], max_seconds: float = 30.0) -> np.ndarray:
    """Стани вибраних шарів для кожного кадру: (кадрів, шарів, 768). Один прохід моделі по запису."""
    import torch

    x = np.asarray(x[: int(16000 * max_seconds)], dtype=np.float32)
    wav = torch.from_numpy(x)[None]
    wav = (wav - wav.mean()) / (wav.std() + 1e-7)
    with torch.inference_mode():
        hidden, _ = model().extract_features(wav)
    return np.stack([hidden[l][0].numpy() for l in layers], axis=1)


def window_stats(states: np.ndarray) -> np.ndarray:
    """Ознаки вікна з кадрових станів: середнє й ст. відхилення кожного шару, усереднені за шарами (1536,)."""
    return np.concatenate([states.mean(0), states.std(0)], axis=1).mean(0)


def layer_stats(x: np.ndarray, sr: int = 16000, max_seconds: float = MAX_SECONDS) -> np.ndarray:
    """Ознаки запису: (шарів, 2 · 768) — середнє і ст. відхилення кожного шару за часом.

    Довгі записи обрізаються до центрального вікна max_seconds (час обчислення).
    """
    import torch

    n = int(sr * max_seconds)
    if len(x) > n:
        start = (len(x) - n) // 2
        x = x[start:start + n]
    wav = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    wav = (wav - wav.mean()) / (wav.std() + 1e-7)      # нормування, як під час попереднього навчання
    with torch.inference_mode():
        layers, _ = model().extract_features(wav)
    return np.stack([np.concatenate([h[0].mean(0).numpy(), h[0].std(0).numpy()]) for h in layers]).astype(np.float16)
