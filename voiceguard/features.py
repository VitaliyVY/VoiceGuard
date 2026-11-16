"""Спектральні ознаки мовлення.

Сигнал → попереднє підсилення високих частот → кадри (вікно Хеммінга) →
спектр потужності ДПФ → банк трикутних фільтрів → логарифм → ДКП-II
(кепстр) → динамічні ознаки Δ, ΔΔ.

  MFCC — фільтри рівномірні за шкалою мел (як сприймає вухо; основна ознака
         для впізнавання мовця);
  LFCC — фільтри рівномірні за лінійною частотою: краще описують високі
         частоти, де вокодери синтезу залишають артефакти (базова ознака
         змагання ASVspoof 2019).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, rfft

from .audio import SAMPLE_RATE


@dataclass(frozen=True)
class FrameConfig:
    sr: int = SAMPLE_RATE
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    n_fft: int = 512
    preemphasis: float = 0.97

    @property
    def frame(self) -> int:
        return int(round(self.sr * self.frame_ms / 1000))

    @property
    def hop(self) -> int:
        return int(round(self.sr * self.hop_ms / 1000))


def power_spectrum(x: np.ndarray, cfg: FrameConfig = FrameConfig()) -> np.ndarray:
    """Спектр потужності кадрів, форма (кадрів, n_fft // 2 + 1)."""
    x = np.asarray(x, dtype=np.float64)
    if cfg.preemphasis:
        x = np.append(x[0], x[1:] - cfg.preemphasis * x[:-1])
    if len(x) < cfg.frame:
        x = np.pad(x, (0, cfg.frame - len(x)))
    n = 1 + (len(x) - cfg.frame) // cfg.hop
    idx = np.arange(cfg.frame)[None, :] + cfg.hop * np.arange(n)[:, None]
    frames = x[idx] * np.hamming(cfg.frame)
    return np.abs(rfft(frames, n=cfg.n_fft, axis=1)) ** 2 / cfg.n_fft


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def triangular_filterbank(edges_hz: np.ndarray, n_fft: int, sr: int) -> np.ndarray:
    """Трикутні фільтри з вершинами edges_hz[1:-1], форма (фільтрів, n_fft // 2 + 1)."""
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    lo, mid, hi = edges_hz[:-2, None], edges_hz[1:-1, None], edges_hz[2:, None]
    up = (freqs[None, :] - lo) / np.maximum(mid - lo, 1e-9)
    down = (hi - freqs[None, :]) / np.maximum(hi - mid, 1e-9)
    return np.maximum(0.0, np.minimum(up, down))


def mel_filterbank(n_filters: int = 40, n_fft: int = 512, sr: int = SAMPLE_RATE,
                   fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2.0
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2))
    return triangular_filterbank(edges, n_fft, sr)


def linear_filterbank(n_filters: int = 20, n_fft: int = 512, sr: int = SAMPLE_RATE,
                      fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2.0
    return triangular_filterbank(np.linspace(fmin, fmax, n_filters + 2), n_fft, sr)


def deltas(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """Регресійні похідні за часом: Δc_t = Σ n (c_{t+n} − c_{t−n}) / (2 Σ n²)."""
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    num = sum(n * (padded[width + n: len(feat) + width + n] - padded[width - n: len(feat) + width - n])
              for n in range(1, width + 1))
    return num / (2.0 * sum(n * n for n in range(1, width + 1)))


def cepstrum(spec: np.ndarray, bank: np.ndarray, n_ceps: int) -> np.ndarray:
    """ДКП-II від логарифма енергій фільтрів, перші n_ceps коефіцієнтів (разом із c0)."""
    energies = spec @ bank.T
    return dct(np.log(energies + 1e-10), type=2, axis=1, norm="ortho")[:, :n_ceps]


def with_deltas(c: np.ndarray) -> np.ndarray:
    d = deltas(c)
    return np.hstack([c, d, deltas(d)])


def lfcc(x: np.ndarray, n_filters: int = 20, n_ceps: int = 20, cfg: FrameConfig = FrameConfig(),
         fmax: float | None = None) -> np.ndarray:
    """LFCC + Δ + ΔΔ (60 ознак на кадр), як у базовій системі ASVspoof 2019.

    fmax обмежує смугу банку фільтрів: з fmax = 4000 ознаки беруться лише з
    частот, які передає будь-який канал зв'язку (телефон, AMR-NB, GSM — до 3,4–4 кГц).
    """
    bank = linear_filterbank(n_filters, cfg.n_fft, cfg.sr, fmax=fmax)
    return with_deltas(cepstrum(power_spectrum(x, cfg), bank, n_ceps)).astype(np.float32)


def mfcc(x: np.ndarray, n_filters: int = 40, n_ceps: int = 20, cfg: FrameConfig = FrameConfig()) -> np.ndarray:
    """MFCC + Δ + ΔΔ (60 ознак на кадр)."""
    bank = mel_filterbank(n_filters, cfg.n_fft, cfg.sr)
    return with_deltas(cepstrum(power_spectrum(x, cfg), bank, n_ceps)).astype(np.float32)


def cmvn(feat: np.ndarray) -> np.ndarray:
    """Нормування середнього й дисперсії по запису (прибирає сталі спотворення каналу)."""
    return ((feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)).astype(np.float32)
