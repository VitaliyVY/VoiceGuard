"""Метрики верифікації та виявлення підробок.

Оцінка (score) більша — більше довіри, що запис справжній / мовець той самий.
  FAR (false acceptance) — частка «чужих» або підроблених, що пройшли поріг;
  FRR (false rejection)  — частка справжніх, відкинутих порогом.
  EER — рівень помилок у точці, де FAR = FRR (основна метрика ASVspoof).
"""

from __future__ import annotations

import numpy as np


def far_frr(target: np.ndarray, nontarget: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FAR і FRR для всіх порогів (поріг — кожна з оцінок)."""
    target, nontarget = np.asarray(target, float), np.asarray(nontarget, float)
    scores = np.concatenate([target, nontarget])
    labels = np.concatenate([np.ones(len(target)), np.zeros(len(nontarget))])
    order = np.argsort(scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    # поріг між сусідніми оцінками: відкидається все, що нижче порогу
    frr = np.concatenate([[0.0], np.cumsum(labels) / len(target)])
    far = np.concatenate([[1.0], 1.0 - np.cumsum(1 - labels) / len(nontarget)])
    thresholds = np.concatenate([[scores[0] - 1e-6], scores + 1e-6])
    return far, frr, thresholds


def eer(target: np.ndarray, nontarget: np.ndarray) -> tuple[float, float]:
    """EER і відповідний поріг (лінійна інтерполяція між сусідніми точками)."""
    far, frr, thr = far_frr(target, nontarget)
    i = int(np.argmax(frr >= far))
    if i == 0:
        return float(frr[0]), float(thr[0])
    # перетин відрізків (far[i-1], frr[i-1]) → (far[i], frr[i])
    d0, d1 = far[i - 1] - frr[i - 1], far[i] - frr[i]
    t = d0 / (d0 - d1) if d0 != d1 else 0.0
    value = far[i - 1] + t * (far[i] - far[i - 1])
    return float(value), float(thr[i - 1] + t * (thr[i] - thr[i - 1]))


def min_dcf(target: np.ndarray, nontarget: np.ndarray, p_target: float = 0.05,
            c_miss: float = 1.0, c_fa: float = 1.0) -> float:
    """Мінімальна нормована функція вартості помилок (detection cost function)."""
    far, frr, _ = far_frr(target, nontarget)
    dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    return float(dcf.min() / min(c_miss * p_target, c_fa * (1.0 - p_target)))
