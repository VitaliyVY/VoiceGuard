"""Моделі на гаусових сумішах (GMM).

Детектор підробки (countermeasure, CM): дві суміші — справжня мова і підробка;
оцінка запису — середнє за кадрами логарифмічне відношення правдоподібностей
    s = (1/T) Σ_t [log p(x_t | bonafide) − log p(x_t | spoof)].
s > поріг — запис справжній. Це базова система змагання ASVspoof 2019
(LFCC-GMM, Todisco et al., 2019).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture


def fit_gmm(frames: np.ndarray, n_components: int, seed: int = 0, max_iter: int = 100) -> GaussianMixture:
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag", max_iter=max_iter,
                          reg_covar=1e-4, random_state=seed, init_params="k-means++")
    return gmm.fit(frames.astype(np.float64))


@dataclass
class GMMCountermeasure:
    bonafide: GaussianMixture
    spoof: GaussianMixture

    @classmethod
    def fit(cls, bonafide_frames: np.ndarray, spoof_frames: np.ndarray, n_components: int = 128,
            seed: int = 0) -> "GMMCountermeasure":
        return cls(fit_gmm(bonafide_frames, n_components, seed), fit_gmm(spoof_frames, n_components, seed))

    def score(self, feat: np.ndarray) -> float:
        f = feat.astype(np.float64)
        return float(np.mean(self.bonafide.score_samples(f) - self.spoof.score_samples(f)))


@dataclass
class GMMUBMVerifier:
    """Верифікація мовця GMM-UBM (Reynolds et al., 2000).

    UBM — універсальна модель фону: суміш, навчена на мові багатьох людей.
    Модель мовця — UBM, у якої середні зсунуто до його записів апостеріорною
    (MAP) адаптацією:
        n_k = Σ_t γ_k(t),  E_k = (1/n_k) Σ_t γ_k(t) x_t,
        α_k = n_k / (n_k + r),  μ_k' = α_k E_k + (1 − α_k) μ_k,
    де γ_k(t) — апостеріорна ймовірність компоненти k для кадру t, r — фактор
    релевантності. Компоненти, яких мало «торкнулися» записи мовця, лишаються
    як в UBM. Оцінка перевірки — середнє за кадрами log p(x|мовець) − log p(x|UBM).
    """

    ubm: GaussianMixture
    relevance: float = 16.0

    @classmethod
    def fit_ubm(cls, frames: np.ndarray, n_components: int = 256, seed: int = 0,
                relevance: float = 16.0) -> "GMMUBMVerifier":
        return cls(fit_gmm(frames, n_components, seed), relevance)

    def enroll(self, feats: list[np.ndarray]) -> np.ndarray:
        """Адаптовані середні моделі мовця, форма (компонент, ознак)."""
        x = np.concatenate(feats).astype(np.float64)
        gamma = self.ubm.predict_proba(x)
        n = gamma.sum(axis=0)
        e = (gamma.T @ x) / np.maximum(n, 1e-10)[:, None]
        alpha = (n / (n + self.relevance))[:, None]
        return alpha * e + (1.0 - alpha) * self.ubm.means_

    def score(self, speaker_means: np.ndarray, feat: np.ndarray) -> float:
        x = feat.astype(np.float64)
        return float(np.mean(_diag_gmm_logpdf(x, self.ubm.weights_, speaker_means, self.ubm.covariances_)
                             - self.ubm.score_samples(x)))


def _diag_gmm_logpdf(x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """log p(x) для суміші з діагональними коваріаціями (по кадрах)."""
    prec = 1.0 / variances
    log_det = np.sum(np.log(variances), axis=1)
    # (x − μ)ᵀ Λ (x − μ) = xᵀΛx − 2 xᵀΛμ + μᵀΛμ, без матриць кадр × компонента × ознака
    quad = (x ** 2) @ prec.T - 2.0 * x @ (means * prec).T + np.sum(means ** 2 * prec, axis=1)
    log_comp = np.log(weights) - 0.5 * (x.shape[1] * np.log(2 * np.pi) + log_det + quad)
    m = log_comp.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(log_comp - m).sum(axis=1, keepdims=True)))[:, 0]


def sample_frames(feats: list[np.ndarray], max_frames: int, seed: int = 0) -> np.ndarray:
    """Випадкова рівномірна вибірка кадрів із набору записів (обмеження пам'яті й часу навчання)."""
    rng = np.random.default_rng(seed)
    all_frames = np.concatenate(feats)
    if len(all_frames) <= max_frames:
        return all_frames
    return all_frames[rng.choice(len(all_frames), max_frames, replace=False)]
