"""Пороги для адаптованого детектора (models/cm_ssl_uk.pkl) — щоб програма могла ним користуватися.

Правило те саме, що й для англомовної моделі: dev ASVspoof (усі 10 каналів);
«м'яка» — EER, «звичайна» / «сувора» — 10 % / 20 % справжніх dev відхилено.
Пороги й розкид оцінок записуються в той самий файл моделі.

Приклад:
    python scripts/uk_calibrate.py
"""

from __future__ import annotations

import pickle

import numpy as np

from _common import DATA, ROOT
from cm_gmm import subset
from voiceguard import data, metrics

CH = ["clean", "opus32", "opus16", "opus8", "telephone", "amrnb", "amrwb", "gsm", "noise20", "noise10"]


def main() -> None:
    path = ROOT / "models" / "cm_ssl_uk.pkl"
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    layers = [l - 1 for l in d["layers"]]
    y = np.array([it.bonafide for it in subset(data.cm_protocol(DATA, "dev"), 100, 0)])
    s = np.concatenate([d["model"].decision_function(
        np.load(ROOT / "data" / "cache" / f"ssl_dev_{c}.npy", mmap_mode="r")[:, layers].astype(np.float32).mean(1))
        for c in CH])
    b = np.tile(y, len(CH))
    d["thresholds"] = {"м'яка": float(metrics.eer(s[b], s[~b])[1]), "звичайна": float(np.quantile(s[b], 0.10)),
                       "сувора": float(np.quantile(s[b], 0.20))}
    d["scale"] = float(s.std())
    with open(path, "wb") as fh:
        pickle.dump(d, fh)
    print(d["thresholds"], "розкид", round(d["scale"], 3))
    uk = d["model"].decision_function(np.load(ROOT / "data" / "cache" / "uk_real_test_clean.npy")[:, layers]
                                      .astype(np.float32).mean(1))
    for k, t in d["thresholds"].items():
        print(f"  {k}: справжніх українських (Common Voice test) відхилено {100 * np.mean(uk < t):.1f} %")


if __name__ == "__main__":
    main()
