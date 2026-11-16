"""Об'єднання двох детекторів підробки на рівні оцінок (score fusion).

Оцінки систем мають різні шкали (GMM — логарифм відношення правдоподібностей,
CNN — логіт), тому кожну спершу нормують: z = (s − μ) / σ, де μ, σ — середнє
й розкид оцінок системи на dev (усі канали разом). Об'єднана оцінка
    s = w · z₁ + (1 − w) · z₂,
вага w підбирається за середнім EER на dev по всіх каналах; eval — лише звіт.
Використовуються записи, які оцінено обома системами.

Приклад:
    python scripts/fuse.py results/cm_scores_lfcc4k_trim_g128_aug-clean-opus16-telephone-noise20.csv \
        results/cm_cnn_scores_lfcc4k_trim_aug-clean-opus16-telephone-noise20.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from _common import ROOT, print_table
from voiceguard import metrics

UNSEEN = {"opus32", "opus8", "amrnb", "amrwb", "gsm", "noise10"}


def load(path: str) -> dict:
    """(split, channel) → {utt: (score, bonafide)}"""
    out: dict = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            out[(r["split"], r["channel"])][r["utt"]] = (float(r["score"]), r["bonafide"] == "1")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--names", nargs=2, default=["D (GMM)", "F (LCNN)"])
    ap.add_argument("--out", default="results/fusion_D_F.json")
    args = ap.parse_args()

    a, b = load(args.first), load(args.second)
    keys = sorted(set(a) & set(b))
    channels = sorted({c for _, c in keys}, key=lambda c: list(dict.fromkeys(c for _, c in keys)).index(c))
    common = {k: sorted(set(a[k]) & set(b[k])) for k in keys}

    def norm_stats(src):
        dev = np.concatenate([[src[k][u][0] for u in common[k]] for k in keys if k[0] == "dev"])
        return dev.mean(), dev.std()

    (ma, sa), (mb, sb) = norm_stats(a), norm_stats(b)

    def eers(w: float, split: str) -> dict[str, float]:
        res = {}
        for (sp, ch) in keys:
            if sp != split:
                continue
            utts = common[(sp, ch)]
            s = np.array([w * (a[(sp, ch)][u][0] - ma) / sa + (1 - w) * (b[(sp, ch)][u][0] - mb) / sb for u in utts])
            bona = np.array([a[(sp, ch)][u][1] for u in utts])
            res[ch] = metrics.eer(s[bona], s[~bona])[0]
        return res

    grid = np.round(np.linspace(0, 1, 11), 2)
    dev_mean = {float(w): float(np.mean(list(eers(w, "dev").values()))) for w in grid}
    best_w = min(dev_mean, key=dev_mean.get)
    print("Середній EER на dev за вагою першої системи:",
          ", ".join(f"{w:.1f}: {v * 100:.1f}" for w, v in dev_mean.items()))
    print(f"Вага за dev: w = {best_w:.1f}  (записів dev на канал: {len(common[keys[0]])})")

    report = {"weight": best_w, "dev_mean_by_weight": dev_mean, "eval": {}}
    rows = []
    for label, w in ((args.names[0], 1.0), (args.names[1], 0.0), ("Об'єднання", best_w)):
        e = eers(w, "eval")
        report["eval"][label] = e
        unseen = np.mean([v for c, v in e.items() if c in UNSEEN])
        rows.append([label] + [f"{e[c] * 100:.1f}" for c in channels] +
                    [f"{np.mean(list(e.values())) * 100:.1f}", f"{unseen * 100:.1f}"])
    print("\nEER на eval, %:")
    print_table(rows, ["система", *channels, "середнє", "нові канали"])
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Звіт:", args.out)


if __name__ == "__main__":
    main()
