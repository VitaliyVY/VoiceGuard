"""Повна система для різних порогів детектора (оцінки зберігає tandem.py, пороги — з dev).

Приклад:
    python scripts/tandem_points.py
"""

from __future__ import annotations

import csv
import json
import pickle
from collections import defaultdict

import numpy as np

from _common import ROOT, print_table
from voiceguard import metrics
from voiceguard.system import _dev_threshold


def main() -> None:
    regime = pickle.load(open(ROOT / "models" / "cm_ssl.pkl", "rb"))["regime"]
    thresholds, _ = _dev_threshold(ROOT / "results" / "cm_ssl_scores.csv", regime)
    cm = defaultdict(dict)
    with open(ROOT / "results" / "tandem_cm_scores_ssl.csv", encoding="utf-8") as fh:
        for line in fh:
            ch, utt, s = line.strip().split(";")
            cm[ch][utt] = float(s)
    trials = defaultdict(list)
    with open(ROOT / "results" / "asv_scores_mfcc_trim_g256_eval.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r["channel"] in cm:
                trials[r["channel"]].append((r["utt"], r["key"], float(r["score"])))
    rows, report = [], {}
    for ch, tr in trials.items():
        key = np.array([k for _, k, _ in tr])
        asv = np.array([s for _, _, s in tr])
        thr = metrics.eer(asv[key == "target"], asv[key == "nontarget"])[1]
        acc_asv = asv >= thr
        cms = np.array([cm[ch][u] for u, _, _ in tr])
        for name, acc in [("лише верифікація", acc_asv)] + [(f"+ WavLM, {k}", acc_asv & (cms >= t))
                                                              for k, t in thresholds.items()]:
            r = {"FRR": float(1 - acc[key == "target"].mean()), "FAR": float(acc[key == "nontarget"].mean()),
                 "SPF": float(acc[key == "spoof"].mean())}
            report.setdefault(ch, {})[name] = r
            rows.append([ch, name, f"{100 * r['FRR']:.1f}", f"{100 * r['FAR']:.1f}", f"{100 * r['SPF']:.1f}"])
    print_table(rows, ["канал", "система", "власників відхилено, %", "чужих прийнято, %", "підробок прийнято, %"])
    (ROOT / "results" / "tandem_points.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
