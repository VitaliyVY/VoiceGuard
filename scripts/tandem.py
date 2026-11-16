"""Повна система: верифікація мовця (GMM-UBM) + детектор підробки (каскад).

Спроба приймається, лише якщо обидві підсистеми «за»:
    оцінка ASV ≥ θ_asv  і  оцінка CM ≥ θ_cm.
θ_asv — поріг EER «свій/чужий» (як у протоколі ASVspoof); θ_cm — поріг EER
детектора на dev, усі канали разом (поріг не підлаштовується під канал, бо
в реальній системі канал невідомий). Звітується по каналах:
  FRR    — частка справжніх власників, яких система відхилила;
  FAR    — частка інших людей, яких прийнято;
  SPF-AR — частка підробок голосу власника, яких прийнято.

Приклад:
    python scripts/tandem.py --cm models/cm_gmm_lfcc4k_trim_g128_aug-clean-opus16-telephone-noise20.pkl --feature lfcc4k
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import DATA, ROOT, print_table
from cm_gmm import FEATURES, extract
from voiceguard import data, metrics

_CM = None


def _init(path: str) -> None:
    global _CM
    with open(path, "rb") as fh:
        _CM = pickle.load(fh)


def cm_task(task) -> float:
    path, feature, chan = task
    if feature == "ssl":                                    # WavLM + логістична регресія
        from voiceguard import audio, channel, ssl

        x = audio.trim_silence(channel.CHANNELS[chan](audio.load(path)))
        stats = ssl.layer_stats(x).astype(np.float32)
        layers = [l - 1 for l in _CM["layers"]]
        return float(_CM["model"].decision_function(stats[layers].mean(axis=0)[None])[0])
    return _CM.score(extract(path, feature, True, chan))


def cm_threshold(scores_csv: str, variant: str | None = None) -> float:
    s, b = [], []
    with open(scores_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r["split"] == "dev" and (variant is None or r.get("variant") == variant):
                s.append(float(r["score"]))
                b.append(r["bonafide"] == "1")
    s, b = np.array(s), np.array(b)
    return metrics.eer(s[b], s[~b])[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--asv-scores", default="results/asv_scores_mfcc_trim_g256_eval.csv")
    ap.add_argument("--cm", nargs="+", default=["models/cm_gmm_lfcc_trim_g128.pkl",
                                                 "models/cm_gmm_lfcc4k_trim_g128_aug-clean-opus16-telephone-noise20.pkl"])
    ap.add_argument("--feature", nargs="+", default=["lfcc", "lfcc4k"])
    ap.add_argument("--names", nargs="+", default=["A (базовий)", "D (0–4 кГц + аугм.)"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/tandem.json")
    args = ap.parse_args()
    t0 = time.perf_counter()

    trials: dict = defaultdict(list)                       # канал → [(utt, key, asv_score)]
    with open(ROOT / args.asv_scores, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            trials[r["channel"]].append((r["utt"], r["key"], float(r["score"])))

    report: dict = {}
    rows = []
    for chan, tr in trials.items():
        key = np.array([k for _, k, _ in tr])
        asv = np.array([s for _, _, s in tr])
        thr_asv = metrics.eer(asv[key == "target"], asv[key == "nontarget"])[1]
        acc_asv = asv >= thr_asv

        def rates(accept: np.ndarray) -> dict:
            return {"FRR": float(1 - accept[key == "target"].mean()), "FAR": float(accept[key == "nontarget"].mean()),
                    "SPF-AR": float(accept[key == "spoof"].mean())}

        report[chan] = {"лише верифікація": rates(acc_asv)}
        rows.append([chan, "лише верифікація"] + [f"{v * 100:.1f}" for v in report[chan]["лише верифікація"].values()])
        utts = sorted({u for u, _, _ in tr})
        for cm_path, feature, name in zip(args.cm, args.feature, args.names):
            if feature == "ssl":
                with open(ROOT / cm_path, "rb") as fh:
                    regime = pickle.load(fh)["regime"]
                thr_cm = cm_threshold(str(ROOT / "results" / "cm_ssl_scores.csv"), regime)
            else:
                tag = cm_path.split("cm_gmm_")[1].removesuffix(".pkl")
                thr_cm = cm_threshold(str(ROOT / "results" / f"cm_scores_{tag}.csv"))
            with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(ROOT / cm_path),)) as pool:
                tasks = [(str(data.audio_path(args.data, "eval", u)), feature, chan) for u in utts]
                cm = dict(zip(utts, pool.map(cm_task, tasks, chunksize=16)))
            with open(ROOT / "results" / f"tandem_cm_scores_{feature}.csv", "a", encoding="utf-8") as fh:
                fh.writelines(f"{chan};{u};{cm[u]:.6f}\n" for u in utts)
            acc = acc_asv & np.array([cm[u] >= thr_cm for u, _, _ in tr])
            report[chan][name] = rates(acc)
            rows.append([chan, f"+ детектор {name}"] + [f"{v * 100:.1f}" for v in report[chan][name].values()])
            print(f"{chan}: {name} готово ({time.perf_counter() - t0:.0f} с)", flush=True)
    print()
    print_table(rows, ["канал", "система", "FRR, %", "FAR, %", "підробок прийнято, %"])
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Звіт:", args.out)


if __name__ == "__main__":
    main()
