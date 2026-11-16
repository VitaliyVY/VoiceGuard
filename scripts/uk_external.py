"""Незалежна перевірка детектора на зовнішніх українських наборах (не використовувались ні в навчанні, ні в адаптації).

Набори (data/external/, завантаження — docs/EXPERIMENTS.md, розділ «Зовнішня перевірка»):
  ml_itw_uk/   — ML-ITW (Multilingual In-The-Wild, 2026), українська частина: справжні записи
                 й дипфейки українських публічних осіб з Facebook і X (як вони поширювались у мережі);
  drssu/       — DRSSU (Scientific Data, 2025): generated/ — Edge-TTS (uk-UA-OstapNeural) + RVC
                 (перетворення голосу); real/ — Common Voice 16.1 uk (можливий перетин мовців
                 з Common Voice, на якому адаптовано детектор, тож звітується окремо);
  fleurs_uk/   — FLEURS uk test (CC BY 4.0): справжнє читане мовлення інших людей, мікрофони різні.

Пари для EER:
  A. ML-ITW: справжні vs дипфейки (усе з тих самих платформ — чесне порівняння умов);
  B. DRSSU-підробки vs FLEURS (незалежні справжні);
  C. DRSSU-підробки vs DRSSU-справжні (Common Voice 16.1).
Для кожного детектора також частка помилок на порогах програми (м'яка / звичайна / сувора).

Приклад:
    python scripts/uk_external.py
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import ROOT  # noqa: E402  (налаштування потоків)

import numpy as np  # noqa: E402

from voiceguard import audio, metrics, ssl  # noqa: E402
from voiceguard.system import _dev_threshold  # noqa: E402

EXT = ROOT / "data" / "external"
CACHE = ROOT / "data" / "cache"


def detectors() -> dict:
    out = {}
    d = pickle.load(open(ROOT / "models" / "cm_ssl.pkl", "rb"))
    thr, _ = _dev_threshold(ROOT / "results" / "cm_ssl_scores.csv", d["regime"])
    out["WavLM, лише ASVspoof (англ.)"] = (d["model"], [l - 1 for l in d["layers"]], thr)
    u = pickle.load(open(ROOT / "models" / "cm_ssl_uk.pkl", "rb"))
    out["WavLM, адаптований до української (програма)"] = (u["model"], [l - 1 for l in u["layers"]], u["thresholds"])
    return out


def stats_for(name: str, files: list[Path]) -> np.ndarray:
    cache = CACHE / f"ext_{name}.npy"
    if cache.exists():
        s = np.load(cache)
        if len(s) == len(files):
            return s.astype(np.float32)
    rows = []
    for i, f in enumerate(files):
        x = audio.trim_silence(audio.load(f))
        rows.append(ssl.layer_stats(x).astype(np.float16))
        if (i + 1) % 50 == 0:
            print(f"  {name}: {i + 1}/{len(files)}", flush=True)
    s = np.stack(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, s)
    return s.astype(np.float32)


def score(det, stats: np.ndarray) -> np.ndarray:
    clf, layers, _ = det
    return clf.decision_function(stats[:, layers].mean(axis=1))


def main() -> None:
    sets: dict[str, list[Path]] = {}
    idx = list(csv.DictReader(open(EXT / "ml_itw_uk" / "index.csv", encoding="utf-8")))
    sets["mlitw_real"] = [EXT / "ml_itw_uk" / r["file"] for r in idx if r["label"] == "bona-fide"]
    sets["mlitw_fake"] = [EXT / "ml_itw_uk" / r["file"] for r in idx if r["label"] == "spoof"]
    sets["drssu_fake"] = sorted((EXT / "drssu" / "generated").glob("*.wav"))
    sets["drssu_real"] = sorted((EXT / "drssu" / "real").glob("*.wav"))
    sets["fleurs_real"] = sorted((EXT / "fleurs_uk").glob("*.wav"))
    for k, v in sets.items():
        print(f"{k}: {len(v)} записів")
    st = {k: stats_for(k, v) for k, v in sets.items()}

    pairs = {"A. ML-ITW: справжні vs дипфейки (соцмережі)": ("mlitw_real", "mlitw_fake"),
             "B. DRSSU Edge-TTS+RVC vs FLEURS": ("fleurs_real", "drssu_fake"),
             "C. DRSSU Edge-TTS+RVC vs DRSSU справжні (CV 16.1)": ("drssu_real", "drssu_fake")}
    report: dict = {"n": {k: len(v) for k, v in sets.items()}}
    for dname, det in detectors().items():
        sc = {k: score(det, s) for k, s in st.items()}
        r: dict = {}
        for pname, (real, fake) in pairs.items():
            r[pname] = {"eer": round(float(metrics.eer(sc[real], sc[fake])[0]), 4)}
        for k in sets:
            r[k] = {lvl: round(float(np.mean(sc[k] < t if "real" in k else sc[k] >= t)), 4)
                    for lvl, t in det[2].items()}
        report[dname] = r
        print(f"\n=== {dname}")
        for pname in pairs:
            print(f"  {pname:52s} EER = {r[pname]['eer'] * 100:5.1f} %")
        print("  Помилки на порогах програми (справжні — частка відхилених, підробки — частка пропущених):")
        for k in sets:
            print(f"    {k:12s} " + "  ".join(f"{lvl}: {v * 100:5.1f} %" for lvl, v in r[k].items()))
        with open(ROOT / "results" / "uk_external_scores.csv", "a" if dname != next(iter(detectors())) else "w",
                  encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter=";")
            if fh.tell() == 0:
                w.writerow(["detector", "set", "file", "score"])
            for k, v in sets.items():
                for f, s in zip(v, sc[k]):
                    w.writerow([dname, k, f.name, f"{s:.4f}"])
    (ROOT / "results" / "uk_external.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nЗвіт: results/uk_external.json | оцінки: results/uk_external_scores.csv")


if __name__ == "__main__":
    main()
