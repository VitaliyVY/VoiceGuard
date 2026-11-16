"""Перевірка детекторів підробки на україномовних записах і адаптація до української.

Дані (scripts/uk_real.py, scripts/uk_fake.py):
  test:  справжні — Common Voice 19 uk (400 записів, 311 мовців);
         підробки — Piper (200) і MMS-TTS (200), українські речення;
  adapt: справжні — Common Voice uk інших мовців (train); підробки — лише Piper.
Канали: clean, opus16 (голосові Telegram/Viber), amrnb (мобільний дзвінок).

1. Детектори, навчені лише на англомовному ASVspoof 2019 LA (WavLM і LFCC-GMM):
   EER і частка справжніх українських записів, хибно відхилених за порогами програми.
2. Адаптація: логістична регресія на ознаках WavLM, навчена на ASVspoof train +
   невеликій українській частині adapt (мовці інші, синтезатор лише Piper).
   Перевірка — на test (нові мовці, MMS невідомий) і на ASVspoof eval (чи не погіршилось).

Приклад:
    python scripts/uk_eval.py
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from _common import DATA, ROOT, print_table
from cm_gmm import subset
from voiceguard import audio, channel, data, features, metrics, ssl

UK = ROOT / "data" / "uk"
CACHE = ROOT / "data" / "cache"
CHANNELS = ["clean", "opus16", "amrnb"]
PARTS = {"real_test": "real/test", "real_adapt": "real/train", "fake_test_piper": "fake/test_piper",
         "fake_test_mms": "fake/test_mms", "fake_adapt_piper": "fake/adapt_piper"}
GMM_TAG = "lfcc4k_trim_g128_aug-clean-opus16-telephone-noise20"


def ssl_task(task):
    path, chan = task
    x = audio.trim_silence(channel.CHANNELS[chan](audio.normalize(audio.load(path))))
    return ssl.layer_stats(x)


def gmm_task(task):
    path, chan, model_path = task
    with open(model_path, "rb") as fh:
        gmm = pickle.load(fh)
    x = audio.trim_silence(channel.CHANNELS[chan](audio.normalize(audio.load(path))))
    return gmm.score(features.lfcc(x, fmax=4000.0))


def files(part: str) -> list[str]:
    return sorted(str(p) for p in (UK / PARTS[part]).glob("*.wav"))


def ssl_feats(pool, part: str, chan: str) -> np.ndarray:
    path = CACHE / f"uk_{part}_{chan}.npy"
    if not path.exists():
        X = np.stack(list(pool.map(ssl_task, [(f, chan) for f in files(part)], chunksize=4)))
        np.save(path, X)
        print(f"  ознаки {part} {chan}: {X.shape}", flush=True)
    return np.load(path)


def main() -> None:
    from sklearn.linear_model import LogisticRegression

    with open(ROOT / "models" / "cm_ssl.pkl", "rb") as fh:
        base = pickle.load(fh)
    layers = [l - 1 for l in base["layers"]]
    view = lambda X: X[:, layers].astype(np.float32).mean(axis=1)  # noqa: E731
    from voiceguard.system import _dev_threshold

    thr, _ = _dev_threshold(ROOT / "results" / "cm_ssl_scores.csv", base["regime"])
    report: dict = {}

    with ProcessPoolExecutor(2) as pool:
        F = {(p, c): view(ssl_feats(pool, p, c)) for p in PARTS for c in CHANNELS}
        gmm_path = str(ROOT / "models" / f"cm_gmm_{GMM_TAG}.pkl")
        G = {(p, c): np.array(list(pool.map(gmm_task, [(f, c, gmm_path) for f in files(p)], chunksize=4)))
             for p in ("real_test", "fake_test_piper", "fake_test_mms") for c in CHANNELS}
    gthr, _ = _dev_threshold(ROOT / "results" / f"cm_scores_{GMM_TAG}.csv")

    def evaluate(name: str, score_fn, thresholds: dict | None):
        rows = []
        for c in CHANNELS:
            r = score_fn("real_test", c)
            fp, fm = score_fn("fake_test_piper", c), score_fn("fake_test_mms", c)
            res = {"eer_piper": metrics.eer(r, fp)[0], "eer_mms": metrics.eer(r, fm)[0],
                   "eer_all": metrics.eer(r, np.concatenate([fp, fm]))[0]}
            if thresholds:
                t = thresholds["звичайна"]
                res["real_rejected"] = float(np.mean(r < t))
                res["fake_accepted"] = float(np.mean(np.concatenate([fp, fm]) >= t))
            report.setdefault(name, {})[c] = res
            rows.append([c, f"{100 * res['eer_piper']:.1f}", f"{100 * res['eer_mms']:.1f}", f"{100 * res['eer_all']:.1f}",
                         f"{100 * res.get('real_rejected', float('nan')):.1f}",
                         f"{100 * res.get('fake_accepted', float('nan')):.1f}"])
        print(f"\n{name}")
        print_table(rows, ["канал", "EER Piper", "EER MMS*", "EER усі", "справжніх відхилено (звич.)",
                           "підробок пропущено (звич.)"])

    evaluate("WavLM, навчено лише на ASVspoof (англ.)",
             lambda p, c: base["model"].decision_function(F[(p, c)]), thr)
    evaluate("LFCC-GMM (D), навчено лише на ASVspoof (англ.)", lambda p, c: G[(p, c)], gthr)

    # ---------------------------------------------------------------- адаптація
    y = {s: np.array([it.bonafide for it in subset(data.cm_protocol(DATA, s), per, 0)])
         for s, per in (("train", 400), ("dev", 100), ("eval", 200))}
    TR = ["clean", "opus16", "telephone", "noise20"]
    Xen = np.concatenate([np.load(CACHE / f"ssl_train_{c}.npy", mmap_mode="r")[:, layers].astype(np.float32).mean(1)
                          for c in TR])
    yen = np.tile(y["train"], len(TR))
    Xua = np.concatenate([F[("real_adapt", c)] for c in ("clean", "opus16")] +
                         [F[("fake_adapt_piper", c)] for c in ("clean", "opus16")])
    yua = np.concatenate([np.ones(2 * len(F[("real_adapt", "clean")]), bool),
                          np.zeros(2 * len(F[("fake_adapt_piper", "clean")]), bool)])
    for ua_weight in (5.0,):
        sw = np.concatenate([np.ones(len(yen)), np.full(len(yua), ua_weight)])
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(), LogisticRegression(C=base["model"][-1].C, max_iter=3000,
                                                                 class_weight="balanced"))
        clf.fit(np.concatenate([Xen, Xua]), np.concatenate([yen, yua]),
                logisticregression__sample_weight=sw)
        # пороги — за тим самим правилом на англомовному dev (усі 10 каналів)
        dev_s, dev_b = [], []
        for c in ["clean", "opus32", "opus16", "opus8", "telephone", "amrnb", "amrwb", "gsm", "noise20", "noise10"]:
            dev_s.append(clf.decision_function(np.load(CACHE / f"ssl_dev_{c}.npy", mmap_mode="r")[:, layers]
                                               .astype(np.float32).mean(1)))
            dev_b.append(y["dev"])
        dev_s, dev_b = np.concatenate(dev_s), np.concatenate(dev_b)
        athr = {"звичайна": float(np.quantile(dev_s[dev_b], 0.10))}
        evaluate(f"WavLM, адаптовано (+ українські записи, лише Piper; вага {ua_weight})",
                 lambda p, c: clf.decision_function(F[(p, c)]), athr)
        en = {}
        for c in ["clean", "opus32", "opus16", "opus8", "telephone", "amrnb", "amrwb", "gsm", "noise20", "noise10"]:
            s = clf.decision_function(np.load(CACHE / f"ssl_eval_{c}.npy", mmap_mode="r")[:, layers].astype(np.float32).mean(1))
            en[c] = metrics.eer(s[y["eval"]], s[~y["eval"]])[0]
        report["adapted_asvspoof_eval"] = en
        print(f"   ASVspoof eval після адаптації: середній EER {100 * np.mean(list(en.values())):.2f} % "
              f"(до адаптації 7.45 %)")
        with open(ROOT / "models" / "cm_ssl_uk.pkl", "wb") as fh:
            pickle.dump({"model": clf, "layers": base["layers"], "regime": "aug+uk"}, fh)
    (ROOT / "results" / "uk_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n* MMS під час адаптації не використовувався. Звіт: results/uk_eval.json")


if __name__ == "__main__":
    main()
