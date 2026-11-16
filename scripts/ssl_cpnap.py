"""Канально-парна проєкція завад (CP-NAP) для детектора підробки на ознаках WavLM.

Ідея: у навчальних даних той самий запис є в кількох каналах (чистий, Opus 16,
G.711, шум). Для запису i і каналу c відхилення від його середнього за каналами
    d_ic = x_ic − (1/C) Σ_c x_ic
описує лише вплив каналу (зміст, мовець і факт підробки однакові). Головні
напрямки коваріації цих відхилень S = Σ d dᵀ — «підпростір каналу» U_k
(k власних векторів). Перед класифікатором ознаки проєктуються на його
доповнення:  x' = (I − U_k U_kᵀ) x.
Класифікатор (логістична регресія) навчається на проєктованих ознаках; k
обирається за середнім EER на dev (усі канали), eval — лише звіт. Окремо
звітується стабільність порогу: частка підробок на eval, прийнятих за порогом
EER, визначеним на dev.

Приклад:
    python scripts/ssl_cpnap.py
"""

from __future__ import annotations

import json

import numpy as np

from _common import DATA, ROOT, print_table
from cm_gmm import subset
from voiceguard import data, metrics

CACHE = ROOT / "data" / "cache"
TRAIN_CH = ["clean", "opus16", "telephone", "noise20"]
CHANNELS = ["clean", "opus32", "opus16", "opus8", "telephone", "amrnb", "amrwb", "gsm", "noise20", "noise10"]
UNSEEN = {"opus32", "opus8", "amrnb", "amrwb", "gsm", "noise10"}
LAYERS = [1, 2, 3, 4, 5, 6]                      # шари 2–7 (як у обраній моделі)


def view(split: str, ch: str) -> np.ndarray:
    X = np.load(CACHE / f"ssl_{split}_{ch}.npy", mmap_mode="r")
    return X[:, LAYERS].astype(np.float32).mean(axis=1)


def main() -> None:
    from sklearn.linear_model import LogisticRegression

    y = {s: np.array([it.bonafide for it in subset(data.cm_protocol(DATA, s), per, 0)])
         for s, per in (("train", 400), ("dev", 100), ("eval", 200))}
    tr = np.stack([view("train", c) for c in TRAIN_CH])            # (каналів, записів, ознак)
    mu, sd = tr.reshape(-1, tr.shape[-1]).mean(0), tr.reshape(-1, tr.shape[-1]).std(0) + 1e-6
    tr = (tr - mu) / sd
    dev = {c: (view("dev", c) - mu) / sd for c in CHANNELS}
    ev = {c: (view("eval", c) - mu) / sd for c in CHANNELS}

    d = (tr - tr.mean(axis=0, keepdims=True)).reshape(-1, tr.shape[-1])   # відхилення «запис у каналі − запис»
    _, svals, vt = np.linalg.svd(d, full_matrices=False)
    energy = np.cumsum(svals ** 2) / np.sum(svals ** 2)
    print("частка енергії каналу в перших k напрямках:",
          {k: round(float(energy[k - 1]), 3) for k in (1, 5, 10, 20, 50, 100)})

    Xtr, ytr = tr.reshape(-1, tr.shape[-1]), np.tile(y["train"], len(TRAIN_CH))
    report, rows = {"k": {}}, []
    for k in (0, 5, 10, 20, 50, 100, 200):
        U = vt[:k].T                                                # (ознак, k)
        proj = (lambda X: X - (X @ U) @ U.T) if k else (lambda X: X)
        clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(proj(Xtr), ytr)
        sdev = {c: clf.decision_function(proj(dev[c])) for c in CHANNELS}
        sev = {c: clf.decision_function(proj(ev[c])) for c in CHANNELS}
        dev_eer = {c: metrics.eer(sdev[c][y["dev"]], sdev[c][~y["dev"]])[0] for c in CHANNELS}
        ev_eer = {c: metrics.eer(sev[c][y["eval"]], sev[c][~y["eval"]])[0] for c in CHANNELS}
        alld = np.concatenate(list(sdev.values()))
        allb = np.concatenate([y["dev"]] * len(CHANNELS))
        thr = metrics.eer(alld[allb], alld[~allb])[1]                 # поріг рівних помилок на dev
        spf = np.mean([np.mean(sev[c][~y["eval"]] >= thr) for c in CHANNELS])
        frr = np.mean([np.mean(sev[c][y["eval"]] < thr) for c in CHANNELS])
        r = {"dev_mean": float(np.mean(list(dev_eer.values()))), "eval": ev_eer,
             "eval_mean": float(np.mean(list(ev_eer.values()))),
             "eval_unseen": float(np.mean([v for c, v in ev_eer.items() if c in UNSEEN])),
             "eval_spoof_pass_at_dev_thr": float(spf), "eval_bona_reject_at_dev_thr": float(frr)}
        report["k"][k] = r
        rows.append([k, f"{100 * r['dev_mean']:.2f}", f"{100 * r['eval_mean']:.2f}", f"{100 * r['eval_unseen']:.2f}",
                     f"{100 * spf:.1f}", f"{100 * frr:.1f}"])
        print(f"k={k}: готово", flush=True)
    print()
    print_table(rows, ["k (прибраних напрямків)", "dev сер. EER", "eval сер. EER", "eval нові канали",
                       "підробок пройшло за порогом dev", "справжніх відхилено"])
    best = min(report["k"], key=lambda k: report["k"][k]["dev_mean"])
    report["chosen_on_dev"] = best
    print(f"\nВибір за dev: k = {best}")
    b, base = report["k"][best], report["k"][0]
    print("канал       без проєкції  CP-NAP")
    for c in CHANNELS:
        print(f"  {c:10s} {100 * base['eval'][c]:8.2f}   {100 * b['eval'][c]:8.2f}")
    (ROOT / "results" / "ssl_cpnap.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
