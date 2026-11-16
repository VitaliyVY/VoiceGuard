"""Додаткові експерименти до адаптації детектора на українському мовленні.

1. Крива навчання: скільки українських пар «справжній + синтез» потрібно (0, 10, 30, 60, 118;
   для неповних вибірок — 5 випадкових підвибірок, середнє й розкид).
2. Довірчі інтервали (бутстреп, 2000 повторів, 95 %) для головних EER: англомовна й
   адаптована моделі, канали clean / opus16 / amrnb, а також підробки з умовами побутового
   запису (scripts/uk_honesty.py). Справжні й синтезовані записи перевибираються окремо.
3. Перехресна перевірка синтезаторів: адаптація лише на Piper → перевірка на MMS (основний
   варіант) і адаптація лише на MMS → перевірка на Piper.

Рецепт адаптації той самий, що в uk_eval.py: логістична регресія на ознаках WavLM,
ASVspoof train (4 канали) + українська частина adapt (clean, opus16) з вагою 5.

Приклад:
    python scripts/uk_extra.py
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import DATA, ROOT, print_table
from cm_gmm import subset
from uk_eval import CACHE, CHANNELS, files, ssl_task
from voiceguard import data, metrics

UA_WEIGHT = 5.0
SIZES = [10, 30, 60, 118]
REPEATS = 5


def boot_eer(r: np.ndarray, f: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = [metrics.eer(r[rng.integers(0, len(r), len(r))], f[rng.integers(0, len(f), len(f))])[0] for _ in range(n)]
    return metrics.eer(r, f)[0], float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    with open(ROOT / "models" / "cm_ssl.pkl", "rb") as fh:
        base = pickle.load(fh)
    with open(ROOT / "models" / "cm_ssl_uk.pkl", "rb") as fh:
        adapted = pickle.load(fh)
    layers = [l - 1 for l in base["layers"]]
    view = lambda X: X[:, layers].astype(np.float32).mean(axis=1)  # noqa: E731

    # ознаки adapt_mms (решта — у кеші після uk_eval.py / uk_honesty.py)
    with ProcessPoolExecutor(2) as pool:
        for c in ("clean", "opus16"):
            path = CACHE / f"uk_fake_adapt_mms_{c}.npy"
            if not path.exists():
                fs = sorted(str(p) for p in (ROOT / "data" / "uk" / "fake" / "adapt_mms").glob("*.wav"))
                np.save(path, np.stack(list(pool.map(ssl_task, [(f, c) for f in fs], chunksize=4))))
                print(f"  ознаки adapt_mms {c}", flush=True)
    F = {}
    for part in ("real_test", "real_adapt", "fake_test_piper", "fake_test_mms", "fake_adapt_piper", "fake_adapt_mms"):
        for c in CHANNELS:
            p = CACHE / f"uk_{part}_{c}.npy"
            if p.exists():
                F[(part, c)] = view(np.load(p))
    for part in ("fake_test_piper", "fake_test_mms"):
        for c in ("mp3", "mp3_room20", "mp3_room10"):
            F[(part, c)] = view(np.load(CACHE / f"uk_{part}_{c}.npy"))

    y_tr = np.array([it.bonafide for it in subset(data.cm_protocol(DATA, "train"), 400, 0)])
    TR = ["clean", "opus16", "telephone", "noise20"]
    Xen = np.concatenate([view(np.load(CACHE / f"ssl_train_{c}.npy", mmap_mode="r")) for c in TR])
    yen = np.tile(y_tr, len(TR))
    C = base["model"][-1].C

    def adapt(real_idx: np.ndarray, fake_part: str, fake_idx: np.ndarray):
        Xua = np.concatenate([F[("real_adapt", c)][real_idx] for c in ("clean", "opus16")] +
                             [F[(fake_part, c)][fake_idx] for c in ("clean", "opus16")])
        yua = np.concatenate([np.ones(2 * len(real_idx), bool), np.zeros(2 * len(fake_idx), bool)])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(np.concatenate([Xen, Xua]), np.concatenate([yen, yua]),
                logisticregression__sample_weight=np.concatenate([np.ones(len(yen)), np.full(len(yua), UA_WEIGHT)]))
        return clf.decision_function

    def eers(score) -> dict:
        out = {}
        for c in CHANNELS:
            r = score(F[("real_test", c)])
            fp, fm = score(F[("fake_test_piper", c)]), score(F[("fake_test_mms", c)])
            out[c] = {"piper": metrics.eer(r, fp)[0], "mms": metrics.eer(r, fm)[0],
                      "all": metrics.eer(r, np.concatenate([fp, fm]))[0]}
        return out

    report: dict = {}
    n_real, n_piper = len(F[("real_adapt", "clean")]), len(F[("fake_adapt_piper", "clean")])

    # ------------------------------------------------------------------ 1. крива навчання
    print("\n1. Скільки українських записів потрібно (адаптація на Piper; EER усі синтезатори, %)")
    curve, rows = {}, []
    base_e = eers(base["model"].decision_function)
    curve[0] = {c: {"mean": base_e[c]["all"], "std": 0.0, "mms_mean": base_e[c]["mms"]} for c in CHANNELS}
    rows.append([0] + [f"{100 * base_e[c]['all']:.1f}" for c in CHANNELS] + [f"{100 * base_e['clean']['mms']:.1f}"])
    for n in SIZES:
        runs = []
        for rep in range(REPEATS if n < min(n_real, n_piper) else 1):
            rng = np.random.default_rng(rep)
            ri = rng.choice(n_real, n, replace=False) if n < n_real else np.arange(n_real)
            fi = rng.choice(n_piper, n, replace=False) if n < n_piper else np.arange(n_piper)
            runs.append(eers(adapt(ri, "fake_adapt_piper", fi)))
        curve[n] = {c: {"mean": float(np.mean([r[c]["all"] for r in runs])), "std": float(np.std([r[c]["all"] for r in runs])),
                        "mms_mean": float(np.mean([r[c]["mms"] for r in runs]))} for c in CHANNELS}
        rows.append([n] + [f"{100 * curve[n][c]['mean']:.1f} ± {100 * curve[n][c]['std']:.1f}" for c in CHANNELS] +
                    [f"{100 * curve[n]['clean']['mms_mean']:.1f}"])
        print(f"  n = {n} готово", flush=True)
    print_table(rows, ["пар записів", *CHANNELS, "MMS*, clean"])
    report["learning_curve"] = curve

    # ------------------------------------------------------------------ 2. довірчі інтервали
    print("\n2. EER з 95 % довірчими інтервалами (бутстреп), %")
    ci, rows = {}, []
    for name, score in (("WavLM (англ.)", base["model"].decision_function),
                        ("WavLM адаптований", adapted["model"].decision_function)):
        r_clean = score(F[("real_test", "clean")])
        for c in CHANNELS + ["mp3_room20"]:
            r = r_clean if c == "mp3_room20" else score(F[("real_test", c)])
            f = np.concatenate([score(F[("fake_test_piper", c)]), score(F[("fake_test_mms", c)])])
            e, lo, hi = boot_eer(r, f)
            ci.setdefault(name, {})[c] = {"eer": e, "lo": lo, "hi": hi}
            rows.append([name, c, f"{100 * e:.1f}", f"[{100 * lo:.1f}; {100 * hi:.1f}]"])
    print_table(rows, ["модель", "умови", "EER", "95 % ДІ"])
    report["bootstrap_ci"] = ci

    # ------------------------------------------------------------------ 3. обмін синтезаторами
    print("\n3. Перехресна перевірка синтезаторів (EER, %; * — синтезатор не бачили в адаптації)")
    rows = []
    n_mms = len(F[("fake_adapt_mms", "clean")])
    n = min(n_real, n_piper, n_mms)
    cross = {}
    for name, part, total in (("адаптація на Piper", "fake_adapt_piper", n_piper), ("адаптація на MMS", "fake_adapt_mms", n_mms)):
        rng = np.random.default_rng(0)
        fi = rng.choice(total, n, replace=False) if n < total else np.arange(total)
        ri = rng.choice(n_real, n, replace=False) if n < n_real else np.arange(n_real)
        e = eers(adapt(ri, part, fi))
        cross[name] = e
        mark = ("", "*") if part == "fake_adapt_piper" else ("*", "")
        for c in CHANNELS:
            rows.append([name, c, f"{100 * e[c]['piper']:.1f}{mark[0]}", f"{100 * e[c]['mms']:.1f}{mark[1]}"])
    print_table(rows, ["варіант", "канал", "EER Piper", "EER MMS"])
    report["cross_synth"] = cross
    report["n_pairs_cross"] = n

    (ROOT / "results" / "uk_extra.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nЗвіт: results/uk_extra.json")


if __name__ == "__main__":
    main()
