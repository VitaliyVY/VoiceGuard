"""Детектор підробки на ознаках WavLM Base+ + логістична регресія.

Етапи:
1. Ознаки (кеш у data/cache/ssl_*.npy, float16): train — під кожним каналом
   із --train-channels; dev і eval — під кожним каналом із --channels.
2. Класифікатор: стандартизація + логістична регресія з L2-регуляризацією на
   ознаках одного шару або середньому кількох шарів; варіант (шар, навчання
   на чистих чи з аугментацією, C) обирається за середнім EER на dev.
3. Звіт на eval (ті самі записи, що й для GMM/LCNN: subset seed 0), оцінки
   кожного запису — у CSV, модель — у models/cm_ssl.pkl.

Приклад:
    python scripts/ssl_cm.py --stage all
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import DATA, ROOT, print_table
from cm_gmm import subset
from voiceguard import audio, channel, data, metrics, ssl

CACHE = ROOT / "data" / "cache"
UNSEEN = {"opus32", "opus8", "amrnb", "amrwb", "gsm", "noise10"}


def feat_task(task) -> np.ndarray:
    path, chan = task
    x = audio.trim_silence(channel.CHANNELS[chan](audio.load(path)))
    return ssl.layer_stats(x)


def items_of(split: str, args) -> list:
    per = {"train": args.train_per_attack, "dev": args.dev_per_attack, "eval": args.eval_per_attack}[split]
    return subset(data.cm_protocol(DATA, split), per, 0)


def extract(args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    plan = [("train", c) for c in args.train_channels] + [(s, c) for s in ("dev", "eval") for c in args.channels]
    with ProcessPoolExecutor(args.workers) as pool:
        for split, chan in plan:
            path = CACHE / f"ssl_{split}_{chan}.npy"
            if path.exists():
                continue
            items = items_of(split, args)
            tasks = [(str(data.audio_path(DATA, split, it.utt)), chan) for it in items]
            X = np.stack(list(pool.map(feat_task, tasks, chunksize=4)))
            np.save(path, X)
            print(f"ознаки {split:5s} {chan:10s} {X.shape}  ({time.perf_counter() - t0:.0f} с)", flush=True)


def layer_view(X: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    return X[:, list(layers)].astype(np.float32).mean(axis=1)


def train_eval(args) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = {s: np.array([it.bonafide for it in items_of(s, args)]) for s in ("train", "dev", "eval")}
    train = {c: np.load(CACHE / f"ssl_train_{c}.npy", mmap_mode="r") for c in args.train_channels}
    dev = {c: np.load(CACHE / f"ssl_dev_{c}.npy", mmap_mode="r") for c in args.channels}

    candidates = [(l,) for l in range(12)] + [(2, 3, 4, 5), (1, 2, 3, 4, 5, 6)]
    results = []
    for regime in ("clean", "aug"):
        chans = ["clean"] if regime == "clean" else args.train_channels
        for layers in candidates:
            Xtr = np.concatenate([layer_view(train[c], layers) for c in chans])
            ytr = np.concatenate([y["train"]] * len(chans))
            for C in (0.01, 0.1):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
                clf.fit(Xtr, ytr)
                dev_eer = {c: metrics.eer(*(lambda s: (s[y["dev"]], s[~y["dev"]]))(
                    clf.decision_function(layer_view(dev[c], layers))))[0] for c in args.channels}
                mean = float(np.mean(list(dev_eer.values())))
                results.append({"regime": regime, "layers": [l + 1 for l in layers], "C": C, "dev_mean": mean,
                                "dev": dev_eer, "model": clf})
                print(f"{regime:5s} шари {str([l + 1 for l in layers]):20s} C={C:<5} dev середній EER {mean * 100:.2f} %",
                      flush=True)

    best = {}
    for regime in ("clean", "aug"):
        best[regime] = min((r for r in results if r["regime"] == regime), key=lambda r: r["dev_mean"])
    chosen = min(best.values(), key=lambda r: r["dev_mean"])
    print(f"\nВибір за dev: {chosen['regime']}, шари {chosen['layers']}, C={chosen['C']} "
          f"(dev {chosen['dev_mean'] * 100:.2f} %)")

    report: dict = {"candidates": [{k: v for k, v in r.items() if k != "model"} for r in results], "eval": {}}
    rows = []
    with open(ROOT / "results" / "cm_ssl_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["variant", "split", "channel", "utt", "attack", "bonafide", "score"])
        items = {s: items_of(s, args) for s in ("dev", "eval")}
        for regime, r in best.items():
            layers = tuple(l - 1 for l in r["layers"])
            ev = {}
            for c in args.channels:
                for split in ("dev", "eval"):
                    X = np.load(CACHE / f"ssl_{split}_{c}.npy", mmap_mode="r")
                    s = r["model"].decision_function(layer_view(X, layers))
                    for it, sc in zip(items[split], s):
                        w.writerow([regime, split, c, it.utt, it.attack, int(it.bonafide), f"{sc:.6f}"])
                    if split == "eval":
                        ev[c] = metrics.eer(s[y["eval"]], s[~y["eval"]])[0]
            report["eval"][regime] = {"layers": r["layers"], "C": r["C"], "dev_mean": r["dev_mean"], "eer": ev}
            rows.append([f"WavLM, {'чисті' if regime == 'clean' else 'аугм.'}"] + [f"{ev[c] * 100:.1f}" for c in args.channels]
                        + [f"{np.mean(list(ev.values())) * 100:.1f}",
                           f"{np.mean([v for c, v in ev.items() if c in UNSEEN]) * 100:.1f}"])
    report["chosen"] = chosen["regime"]
    print("\nEER на eval, %:")
    print_table(rows, ["варіант", *args.channels, "середнє", "нові канали"])
    with open(ROOT / "models" / "cm_ssl.pkl", "wb") as fh:
        pickle.dump({"model": chosen["model"], "layers": chosen["layers"], "regime": chosen["regime"],
                     "threshold_dev": None}, fh)
    (ROOT / "results" / "cm_ssl.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Звіт: results/cm_ssl.json | модель: models/cm_ssl.pkl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "train", "all"], default="all")
    ap.add_argument("--train-channels", nargs="+", default=["clean", "opus16", "telephone", "noise20"])
    ap.add_argument("--channels", nargs="+", default=["clean", "opus32", "opus16", "opus8", "telephone",
                                                      "amrnb", "amrwb", "gsm", "noise20", "noise10"])
    ap.add_argument("--train-per-attack", type=int, default=400)
    ap.add_argument("--dev-per-attack", type=int, default=100)
    ap.add_argument("--eval-per-attack", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.stage in ("extract", "all"):
        extract(args)
    if args.stage in ("train", "all"):
        train_eval(args)


if __name__ == "__main__":
    main()
