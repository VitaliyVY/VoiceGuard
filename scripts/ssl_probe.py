"""Швидка проба: чи кращі ознаки WavLM за LFCC (мала вибірка, чисті записи).

Приклад:
    python scripts/ssl_probe.py --train-per-attack 50 --dev-per-attack 50
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import DATA, ROOT
from cm_gmm import subset
from voiceguard import audio, channel, data, metrics, ssl


def feat_task(task) -> np.ndarray:
    path, chan = task
    x = audio.trim_silence(channel.CHANNELS[chan](audio.load(path)))
    return ssl.layer_stats(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-per-attack", type=int, default=50)
    ap.add_argument("--dev-per-attack", type=int, default=50)
    ap.add_argument("--channels", nargs="+", default=["clean", "amrnb"])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    t0 = time.perf_counter()

    sets = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for split, per in (("train", args.train_per_attack), ("dev", args.dev_per_attack)):
            items = subset(data.cm_protocol(DATA, split), per, 0)
            for chan in (["clean"] if split == "train" else args.channels):
                tasks = [(str(data.audio_path(DATA, split, it.utt)), chan) for it in items]
                X = np.stack(list(pool.map(feat_task, tasks, chunksize=4))).astype(np.float32)
                sets[(split, chan)] = (X, np.array([it.bonafide for it in items]))
                print(f"{split} {chan}: {X.shape} ({time.perf_counter() - t0:.0f} с)", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = sets[("train", "clean")]
    print("\nEER на dev (логістична регресія на ознаках одного шару), %:")
    print("шар  " + "  ".join(f"{c:>7s}" for c in args.channels))
    for layer in range(Xtr.shape[1]):
        sc = StandardScaler().fit(Xtr[:, layer])
        clf = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr[:, layer]), ytr)
        row = []
        for chan in args.channels:
            Xd, yd = sets[("dev", chan)]
            s = clf.decision_function(sc.transform(Xd[:, layer]))
            row.append(metrics.eer(s[yd], s[~yd])[0] * 100)
        print(f"{layer + 1:3d}  " + "  ".join(f"{v:7.2f}" for v in row), flush=True)


if __name__ == "__main__":
    main()
