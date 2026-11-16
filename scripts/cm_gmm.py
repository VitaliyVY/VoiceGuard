"""Детектор підробки голосу LFCC/MFCC-GMM і його стійкість до каналу зв'язку.

Протокол ASVspoof 2019 LA:
  train — навчання двох GMM (справжня мова / підробка);
  dev   — контроль (ті самі методи атак, що й у train, інші мовці);
  eval  — підсумок: інші мовці й частково НОВІ методи атак (A07–A19).
Кожен запис dev/eval пропускається через канал (див. voiceguard.channel)
і лише потім оцінюється: модель навчена на чистих записах, тож це перевірка
того, що станеться з детектором, коли голос надійде через месенджер чи телефон.

--trim відкидає тишу на краях: у ASVspoof 2019 LA її тривалість відрізняється
у справжніх і синтезованих записах (Müller et al., 2021), і без цього детектор
частково впізнає тишу, а не голос.

Приклад:
    python scripts/cm_gmm.py --feature lfcc --trim --eval-per-attack 300 --channels clean opus16 telephone
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from _common import DATA, ROOT, print_table
from voiceguard import audio, channel, data, features, metrics
from voiceguard.models import GMMCountermeasure

FEATURES = {"lfcc": features.lfcc, "mfcc": features.mfcc,
            "lfcc4k": lambda x: features.lfcc(x, fmax=4000.0)}
_MODEL: GMMCountermeasure | None = None


def extract(path: str, feature: str, trim: bool, chan: str = "clean") -> np.ndarray:
    x = audio.load(path)
    x = channel.CHANNELS[chan](x)
    if trim:
        x = audio.trim_silence(x)
    return FEATURES[feature](x)


def train_frames(task: tuple[str, str, bool, int, int, tuple[str, ...]]) -> np.ndarray:
    path, feature, trim, per_utt, seed, train_channels = task
    rng = np.random.default_rng(seed)
    # навчання з аугментацією: кожен запис — через випадковий канал зі списку
    f = extract(path, feature, trim, str(rng.choice(train_channels)))
    return f if len(f) <= per_utt else f[rng.choice(len(f), per_utt, replace=False)]


def _init(model_path: str) -> None:
    global _MODEL
    with open(model_path, "rb") as fh:
        _MODEL = pickle.load(fh)


def score_task(task: tuple[str, str, bool, str]) -> float:
    path, feature, trim, chan = task
    return _MODEL.score(extract(path, feature, trim, chan))


def subset(items: list[data.CMItem], per_attack: int | None, seed: int) -> list[data.CMItem]:
    """Стратифікована вибірка: до per_attack записів кожного методу атаки і справжніх."""
    if not per_attack:
        return items
    rng = np.random.default_rng(seed)
    groups: dict[str, list] = defaultdict(list)
    for it in items:
        groups[it.attack].append(it)
    out = []
    for key in sorted(groups):
        g = groups[key]
        n = per_attack * (3 if key == "-" else 1)          # справжніх більше: вони спільні для всіх атак
        out += [g[i] for i in sorted(rng.choice(len(g), min(n, len(g)), replace=False))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--feature", choices=list(FEATURES), default="lfcc")
    ap.add_argument("--trim", action="store_true")
    ap.add_argument("--components", type=int, default=128)
    ap.add_argument("--frames-per-utt", type=int, default=60)
    ap.add_argument("--train-frames", type=int, default=400_000)
    ap.add_argument("--dev-per-attack", type=int, default=300)
    ap.add_argument("--eval-per-attack", type=int, default=300)
    ap.add_argument("--channels", nargs="+", default=["clean"], choices=list(channel.CHANNELS))
    ap.add_argument("--train-channels", nargs="+", default=["clean"], choices=list(channel.CHANNELS),
                    help="канали для аугментації навчальних записів (за замовчуванням — лише чисті)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    aug = "" if args.train_channels == ["clean"] else "_aug-" + "-".join(args.train_channels)
    tag = f"{args.feature}{'_trim' if args.trim else ''}_g{args.components}{aug}"
    model_path = ROOT / "models" / f"cm_gmm_{tag}.pkl"
    out_path = ROOT / "results" / f"cm_gmm_{tag}.json"
    model_path.parent.mkdir(exist_ok=True)
    out_path.parent.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    with ProcessPoolExecutor(args.workers) as pool:
        if args.retrain or not model_path.exists():
            items = data.cm_protocol(args.data, "train")
            tasks = [(str(data.audio_path(args.data, "train", it.utt)), args.feature, args.trim,
                      args.frames_per_utt, args.seed + i, tuple(args.train_channels)) for i, it in enumerate(items)]
            frames = list(pool.map(train_frames, tasks, chunksize=32))
            rng = np.random.default_rng(args.seed)
            sets = {}
            for name, flag in (("bonafide", True), ("spoof", False)):
                f = np.concatenate([fr for fr, it in zip(frames, items) if it.bonafide == flag])
                if len(f) > args.train_frames:
                    f = f[rng.choice(len(f), args.train_frames, replace=False)]
                sets[name] = f
            print(f"Ознаки train: справжніх кадрів {len(sets['bonafide'])}, підроблених {len(sets['spoof'])} "
                  f"({time.perf_counter() - t0:.0f} с)", flush=True)
            model = GMMCountermeasure.fit(sets["bonafide"], sets["spoof"], args.components, args.seed)
            with open(model_path, "wb") as fh:
                pickle.dump(model, fh)
            print(f"GMM навчено ({time.perf_counter() - t0:.0f} с): {model_path.name}", flush=True)

    report: dict = {"config": vars(args)}
    scores_csv = ROOT / "results" / f"cm_scores_{tag}.csv"
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(model_path),)) as pool,             open(scores_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["split", "channel", "utt", "attack", "bonafide", "score"])
        for split, per_attack in (("dev", args.dev_per_attack), ("eval", args.eval_per_attack)):
            items = subset(data.cm_protocol(args.data, split), per_attack, args.seed)
            for chan in args.channels:          # dev — для вибору варіанта, eval — лише для звіту
                tasks = [(str(data.audio_path(args.data, split, it.utt)), args.feature, args.trim, chan) for it in items]
                scores = np.array(list(pool.map(score_task, tasks, chunksize=16)))
                for it, sc in zip(items, scores):
                    writer.writerow([split, chan, it.utt, it.attack, int(it.bonafide), f"{sc:.6f}"])
                bona = np.array([it.bonafide for it in items])
                res = {"n": len(items), "eer": metrics.eer(scores[bona], scores[~bona])[0], "per_attack": {}}
                for attack in sorted({it.attack for it in items if not it.bonafide}):
                    m = np.array([it.attack == attack for it in items])
                    res["per_attack"][attack] = metrics.eer(scores[bona], scores[m])[0]
                report.setdefault(split, {})[chan] = res
                print(f"{split:4s} {chan:10s} EER = {res['eer'] * 100:.2f} %  (записів {len(items)}, "
                      f"{time.perf_counter() - t0:.0f} с)", flush=True)

    attacks = sorted(report["eval"]["clean" if "clean" in args.channels else args.channels[0]]["per_attack"])
    rows = [[chan] + [f"{report['eval'][chan]['per_attack'][a] * 100:.1f}" for a in attacks] +
            [f"{report['eval'][chan]['eer'] * 100:.2f}"] for chan in args.channels]
    print("\nEER на eval за методами атак, %:")
    print_table(rows, ["канал", *attacks, "усі"])
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Звіт:", out_path.relative_to(ROOT), "| оцінки:", scores_csv.relative_to(ROOT))


if __name__ == "__main__":
    main()
