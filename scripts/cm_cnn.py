"""Детектор підробки LightCNN на LFCC з аугментацією каналами (порівняння з GMM).

1. Кеш: навчальні записи (усі справжні + рівна кількість підроблених,
   порівну за методами атак) один раз пропускаються через кожен канал із
   --train-channels; ознаки обрізаються до --frames кадрів і зберігаються.
2. Навчання: у кожній епосі для кожного запису випадково обирається один
   із його канальних варіантів і випадкове вікно (аугментація).
3. Оцінка: dev (вибір епохи) і eval (звіт) за всіма --channels, ознаки
   рахуються на повному записі.

Приклад:
    python scripts/cm_cnn.py --feature lfcc --train-channels clean opus16 telephone noise20 \
        --channels clean opus32 opus16 opus8 telephone amrnb amrwb gsm noise20 noise10
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import DATA, ROOT, print_table
from cm_gmm import FEATURES, extract, subset
from voiceguard import channel, data, metrics


def cache_task(task) -> np.ndarray:
    path, feature, chan, frames, seed = task
    f = extract(path, feature, True, chan)
    if len(f) < frames:                                    # короткі записи — циклічне доповнення
        f = np.resize(f, (frames, f.shape[1]))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(f) - frames + 1))
    return f[start:start + frames].astype(np.float16)


def feat_task(task) -> np.ndarray:
    path, feature, chan, max_frames = task
    f = extract(path, feature, True, chan)
    if len(f) > max_frames:                               # центральне вікно — обмеження пам'яті
        start = (len(f) - max_frames) // 2
        f = f[start:start + max_frames]
    return f.astype(np.float16)


def build_cache(args, pool, x_path, y_path) -> None:
    """Навчальний кеш (записів, каналів, кадрів, ознак) у float16 — у файлі, не в ОЗП."""
    items = data.cm_protocol(args.data, "train")
    rng = np.random.default_rng(args.seed)
    bona = [it for it in items if it.bonafide]
    spoof_by: dict[str, list] = {}
    for it in items:
        if not it.bonafide:
            spoof_by.setdefault(it.attack, []).append(it)
    per = len(bona) // len(spoof_by)
    spoof = [g[i] for g in spoof_by.values() for i in rng.choice(len(g), per, replace=False)]
    chosen = bona + spoof
    if args.limit:                                         # для швидкої перевірки скрипту
        chosen = [chosen[i] for i in sorted(rng.choice(len(chosen), args.limit, replace=False))]
    np.save(y_path, np.array([it.bonafide for it in chosen], dtype=np.float32))
    n_feat = len(extract(str(data.audio_path(args.data, "train", chosen[0].utt)), args.feature, True)[0])
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float16,
                                  shape=(len(chosen), len(args.train_channels), args.frames, n_feat))
    for c, chan in enumerate(args.train_channels):
        tasks = [(str(data.audio_path(args.data, "train", it.utt)), args.feature, chan, args.frames,
                  args.seed + 1000 * c + i) for i, it in enumerate(chosen)]
        for i, f in enumerate(pool.map(cache_task, tasks, chunksize=16)):
            X[i, c] = f
        X.flush()
        print(f"  кеш {chan}: {len(chosen)} записів", flush=True)
    del X


def score_all(model, feats: list[np.ndarray], mean, std) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for f in feats:
            x = torch.from_numpy(((f.astype(np.float32) - mean) / std))[None]
            out.append(float(model(x)[0]))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--feature", choices=list(FEATURES), default="lfcc")
    ap.add_argument("--train-channels", nargs="+", default=["clean"], choices=list(channel.CHANNELS))
    ap.add_argument("--channels", nargs="+", default=["clean"], choices=list(channel.CHANNELS))
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--crop", type=int, default=200, help="довжина вікна під час навчання, кадрів")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-frames", type=int, default=600, help="найдовше вікно оцінки, кадрів")
    ap.add_argument("--dev-per-attack", type=int, default=100)
    ap.add_argument("--eval-per-attack", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="лише N навчальних записів (перевірка скрипту)")
    args = ap.parse_args()

    import torch
    from voiceguard.cnn import LightCNN

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    aug = "" if args.train_channels == ["clean"] else "_aug-" + "-".join(args.train_channels)
    tag = f"{args.feature}_trim{aug}" + (f"_limit{args.limit}" if args.limit else "")
    out_tag = tag + (f"_seed{args.seed}" if args.seed else "")      # кеш спільний, результати — окремо для кожного запуску
    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path = cache_dir / f"cnn_X_{tag}_{args.frames}.npy", cache_dir / f"cnn_y_{tag}_{args.frames}.npy"
    t0 = time.perf_counter()

    def features_of(pool, split: str, per: int, chan: str):
        items = subset(data.cm_protocol(args.data, split), per, 0)      # ті самі записи для всіх запусків
        tasks = [(str(data.audio_path(args.data, split, it.utt)), args.feature, chan, args.max_frames) for it in items]
        return items, list(pool.map(feat_task, tasks, chunksize=16))

    with ProcessPoolExecutor(args.workers) as pool:
        if not x_path.exists():
            build_cache(args, pool, x_path, y_path)
        # dev — для вибору епохи, тримається в пам'яті (невеликий)
        devsets = {chan: features_of(pool, "dev", args.dev_per_attack, chan) for chan in args.channels}
    X, y = np.load(x_path, mmap_mode="r"), np.load(y_path)
    print(f"Навчальний кеш: {X.shape}, dev: {len(devsets[args.channels[0]][0])} записів × "
          f"{len(args.channels)} каналів ({time.perf_counter() - t0:.0f} с)", flush=True)

    sample = np.asarray(X[np.random.default_rng(1).choice(len(X), min(500, len(X)), replace=False)], dtype=np.float32)
    mean = sample.reshape(-1, X.shape[-1]).mean(0)
    std = sample.reshape(-1, X.shape[-1]).std(0) + 1e-5
    model = LightCNN(X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([(1 - y.mean()) / y.mean()])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(args.seed)
    best, best_state, history = None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(X))
        total = 0.0
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            chans = rng.integers(0, X.shape[1], len(idx))
            starts = rng.integers(0, X.shape[2] - args.crop + 1, len(idx))
            xb = np.stack([X[i, c, s:s + args.crop] for i, c, s in zip(np.sort(idx), chans, starts)]).astype(np.float32)
            yb = y[np.sort(idx)]
            xb = torch.from_numpy((xb - mean) / std)
            loss = loss_fn(model(xb), torch.from_numpy(yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        # вибір епохи — за середнім EER на dev по всіх каналах
        dev_eers = {}
        for chan in args.channels:
            items, feats = devsets[chan]
            s = score_all(model, feats, mean, std)
            bona = np.array([it.bonafide for it in items])
            dev_eers[chan] = metrics.eer(s[bona], s[~bona])[0]
        dev_mean = float(np.mean(list(dev_eers.values())))
        history.append({"epoch": epoch, "loss": total / len(X), "dev_mean_eer": dev_mean, "dev": dev_eers})
        print(f"епоха {epoch:2d}: втрата {total / len(X):.4f}, dev середній EER {dev_mean * 100:.2f} % "
              f"({time.perf_counter() - t0:.0f} с)", flush=True)
        if best is None or dev_mean < best:
            best, best_state = dev_mean, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    best_epoch = int(np.argmin([h["dev_mean_eer"] for h in history])) + 1
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save({"state": best_state, "mean": mean, "std": std}, ROOT / "models" / f"cm_cnn_{out_tag}.pt")

    report: dict = {"config": vars(args), "best_epoch": best_epoch, "history": history}
    rows = []
    scores_csv = ROOT / "results" / f"cm_cnn_scores_{out_tag}.csv"
    with open(scores_csv, "w", newline="", encoding="utf-8") as fh, ProcessPoolExecutor(args.workers) as pool:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["split", "channel", "utt", "attack", "bonafide", "score"])
        for split in ("dev", "eval"):
            for chan in args.channels:
                items, feats = devsets[chan] if split == "dev" else features_of(pool, "eval", args.eval_per_attack, chan)
                s = score_all(model, feats, mean, std)
                for it, sc in zip(items, s):
                    w.writerow([split, chan, it.utt, it.attack, int(it.bonafide), f"{sc:.6f}"])
                bona = np.array([it.bonafide for it in items])
                report.setdefault(split, {})[chan] = {"eer": metrics.eer(s[bona], s[~bona])[0]}
                if split == "eval":
                    print(f"eval {chan:10s} EER = {report['eval'][chan]['eer'] * 100:.2f} % "
                          f"({time.perf_counter() - t0:.0f} с)", flush=True)
    for chan in args.channels:
        rows.append([chan, f"{report['dev'][chan]['eer'] * 100:.2f}", f"{report['eval'][chan]['eer'] * 100:.2f}"])
    print(f"\nНайкраща епоха за dev: {best_epoch}")
    print_table(rows, ["канал", "dev EER, %", "eval EER, %"])
    out = ROOT / "results" / f"cm_cnn_{out_tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Звіт:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
