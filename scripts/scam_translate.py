"""Переклад діалогів BothBosu/scam-dialogue (Apache-2.0) українською локальною моделлю.

Джерело — синтетичні англомовні телефонні розмови (згенеровані мовною моделлю):
шахрайські (ssn, refund, support, reward) і звичайні (delivery, insurance,
telemarketing, wrong). Переклад дослівний, без адаптації до українських реалій —
щоб модель не вписала в текст саме ті слова, які шукають правила scam.py.
Обмеження: це машинний переклад синтетичних діалогів, а не справжні розмови;
схеми американські (SSN, IRS, Amazon), не українські.

Можна переривати: готові переклади зберігаються й пропускаються.

Приклад:
    python scripts/scam_translate.py --per-type 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request

from _common import ROOT
from voiceguard import ai

SRC = ROOT / "data" / "external" / "scam_dialogue" / "scam-dialogue_test.csv"
DST = ROOT / "data" / "uk" / "scam_dialogue_uk.csv"
PROMPT = ("Translate this phone conversation into Ukrainian. Keep the speaker labels exactly as "
          "'caller:' and 'receiver:' at the start of each turn. Translate literally and completely; "
          "do not add, remove or explain anything. Output only the translation.")


def translate(text: str, model: str) -> str:
    body = {"model": model, "stream": False, "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 1500},
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}]}
    req = urllib.request.Request(f"{ai.URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(4):                     # сервер іноді відповідає 500 при нестачі пам'яті
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if attempt == 3:
                raise
            print(f"  помилка сервера {e.code}, повтор через 60 с", flush=True)
            time.sleep(60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-type", type=int, default=20)
    ap.add_argument("--model", default=ai.DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ok, why = ai.available(args.model, timeout=10)
    if not ok:
        raise SystemExit(why)
    with open(SRC, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for i, r in enumerate(rows):
        r["id"] = str(i)
    rng = random.Random(args.seed)
    plan = []
    for t in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == t]
        plan += rng.sample(group, min(args.per_type, len(group)))
    done = set()
    if DST.exists():
        with open(DST, encoding="utf-8") as fh:
            done = {r["id"] for r in csv.DictReader(fh)}
    new = not DST.exists()
    DST.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(DST, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "type", "label", "dialogue_uk", "dialogue_en"])
        if new:
            w.writeheader()
        todo = [r for r in plan if r["id"] not in done]
        for k, r in enumerate(todo, 1):
            uk = translate(r["dialogue"], args.model)
            w.writerow({"id": r["id"], "type": r["type"], "label": r["label"], "dialogue_uk": uk,
                        "dialogue_en": r["dialogue"]})
            fh.flush()
            print(f"{k}/{len(todo)} {r['type']} ({time.perf_counter() - t0:.0f} с)", flush=True)


if __name__ == "__main__":
    main()
