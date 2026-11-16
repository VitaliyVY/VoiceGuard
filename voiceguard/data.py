"""Протоколи ASVspoof 2019 LA.

Структура архіву LA.zip:
  LA/ASVspoof2019_LA_{train,dev,eval}/flac/*.flac
  LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt
      рядок: SPEAKER FILE - ATTACK KEY      (ATTACK: "-" або A01…A19, KEY: bonafide | spoof)
  LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.{dev,eval}.{female,male}.trn.txt
      рядок: SPEAKER FILE1,FILE2,…           (записи для реєстрації мовця)
  LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.{dev,eval}.gi.trl.txt
      рядок: SPEAKER FILE ATTACK KEY         (KEY: target | nontarget | spoof)

Частини не перетинаються за мовцями; атаки A07–A19 у eval частково нові
(відсутні в train) — це перевірка узагальнення на невідомі методи синтезу.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SPLIT_FILES = {"train": "train.trn", "dev": "dev.trl", "eval": "eval.trl"}


@dataclass(frozen=True)
class CMItem:
    speaker: str
    utt: str
    attack: str        # "-" для справжніх записів, A01…A19 — метод підробки
    bonafide: bool


@dataclass(frozen=True)
class ASVTrial:
    speaker: str       # заявлений мовець
    utt: str
    attack: str
    key: str           # target | nontarget | spoof


def la_root(root: str | Path) -> Path:
    root = Path(root)
    return root / "LA" if (root / "LA").is_dir() else root


def cm_protocol(root: str | Path, split: str) -> list[CMItem]:
    path = la_root(root) / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{SPLIT_FILES[split]}.txt"
    items = []
    for line in path.read_text().splitlines():
        spk, utt, _, attack, key = line.split()
        items.append(CMItem(spk, utt, attack, key == "bonafide"))
    return items


def audio_path(root: str | Path, split: str, utt: str) -> Path:
    return la_root(root) / f"ASVspoof2019_LA_{split}" / "flac" / f"{utt}.flac"


def asv_enrollment(root: str | Path, split: str) -> dict[str, list[str]]:
    """Мовець → записи для реєстрації (split: dev | eval)."""
    out: dict[str, list[str]] = {}
    for gender in ("female", "male"):
        path = la_root(root) / "ASVspoof2019_LA_asv_protocols" / f"ASVspoof2019.LA.asv.{split}.{gender}.trn.txt"
        for line in path.read_text().splitlines():
            spk, utts = line.split()
            out[spk] = utts.split(",")
    return out


def asv_trials(root: str | Path, split: str) -> list[ASVTrial]:
    path = la_root(root) / "ASVspoof2019_LA_asv_protocols" / f"ASVspoof2019.LA.asv.{split}.gi.trl.txt"
    return [ASVTrial(*line.split()) for line in path.read_text().splitlines()]
