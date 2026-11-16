"""Невелика згорткова мережа для виявлення підробки за матрицею ознак (кадри × LFCC).

Архітектура — спрощена LCNN (Lavrentyeva et al., 2019; одна з найкращих у
ASVspoof 2019): згорткові блоки з активацією Max-Feature-Map (MFM) —
канали діляться навпіл і береться поелементний максимум, що працює як
вбудований відбір ознак і зменшує кількість параметрів удвічі. Наприкінці —
усереднення за часом, тож мережа приймає записи будь-якої тривалості.
"""

from __future__ import annotations

import torch
from torch import nn


class MFM(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        return torch.maximum(a, b)


def block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, 2 * c_out, 3, padding=1), MFM(), nn.BatchNorm2d(c_out),
                         nn.MaxPool2d(2))


class LightCNN(nn.Module):
    """Вхід (батч, кадри, ознаки) → логіт «справжній» (більше — справжніший)."""

    def __init__(self, n_features: int = 60, width: int = 16, dropout: float = 0.3):
        super().__init__()
        self.body = nn.Sequential(block(1, width), block(width, 2 * width), block(2 * width, 2 * width),
                                  block(2 * width, 4 * width))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width * (n_features // 16), 64), MFMLinear(),
                                  nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x.unsqueeze(1))            # (батч, канали, кадри/16, ознаки/16)
        h = h.mean(dim=2).flatten(1)             # усереднення за часом
        return self.head(h).squeeze(1)


class MFMLinear(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        return torch.maximum(a, b)
