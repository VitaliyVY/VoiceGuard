"""Запуск десктопної програми VoiceGuard (подвійний клік — без вікна консолі)."""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from voiceguard.qt_app import main          # сучасний інтерфейс (PySide6)
except ImportError:                             # без PySide6 — простий інтерфейс Tkinter
    from voiceguard.gui import main

main()
