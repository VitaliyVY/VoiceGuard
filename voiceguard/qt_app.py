"""VoiceGuard — десктопна програма на Qt (PySide6).

Сторінки: «Перевірка», «Мовці», «Журнал», «Статистика», «Налаштування».
Власні віджети малюються QPainter зі згладжуванням і анімуються
QVariantAnimation: кругові шкали, пульсуюча кнопка запису, живий осцилограф,
кроки обробки зі спінером, спектрограма з поступовою появою, сповіщення.
Важкі операції виконуються в QThreadPool, результати повертаються сигналами Qt.
"""

from __future__ import annotations

import csv
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import (QEasingCurve, QObject, QPointF, QRectF, QRunnable, QSize, Qt, QThreadPool, QTimer,
                            QVariantAnimation, Signal)
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QPlainTextEdit, QTextBrowser, QFileDialog, QFrame, QGraphicsOpacityEffect,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import audio, channel

SR = audio.SAMPLE_RATE
ICON_FONT = "Segoe MDL2 Assets"
ICONS = {"batch": "\ue8b7", "text": "\ue8bd", "check": "", "speakers": "", "journal": "", "stats": "", "settings": "",
         "open": "", "play": "", "stop": "", "mic": "", "ok": "", "no": "",
         "warn": "", "delete": "", "add": "", "export": "", "shield": ""}
CHANNEL_CHOICES = {
    "Без змін (як записано)": "clean",
    "Голосове Telegram / Viber (Opus 16 кбіт/с)": "opus16",
    "Мобільний дзвінок (AMR-NB)": "amrnb",
    "HD Voice / VoLTE (AMR-WB)": "amrwb",
    "Стаціонарний телефон (G.711)": "telephone",
    "Шум 20 дБ": "noise20",
}
THEMES = {
    "dark": {"bg": "#111318", "side": "#16191f", "card": "#1d2129", "card2": "#232833", "border": "#2c323d",
             "text": "#e8eaee", "muted": "#9aa1ad", "accent": "#3d8bfd", "good": "#2fbf71", "bad": "#f05252",
             "warn": "#f0a73a", "track": "#2c323d"},
    "light": {"bg": "#f3f4f7", "side": "#ffffff", "card": "#ffffff", "card2": "#f6f7f9", "border": "#e1e4ea",
              "text": "#1c1f25", "muted": "#68707c", "accent": "#2a78d6", "good": "#1f9d55", "bad": "#d93636",
              "warn": "#b7791f", "track": "#e6e8ed"},
}
T = dict(THEMES["dark"])


def qc(name: str, alpha: int | None = None) -> QColor:
    c = QColor(T[name])
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def stylesheet() -> str:
    return f"""
    QWidget {{ background: {T['bg']}; color: {T['text']}; }}
    QLabel {{ background: transparent; }}
    QFrame#card {{ background: {T['card']}; border: 1px solid {T['border']}; border-radius: 14px; }}
    QFrame#side {{ background: {T['side']}; border-right: 1px solid {T['border']}; }}
    QTextBrowser#vgBrowser, QPlainTextEdit#vgBrowser {{ background: {T['card2']}; color: {T['text']}; border: 1px solid {T['border']};
        border-radius: 10px; padding: 8px; }}
    QLabel#h1 {{ font-size: 18pt; font-weight: 600; }}
    QLabel#h2 {{ font-size: 11.5pt; font-weight: 600; }}
    QPushButton {{ font-size: 10pt; }}
    QLabel#muted {{ color: {T['muted']}; }}
    QPushButton {{ background: {T['card2']}; border: 1px solid {T['border']}; border-radius: 9px; padding: 8px 14px; }}
    QPushButton:hover {{ border-color: {T['accent']}; }}
    QPushButton:pressed {{ background: {T['border']}; }}
    QPushButton:disabled {{ color: {T['muted']}; }}
    QPushButton#primary {{ background: {T['accent']}; color: white; border: none; font-weight: 600; }}
    QPushButton#primary:hover {{ background: {QColor(T['accent']).lighter(115).name()}; }}
    QPushButton#primary:disabled {{ background: {T['border']}; color: {T['muted']}; }}
    QPushButton#nav {{ background: transparent; border: none; border-radius: 10px; text-align: left; padding: 10px 14px; }}
    QPushButton#nav:hover {{ background: {T['card2']}; }}
    QPushButton#nav:checked {{ background: {qc('accent', 40).name(QColor.HexArgb)}; color: {T['text']}; font-weight: 600; }}
    QComboBox, QLineEdit, QSpinBox {{ background: {T['card2']}; border: 1px solid {T['border']}; border-radius: 9px;
                                      padding: 6px 10px; }}
    QComboBox:hover, QLineEdit:focus {{ border-color: {T['accent']}; }}
    QComboBox QAbstractItemView {{ background: {T['card']}; border: 1px solid {T['border']};
                                   selection-background-color: {T['accent']}; }}
    QTableWidget, QListWidget {{ background: {T['card']}; border: none; gridline-color: {T['border']}; }}
    QHeaderView::section {{ background: {T['card2']}; color: {T['muted']}; border: none; padding: 6px; }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QSlider::groove:horizontal {{ height: 6px; background: {T['track']}; border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: {T['accent']}; width: 18px; margin: -6px 0; border-radius: 9px; }}
    QSlider::sub-page:horizontal {{ background: {T['accent']}; border-radius: 3px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {T['border']}; border-radius: 4px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {T['muted']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {T['border']}; border-radius: 4px; min-width: 30px; }}
    QToolTip {{ background: {T['card2']}; color: {T['text']}; border: 1px solid {T['border']}; }}
    """


def ui_font(size: float = 10, weight=QFont.Normal) -> QFont:
    """Шрифт інтерфейсу з запасним шрифтом іконок (гліфи MDL2 у тексті кнопок)."""
    f = QFont("Segoe UI", size, weight)
    f.setFamilies(["Segoe UI", ICON_FONT])
    return f


def icon_label(name: str, size: int = 14, color: str = "muted") -> QLabel:
    lab = QLabel(ICONS[name])
    lab.setFont(QFont(ICON_FONT, size))
    lab.setStyleSheet(f"color: {T[color]};")
    return lab


def card(*widgets, margins=16, spacing=10, expand=None) -> QFrame:
    """Картка; expand — віджет, що забирає вільну висоту (інакше вміст притиснуто догори)."""
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(margins, margins, margins, margins)
    lay.setSpacing(spacing)
    for w in widgets:
        if isinstance(w, QWidget):
            lay.addWidget(w, 1 if w is expand else 0)
        else:
            lay.addLayout(w)
    if expand is None:
        lay.addStretch(1)
    return f


def text(s: str, obj: str | None = None, wrap: bool = True) -> QLabel:
    lab = QLabel(s)
    if obj:
        lab.setObjectName(obj)
    lab.setWordWrap(wrap)
    return lab


def anim(parent, start, end, ms, on_value, easing=QEasingCurve.OutCubic, loop=False) -> QVariantAnimation:
    a = QVariantAnimation(parent)
    a.setStartValue(float(start))
    a.setEndValue(float(end))
    a.setDuration(ms)
    a.setEasingCurve(easing)
    a.valueChanged.connect(on_value)
    if loop:
        a.setLoopCount(-1)
    return a


# --------------------------------------------------------------------------- фонові завдання
class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str)


class Task(QRunnable):
    """Завдання для QThreadPool. Живі завдання зберігаються в RUNNING, інакше Python може знищити
    об'єкт сигналів раніше, ніж головний потік отримає результат."""

    RUNNING: set = set()

    def __init__(self, fn, *args, with_progress: bool = False):
        super().__init__()
        self.setAutoDelete(False)
        self.fn, self.args, self.with_progress = fn, args, with_progress
        self.signals = _Signals()
        Task.RUNNING.add(self)
        self.signals.done.connect(lambda _: Task.RUNNING.discard(self))
        self.signals.failed.connect(lambda _: Task.RUNNING.discard(self))

    def run(self):
        try:
            if self.with_progress:
                res = self.fn(*self.args, progress=self.signals.progress.emit)
            else:
                res = self.fn(*self.args)
            self.signals.done.emit(res)
        except Exception as e:                                   # noqa: BLE001
            self.signals.failed.emit(str(e))


# --------------------------------------------------------------------------- власні віджети
class Gauge(QWidget):
    """Кругова шкала 0..1 з поділкою порогу (0,5); значення змінюється з анімацією."""

    def __init__(self, title: str, size: int = 168, risk: bool = False):
        super().__init__()
        self.risk = risk                     # шкала ризику: зелена → жовта → червона, поділки 30 % і 60 %
        self.title, self.caption = title, "—"
        self._v, self._shown = None, 0.0
        self.setFixedSize(size, size + 26)
        self._anim = None

    def set_value(self, v: float | None, caption: str = ""):
        self.caption = caption
        self._v = v
        if v is None:
            self._shown = 0.0
            self.update()
            return
        self._anim = anim(self, self._shown, v, 1100, self._step)
        self._anim.start()

    def _step(self, x):
        self._shown = float(x)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self.width()
        r = QRectF(14, 14, s - 28, s - 28)
        start, span = 225 * 16, -270 * 16
        pen = QPen(qc("track"), 13, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(r, start, span)
        if self._v is not None:
            if self.risk:
                col = qc("good") if self._shown < 0.3 else qc("warn") if self._shown < 0.6 else qc("bad")
            else:
                col = qc("good") if self._shown >= 0.5 else qc("bad")
            grad_pen = QPen(col, 13, Qt.SolidLine, Qt.RoundCap)
            p.setPen(grad_pen)
            p.drawArc(r, start, int(span * max(self._shown, 0.004)))
        # поріг (для шкали ризику — межі «низький / середній / високий»)
        cx, cy, rad = r.center().x(), r.center().y(), r.width() / 2
        p.setPen(QPen(qc("text"), 3, Qt.SolidLine, Qt.RoundCap))
        for frac in ((0.3, 0.6) if self.risk else (0.5,)):
            ang = math.radians(225 - 270 * frac)
            p.drawLine(QPointF(cx + (rad - 11) * math.cos(ang), cy - (rad - 11) * math.sin(ang)),
                       QPointF(cx + (rad + 11) * math.cos(ang), cy - (rad + 11) * math.sin(ang)))
        p.setPen(qc("text"))
        p.setFont(QFont("Segoe UI", 21, QFont.DemiBold))
        p.drawText(QRectF(0, cy - 26, s, 36), Qt.AlignCenter, "—" if self._v is None else f"{100 * self._shown:.0f}%")
        p.setPen(qc("muted"))
        p.setFont(QFont("Segoe UI", 8.5))
        p.drawText(QRectF(10, cy + 10, s - 20, 34), Qt.AlignCenter | Qt.TextWordWrap, self.caption)
        p.setPen(qc("text"))
        p.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
        p.drawText(QRectF(0, s - 6, s, 26), Qt.AlignCenter, self.title)


class RecordButton(QPushButton):
    """Кругла кнопка мікрофона; під час запису пульсує й показує зворотний відлік."""

    def __init__(self, size: int = 64):
        super().__init__()
        self.setFixedSize(size + 24, size + 24)
        self.setCursor(Qt.PointingHandCursor)
        self.recording, self.remaining, self._pulse = False, 0.0, 0.0
        self._anim = anim(self, 0, 1, 1100, self._set_pulse, QEasingCurve.OutQuad, loop=True)
        self.setStyleSheet("background: transparent; border: none;")
        self.setToolTip("Записати з мікрофона")

    def _set_pulse(self, v):
        self._pulse = float(v)
        self.update()

    def set_recording(self, on: bool):
        self.recording = on
        (self._anim.start() if on else self._anim.stop())
        self._pulse = 0.0
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        base = (self.width() - 24) / 2
        if self.recording:
            ring = base + 10 * self._pulse
            p.setPen(Qt.NoPen)
            p.setBrush(qc("bad", int(110 * (1 - self._pulse))))
            p.drawEllipse(c, ring, ring)
        col = qc("bad") if self.recording else (qc("accent").lighter(112) if self.underMouse() else qc("accent"))
        p.setBrush(col)
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, base, base)
        p.setPen(QColor("white"))
        if self.recording:
            p.setFont(QFont("Segoe UI", 13, QFont.DemiBold))
            p.drawText(self.rect(), Qt.AlignCenter, f"{self.remaining:.0f}")
        else:
            p.setFont(QFont(ICON_FONT, 20))
            p.drawText(self.rect(), Qt.AlignCenter, ICONS["mic"])


PLOT_LEFT, PLOT_RIGHT = 44, 8          # однакові поля → час на всіх графіках збігається по вертикалі


def draw_playhead(p: QPainter, area: QRectF, frac: float | None, color: QColor | None = None):
    if frac is None:
        return
    x = area.left() + area.width() * min(max(frac, 0.0), 1.0)
    p.setPen(QPen(color or qc("text"), 2))
    p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
    p.setPen(Qt.NoPen)
    p.setBrush(qc("accent"))
    p.drawEllipse(QPointF(x, area.top() + 2), 4, 4)


class WaveView(QWidget):
    """Осцилограма: «наживо» під час запису (останні 3 с) або весь запис з анімованою появою."""

    def __init__(self, height: int = 130):
        super().__init__()
        self.playhead: float | None = None
        self.setMinimumHeight(height)
        self.live: deque | None = None
        self.env: np.ndarray | None = None
        self.level = 0.0
        self._reveal = 1.0
        self._anim = None
        self.placeholder = "Запишіть фразу або відкрийте файл"

    def start_live(self, buffer: deque):
        self.live, self.env = buffer, None
        self.update()

    def set_signal(self, x: np.ndarray | None):
        self.live = None
        if x is None or not len(x):
            self.env = None
            self.update()
            return
        cols = 600
        chunks = np.array_split(np.abs(x), cols)
        self.env = np.array([c.max() if len(c) else 0.0 for c in chunks])
        self.env = self.env / (self.env.max() + 1e-9)
        self._anim = anim(self, 0, 1, 900, self._set_reveal, QEasingCurve.OutCubic)
        self._anim.start()

    def _set_reveal(self, v):
        self._reveal = float(v)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2
        p.setPen(QPen(qc("border"), 1))
        p.drawLine(0, int(mid), w, int(mid))
        if self.live is not None:
            data = np.array(self.live, dtype=np.float32)
            if len(data) > 2:
                cols = w // 2
                seg = np.array_split(np.abs(data), cols)
                env = np.array([s.max() if len(s) else 0 for s in seg])
                env = np.clip(env / max(0.25, env.max()), 0, 1)
                self._draw_env(p, env, w, h, 1.0, qc("bad"))
            return
        if self.env is None:
            p.setPen(qc("muted"))
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        p.save()
        p.translate(PLOT_LEFT, 0)
        self._draw_env(p, self.env, w - PLOT_LEFT - PLOT_RIGHT, h, self._reveal, qc("accent"))
        p.restore()
        draw_playhead(p, QRectF(PLOT_LEFT, 2, w - PLOT_LEFT - PLOT_RIGHT, h - 4), self.playhead)

    @staticmethod
    def _draw_env(p, env, w, h, reveal, color):
        n = len(env)
        upto = max(1, int(n * reveal))
        path = QPainterPath()
        mid = h / 2
        xs = np.linspace(0, w * upto / n, upto)
        path.moveTo(0, mid)
        for x, e in zip(xs, env[:upto]):
            path.lineTo(x, mid - e * (h / 2 - 6))
        for x, e in zip(xs[::-1], env[:upto][::-1]):
            path.lineTo(x, mid + e * (h / 2 - 6))
        path.closeSubpath()
        g = QLinearGradient(0, 0, 0, h)
        c1, c2 = QColor(color), QColor(color)
        c1.setAlpha(230)
        c2.setAlpha(120)
        g.setColorAt(0, c2)
        g.setColorAt(0.5, c1)
        g.setColorAt(1, c2)
        p.setPen(Qt.NoPen)
        p.setBrush(g)
        p.drawPath(path)


class LevelMeter(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(8)
        self.level = 0.0

    def set_level(self, v: float):
        self.level = 0.7 * self.level + 0.3 * min(1.0, v)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(qc("track"))
        p.drawRoundedRect(self.rect(), 4, 4)
        col = qc("good") if self.level < 0.8 else qc("warn") if self.level < 0.97 else qc("bad")
        p.setBrush(col)
        p.drawRoundedRect(QRectF(0, 0, self.width() * self.level, self.height()), 4, 4)


class SpectrogramView(QWidget):
    """Спектрограма (діапазон 70 дБ) з анімованою появою зліва направо і лінією 4 кГц."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(140)
        self.img: QImage | None = None
        self.duration = 0.0
        self.playhead: float | None = None
        self._reveal = 1.0
        self._anim = None
        import matplotlib

        lut = (matplotlib.colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint32)
        self.lut = (0xFF << 24) | (lut[:, 0] << 16) | (lut[:, 1] << 8) | lut[:, 2]

    def set_signal(self, x: np.ndarray | None):
        if x is None or len(x) < 512:
            self.img = None
            self.update()
            return
        from scipy import signal as sps

        _, _, z = sps.stft(x, fs=SR, nperseg=512, noverlap=384)
        db = 20 * np.log10(np.abs(z) + 1e-9)
        top = db.max()
        idx = np.clip((db - (top - 70)) / 70 * 255, 0, 255).astype(np.uint8)[::-1]
        argb = np.ascontiguousarray(self.lut[idx]).astype(np.uint32)
        self._buf = argb
        self.img = QImage(argb.data, argb.shape[1], argb.shape[0], argb.shape[1] * 4, QImage.Format_ARGB32)
        self.duration = len(x) / SR
        self._anim = anim(self, 0, 1, 1200, self._set_reveal, QEasingCurve.InOutCubic)
        self._anim.start()

    def _set_reveal(self, v):
        self._reveal = float(v)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        left, bottom = PLOT_LEFT, 22
        area = QRectF(left, 6, self.width() - left - PLOT_RIGHT, self.height() - bottom - 6)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#000000"))
        p.drawRoundedRect(area, 8, 8)
        p.setPen(qc("muted"))
        p.setFont(QFont("Segoe UI", 8))
        for khz in (0, 2, 4, 6, 8):
            y = area.bottom() - area.height() * khz / 8
            p.drawText(QRectF(0, y - 8, left - 6, 16), Qt.AlignRight | Qt.AlignVCenter, f"{khz} кГц")
        if self.img is None:
            p.drawText(area, Qt.AlignCenter, "Спектрограма з’явиться після вибору запису")
            return
        clip = QRectF(area.left(), area.top(), area.width() * self._reveal, area.height())
        p.save()
        path = QPainterPath()
        path.addRoundedRect(area, 8, 8)
        p.setClipPath(path)
        p.setClipRect(clip, Qt.IntersectClip)
        p.drawImage(area, self.img)
        p.restore()
        y4 = area.bottom() - area.height() / 2
        pen = QPen(QColor(255, 255, 255, 170), 1, Qt.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(area.left(), y4), QPointF(area.right(), y4))
        p.setPen(QColor(255, 255, 255, 200))
        p.drawText(QRectF(area.right() - 230, y4 - 18, 224, 16), Qt.AlignRight, "межа телефонної смуги (4 кГц)")
        p.setPen(qc("muted"))
        for k in range(0, int(self.duration) + 1):
            x = area.left() + area.width() * k / max(self.duration, 1e-9)
            if x <= area.right():
                p.drawText(QRectF(x - 20, area.bottom() + 3, 40, 16), Qt.AlignCenter, f"{k} с")
        draw_playhead(p, area, self.playhead, QColor(255, 255, 255, 230))


class TimelineView(QWidget):
    """Підозрілість у часі: впевненість «справжній» у ковзних вікнах.

    Вище порогу (50 %) — зелена заливка, нижче — червона; сірі смуги — вікна майже без
    мовлення (там оцінка малоінформативна). Наведення миші показує час і значення.
    """

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(130)
        self.setMouseTracking(True)
        self.t: list[float] = []
        self.c: list[float] = []
        self.speech: list[bool] = []
        self.duration = 0.0
        self.playhead: float | None = None
        self._reveal = 1.0
        self._anim = None
        self.hover: int | None = None

    def set_timeline(self, tl: dict | None):
        if not tl:
            self.t, self.c, self.speech = [], [], []
        else:
            self.t, self.c, self.speech = tl["t"], tl["confidence"], tl["speech"]
            self.duration = tl["duration"]
        self.hover = None
        self._anim = anim(self, 0, 1, 1300, self._set_reveal, QEasingCurve.InOutCubic)
        self._anim.start()

    def _set_reveal(self, v):
        self._reveal = float(v)
        self.update()

    def _area(self) -> QRectF:
        return QRectF(PLOT_LEFT, 8, self.width() - PLOT_LEFT - PLOT_RIGHT, self.height() - 30)

    def _xy(self, area, t, c):
        return QPointF(area.left() + area.width() * t / max(self.duration, 1e-9), area.bottom() - area.height() * c)

    def mouseMoveEvent(self, e):
        if not self.t:
            return
        area = self._area()
        t = (e.position().x() - area.left()) / max(area.width(), 1) * self.duration
        i = int(np.argmin(np.abs(np.array(self.t) - t)))
        if i != self.hover:
            self.hover = i
            self.update()
        from PySide6.QtWidgets import QToolTip

        note = "" if self.speech[i] else " (майже без мовлення)"
        verdict = "схоже на справжній голос" if self.c[i] >= 0.5 else "ознаки підробки"
        QToolTip.showText(e.globalPosition().toPoint(),
                          f"{self.t[i]:.1f} с: {100 * self.c[i]:.0f} % — {verdict}{note}", self)

    def leaveEvent(self, _):
        self.hover = None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        area = self._area()
        p.setPen(Qt.NoPen)
        p.setBrush(qc("card2"))
        p.drawRoundedRect(area, 8, 8)
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(qc("muted"))
        for v, lab in ((1.0, "100 %"), (0.5, "поріг"), (0.0, "0 %")):
            y = area.bottom() - area.height() * v
            p.drawText(QRectF(0, y - 8, PLOT_LEFT - 6, 16), Qt.AlignRight | Qt.AlignVCenter, lab)
        y50 = area.bottom() - area.height() * 0.5
        if not self.t:
            p.drawText(area, Qt.AlignCenter, "Графік з’явиться після перевірки")
            return
        hop = (self.t[1] - self.t[0]) if len(self.t) > 1 else self.duration
        for t, sp in zip(self.t, self.speech):                          # тиша — сірі смуги
            if not sp:
                a = self._xy(area, max(t - hop / 2, 0), 1).x()
                b = self._xy(area, min(t + hop / 2, self.duration), 1).x()
                p.setBrush(qc("border", 140))
                p.drawRect(QRectF(a, area.top(), b - a, area.height()))
        ts = [0.0] + list(self.t) + [self.duration]                    # краї запису — значення крайніх вікон
        cs = [self.c[0]] + list(self.c) + [self.c[-1]]
        n = max(2, int(round(len(ts) * self._reveal)))
        pts = [self._xy(area, t, c) for t, c in zip(ts[:n], cs[:n])]
        fill = QPainterPath(QPointF(pts[0].x(), y50))
        for q in pts:
            fill.lineTo(q)
        fill.lineTo(QPointF(pts[-1].x(), y50))
        fill.closeSubpath()
        for color, top in (("good", True), ("bad", False)):             # зелене вище порогу, червоне нижче
            p.save()
            p.setClipRect(QRectF(area.left(), area.top(), area.width(), y50 - area.top()) if top
                          else QRectF(area.left(), y50, area.width(), area.bottom() - y50))
            p.setPen(Qt.NoPen)
            p.setBrush(qc(color, 90))
            p.drawPath(fill)
            p.restore()
        line = QPainterPath(pts[0])
        for q in pts[1:]:
            line.lineTo(q)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(qc("text"), 2))
        p.drawPath(line)
        p.setPen(QPen(qc("muted"), 1, Qt.DashLine))
        p.drawLine(QPointF(area.left(), y50), QPointF(area.right(), y50))
        if self.hover is not None and self.hover < n:
            q = self._xy(area, self.t[self.hover], self.c[self.hover])
            p.setPen(QPen(qc("text"), 2))
            p.setBrush(qc("good") if self.c[self.hover] >= 0.5 else qc("bad"))
            p.drawEllipse(q, 5, 5)
        p.setPen(qc("muted"))
        for k in range(0, int(self.duration) + 1):
            x = area.left() + area.width() * k / max(self.duration, 1e-9)
            p.drawText(QRectF(x - 20, area.bottom() + 4, 40, 16), Qt.AlignCenter, f"{k} с")
        draw_playhead(p, area, self.playhead)


class Steps(QWidget):
    """Кроки обробки: очікує → виконується (спінер) → готово (галочка)."""

    NAMES = [("analysis", "Аналіз сигналу"), ("detector", "Детектор підробки"), ("timeline", "Підозрілість у часі"),
             ("verifier", "Верифікація мовця"), ("asr", "Розпізнавання мовлення"), ("content", "Аналіз змісту")]

    def __init__(self):
        super().__init__()
        self.state = {k: "idle" for k, _ in self.NAMES}
        self.times: dict[str, float] = {}
        self._angle = 0.0
        self._spin = anim(self, 0, 360, 900, self._rot, QEasingCurve.Linear, loop=True)
        self.setFixedHeight(6 * 30 + 6)

    def _rot(self, v):
        self._angle = float(v)
        self.update()

    def reset(self, with_verifier: bool, with_asr: bool = True):
        self.state = {k: "idle" for k, _ in self.NAMES}
        if not with_verifier:
            self.state["verifier"] = "skip"
        if not with_asr:
            self.state["asr"] = self.state["content"] = "off"
        self.times = {}
        self.update()

    def advance(self, step: str):
        order = [k for k, _ in self.NAMES]
        for k in order:
            if self.state[k] == "run":
                self.state[k] = "done"
        if step in self.state and self.state[step] not in ("skip", "off"):
            self.state[step] = "run"
            self._spin.start()
        if step == "done":
            self._spin.stop()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        for i, (k, name) in enumerate(self.NAMES):
            y = 4 + i * 30
            c = QPointF(12, y + 11)
            st = self.state[k]
            if st == "done":
                p.setPen(Qt.NoPen)
                p.setBrush(qc("good"))
                p.drawEllipse(c, 9, 9)
                p.setPen(QPen(QColor("white"), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                path = QPainterPath(QPointF(c.x() - 4, c.y()))
                path.lineTo(c.x() - 1, c.y() + 3.5)
                path.lineTo(c.x() + 4.5, c.y() - 3.5)
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
            elif st == "run":
                p.setPen(QPen(qc("track"), 2.5))
                p.drawEllipse(c, 8, 8)
                p.setPen(QPen(qc("accent"), 2.5, Qt.SolidLine, Qt.RoundCap))
                p.drawArc(QRectF(c.x() - 8, c.y() - 8, 16, 16), int(-self._angle * 16), 100 * 16)
            else:
                p.setPen(QPen(qc("track"), 2.5))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, 8, 8)
            p.setPen(qc("text") if st in ("run", "done") else qc("muted"))
            p.setFont(QFont("Segoe UI", 9.5, QFont.DemiBold if st == "run" else QFont.Normal))
            label = name + ("  (мовця не вибрано)" if st == "skip" else "  (вимкнено в налаштуваннях)" if st == "off" else "")
            p.drawText(QRectF(30, y, self.width() - 100, 22), Qt.AlignVCenter, label)
            if k in self.times:
                p.setPen(qc("muted"))
                p.drawText(QRectF(self.width() - 90, y, 86, 22), Qt.AlignRight | Qt.AlignVCenter,
                           f"{1000 * self.times[k]:.0f} мс")


class VerdictCard(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        self.icon = QLabel(ICONS["shield"])
        self.icon.setFont(QFont(ICON_FONT, 30))
        self.icon.setFixedWidth(52)
        col = QVBoxLayout()
        self.title = text("Очікування запису", "h1")
        self.sub = text("Запишіть 3–5 с мовлення або відкрийте аудіофайл, потім натисніть «Перевірити».", "muted")
        col.addWidget(self.title)
        col.addWidget(self.sub)
        lay.addWidget(self.icon)
        lay.addLayout(col, 1)
        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self._fx.setOpacity(1.0)
        self._anim = None
        self.kind = "idle"

    def show_verdict(self, kind: str, title: str, sub: str):
        self.kind = kind
        color = {"good": T["good"], "bad": T["bad"], "warn": T["warn"]}.get(kind, T["muted"])
        glyph = {"good": ICONS["ok"], "bad": ICONS["no"], "warn": ICONS["warn"]}.get(kind, ICONS["shield"])
        self.icon.setText(glyph)
        self.icon.setStyleSheet(f"color: {color};")
        self.title.setText(title)
        self.title.setStyleSheet(f"color: {color};")
        self.sub.setText(sub)
        tint = QColor(color)
        tint.setAlpha(28)
        self.setStyleSheet(f"QFrame#card {{ background: {tint.name(QColor.HexArgb)}; border: 1px solid {color}; "
                           f"border-radius: 14px; }}")
        self._anim = anim(self, 0.0, 1.0, 600, lambda v: self._fx.setOpacity(float(v)))
        self._anim.start()

    def reset(self):
        self.kind = "idle"
        self.setStyleSheet("")
        self.icon.setText(ICONS["shield"])
        self.icon.setStyleSheet(f"color: {T['muted']};")
        self.title.setStyleSheet("")


class Toast(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setFixedWidth(360)
        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self.hide()
        self._timer = QTimer(self, singleShot=True, timeout=self._fade_out)
        self._anim = None

    def show_msg(self, msg: str, kind: str = "info"):
        color = {"good": T["good"], "bad": T["bad"], "warn": T["warn"]}.get(kind, T["accent"])
        self.setStyleSheet(f"background: {T['card2']}; color: {T['text']}; border: 1px solid {T['border']}; "
                           f"border-left: 4px solid {color}; border-radius: 10px; padding: 12px 14px;")
        self.setText(msg)
        self.adjustSize()
        par = self.parentWidget()
        self.move(par.width() - self.width() - 24, 20)              # угорі праворуч — не закриває графіки
        self.show()
        self.raise_()
        self._anim = anim(self, 0, 1, 250, lambda v: self._fx.setOpacity(float(v)))
        self._anim.start()
        self._timer.start(3200)

    def _fade_out(self):
        self._anim = anim(self, 1, 0, 400, lambda v: self._fx.setOpacity(float(v)))
        self._anim.finished.connect(self.hide)
        self._anim.start()


class StatTile(QFrame):
    def __init__(self, title: str, color: str):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        self.value = QLabel("0")
        self.value.setFont(QFont("Segoe UI", 24, QFont.DemiBold))
        self.value.setStyleSheet(f"color: {T[color]};")
        lay.addWidget(self.value)
        lay.addWidget(text(title, "muted"))
        self._anim = None

    def set_count(self, n: int):
        self._anim = anim(self, 0, n, 900, lambda v: self.value.setText(f"{round(float(v))}"))
        self._anim.start()


class BarChart(QWidget):
    """Горизонтальні стовпці «канал → EER» або «день → кількість» з анімованим ростом і підписами."""

    def __init__(self, unit: str = "%"):
        super().__init__()
        self.rows: list[tuple[str, float, str]] = []
        self.unit = unit
        self._grow = 1.0
        self._anim = None
        self.setMinimumHeight(120)

    def set_rows(self, rows: list[tuple[str, float, str]], vmax: float | None = None):
        self.rows, self.vmax = rows, vmax or (max((v for _, v, _ in rows), default=1.0) or 1.0)
        self.setMinimumHeight(28 * len(rows) + 10)
        self._anim = anim(self, 0, 1, 900, self._set_grow)
        self._anim.start()

    def _set_grow(self, v):
        self._grow = float(v)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        label_w = 190
        for i, (name, v, color) in enumerate(self.rows):
            y = 4 + i * 28
            p.setPen(qc("text"))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(0, y, label_w - 10, 20), Qt.AlignRight | Qt.AlignVCenter, name)
            full = self.width() - label_w - 60
            p.setPen(Qt.NoPen)
            p.setBrush(qc("track"))
            p.drawRoundedRect(QRectF(label_w, y + 4, full, 12), 6, 6)
            p.setBrush(qc(color))
            p.drawRoundedRect(QRectF(label_w, y + 4, max(4, full * v / self.vmax * self._grow), 12), 6, 6)
            p.setPen(qc("muted"))
            p.drawText(QRectF(label_w + full + 6, y, 54, 20), Qt.AlignLeft | Qt.AlignVCenter,
                       f"{v:.1f}{self.unit}" if self.unit else f"{v:.0f}")


class Avatar(QWidget):
    def __init__(self, name: str, size: int = 44):
        super().__init__()
        self.name = name
        self.setFixedSize(size, size)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        hue = (sum(map(ord, self.name)) * 47) % 360
        p.setPen(Qt.NoPen)
        p.setBrush(QColor.fromHsv(hue, 150, 200))
        p.drawEllipse(self.rect())
        p.setPen(QColor("white"))
        p.setFont(QFont("Segoe UI", 13, QFont.DemiBold))
        initials = "".join(w[0] for w in self.name.replace("_", " ").split() if w[0].isalnum())[:2].upper() or "?"
        p.drawText(self.rect(), Qt.AlignCenter, initials)


# --------------------------------------------------------------------------- запис з мікрофона
class Recorder(QObject):
    finished = Signal(object)
    tick = Signal(float, float)                                  # залишилось секунд, рівень

    def __init__(self):
        super().__init__()
        self.buffer: deque = deque(maxlen=SR * 30)           # увесь запис
        self.view: deque = deque(maxlen=SR * 3)              # останні 3 с — для осцилографа
        self.recent: deque = deque(maxlen=1600)              # останні 0,1 с — для рівня
        self.stream = None
        self.device = None
        self._timer = QTimer(self, interval=33, timeout=self._poll)
        self._t0 = 0.0
        self.seconds = 5.0

    def start(self, seconds: float):
        import sounddevice as sd

        self.seconds = seconds
        for b in (self.buffer, self.view, self.recent):
            b.clear()

        def callback(data, *_):                                   # потік PortAudio: лише дописування
            chunk = data[:, 0].copy()
            self.buffer.extend(chunk)
            self.view.extend(chunk)
            self.recent.extend(chunk)

        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=self.device,
                                     callback=callback)
        self.stream.start()
        self._t0 = time.perf_counter()
        self._timer.start()

    def stop(self):
        if self.stream is None:
            return
        self._timer.stop()
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.finished.emit(np.array(self.buffer, dtype=np.float32))

    @property
    def active(self) -> bool:
        return self.stream is not None

    def _poll(self):
        left = self.seconds - (time.perf_counter() - self._t0)
        recent = np.fromiter(self.recent, dtype=np.float32, count=len(self.recent))
        self.tick.emit(max(left, 0.0), float(np.abs(recent).max()) if len(recent) else 0.0)
        if left <= 0:
            self.stop()


# --------------------------------------------------------------------------- сторінки
TACTIC_COLORS = {"codes": "bad", "money": "bad", "authority": "warn", "relative": "warn", "secrecy": "warn",
                 "urgency": "warn", "prize": "warn", "remote": "bad", "investment": "warn", "link": "accent"}


def highlight_html(text_: str, keep: set[str] | None = None) -> str:
    """Текст з підсвіченими ознаками шахрайства (HTML для QTextBrowser)."""
    import html

    from . import scam

    out, pos = [], 0
    for a, b, key in scam.spans(text_, keep):
        out.append(html.escape(text_[pos:a]))
        col = QColor(T[TACTIC_COLORS.get(key, "warn")])
        bg = QColor(col)
        bg.setAlpha(70)
        title = scam.TACTICS[key]["name"] if key in scam.TACTICS else "посилання / номер"
        out.append(f'<span style="background-color:{bg.name(QColor.HexArgb)}; color:{col.name()}; font-weight:600;" '
                   f'title="{html.escape(title)}">{html.escape(text_[a:b])}</span>')
        pos = b
    out.append(html.escape(text_[pos:]))
    return "".join(out)


def transcript_html(segments: list[dict], keep: set[str] | None = None) -> str:
    rows = []
    for sg in segments:
        rows.append(f'<p style="margin:0 0 6px 0;"><span style="color:{T["muted"]};">{sg["start"]:5.1f} с</span>&nbsp;&nbsp;'
                    f'{highlight_html(sg["text"], keep)}</p>')
    return "".join(rows) or f'<p style="color:{T["muted"]};">Мовлення не розпізнано.</p>'


def reasons_html(risk_: dict) -> str:
    import html

    parts = []
    if risk_.get("reasons"):
        parts.append("<b>Чому:</b><ul style='margin-top:2px;'>" + "".join(f"<li>{html.escape(r)}</li>" for r in risk_["reasons"])
                     + "</ul>")
    else:
        parts.append(f"<span style='color:{T['muted']};'>Ознак шахрайства не виявлено.</span>")
    if risk_.get("advice"):
        parts.append("<b>Що робити:</b><ul style='margin-top:2px;'>" +
                     "".join(f"<li>{html.escape(a)}</li>" for a in risk_["advice"]) + "</ul>")
    return "".join(parts)


RISK_TITLES = {"високий": ("bad", "Високий ризик шахрайства"), "середній": ("warn", "Середній ризик — будьте обережні"),
               "низький": ("good", "Ознак шахрайства не виявлено")}


def browser(min_h: int = 120) -> QTextBrowser:
    b = QTextBrowser()
    b.setOpenExternalLinks(False)
    b.setMinimumHeight(min_h)
    b.setObjectName("vgBrowser")             # оформлення — у спільній таблиці стилів (змінюється з темою)
    return b


def md_html(t: str) -> str:
    """Простий markdown моделі (**жирний**, списки) → HTML."""
    import html
    import re as _re

    lines = []
    for ln in html.escape(t).split(chr(10)):
        ln = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ln)
        st = ln.lstrip()
        if st[:2] in ("- ", "* ", "• ") or _re.match(r"\d+[.)] ", st):
            ln = "&nbsp;&nbsp;• " + _re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", st)
        lines.append(ln)
    return "<br>".join(lines)


class AIBox(QFrame):
    """Картка «ШІ-аналітик»: кнопка «Пояснити», відповідь виводиться поступово."""

    def __init__(self, win: "MainWindow", facts_fn):
        super().__init__()
        self.setObjectName("card")
        self.win, self.facts_fn, self.text = win, facts_fn, ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        head = QHBoxLayout()
        head.addWidget(text("Висновок ШІ-аналітика", "h2"))
        head.addStretch(1)
        self.btn = QPushButton("✨  Пояснити")
        self.btn.clicked.connect(self.run)
        head.addWidget(self.btn)
        lay.addLayout(head)
        self.note = text("Локальна мовна модель пояснює висновок простими словами. Оцінку ризику вона не змінює; "
                         "дані не залишають комп’ютер.", "muted")
        lay.addWidget(self.note)
        self.out = browser(90)
        lay.addWidget(self.out)

    def reset(self):
        self.text = ""
        self.out.setHtml("")

    def run(self):
        from . import ai

        facts = self.facts_fn()
        if facts is None:
            self.win.toast.show_msg("Спершу виконайте перевірку.", "warn")
            return
        model = self.win.settings["ai_model"]
        ok, why = ai.available(model)
        if not ok:
            self.out.setHtml(f"<span style='color:{T['warn']};'>{why}</span>")
            return
        self.reset()
        self.btn.setEnabled(False)
        self.out.setHtml(f"<span style='color:{T['muted']};'>Модель думає… (на процесорі це 20–60 с)</span>")
        t0 = time.perf_counter()

        def work(progress):
            return ai.explain(facts, on_token=progress, model=model)

        def token(tok: str):
            self.text += tok
            self.out.setHtml(md_html(self.text))
            self.out.verticalScrollBar().setValue(self.out.verticalScrollBar().maximum())

        def done(_):
            self.btn.setEnabled(True)
            self.note.setText(f"Модель {model}, {time.perf_counter() - t0:.0f} с. Оцінку ризику визначили "
                              "вимірювальні модулі; ШІ лише пояснює.")

        t = Task(work, with_progress=True)
        t.signals.progress.connect(token)
        t.signals.done.connect(done)
        t.signals.failed.connect(lambda e: (self.btn.setEnabled(True),
                                            self.out.setHtml(f"<span style='color:{T['bad']};'>Помилка: {e}</span>")))
        self.win.pool.start(t)


class CheckPage(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.signal: np.ndarray | None = None
        self.source = ""
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        head = QHBoxLayout()
        head.addWidget(text("Перевірка голосу", "h1"))
        head.addStretch(1)
        root.addLayout(head)

        # панель керування
        self.speaker = QComboBox()
        self.speaker.setMinimumWidth(220)
        self.chan = QComboBox()
        self.chan.addItems(list(CHANNEL_CHOICES))
        self.chan.currentIndexChanged.connect(self._preview_channel)
        self.rec = RecordButton(52)
        self.rec.clicked.connect(self._toggle_record)
        self.btn_open = QPushButton(f"{ICONS['open']}   Відкрити файл")
        self.btn_play = QPushButton(f"{ICONS['play']}   Відтворити")
        self.btn_check = QPushButton("Перевірити")
        self.btn_check.setObjectName("primary")
        self.btn_check.setMinimumWidth(150)
        self.btn_open.clicked.connect(self._open)
        self.btn_play.clicked.connect(self._play)
        self.btn_check.clicked.connect(self._check)
        self.level = LevelMeter()
        self.src = text("Запис не вибрано", "muted", wrap=False)
        controls = QGridLayout()
        controls.setHorizontalSpacing(12)
        controls.addWidget(self.rec, 0, 0, 3, 1)
        controls.addWidget(text("Мовець", "muted"), 0, 1)
        controls.addWidget(text("Канал зв’язку (імітація)", "muted"), 0, 2)
        controls.addWidget(self.speaker, 1, 1)
        controls.addWidget(self.chan, 1, 2)
        controls.addWidget(self.btn_open, 1, 3)
        controls.addWidget(self.btn_play, 1, 4)
        controls.addWidget(self.btn_check, 1, 5)
        controls.addWidget(self.src, 2, 1, 1, 2)
        controls.addWidget(self.level, 2, 3, 1, 3)
        controls.setColumnStretch(2, 1)
        root.addWidget(card(controls, margins=14))

        body = QHBoxLayout()
        body.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(14)
        self.verdict = VerdictCard()
        left.addWidget(self.verdict)
        gauges = QHBoxLayout()
        self.g_risk = Gauge("Ризик шахрайства", 128, risk=True)
        self.g_cm = Gauge("Справжність голосу", 128)
        self.g_asv = Gauge("Збіг із мовцем", 128)
        gauges.addStretch(1)
        for g_ in (self.g_risk, self.g_cm, self.g_asv):
            gauges.addWidget(g_)
        gauges.addStretch(1)
        left.addWidget(card(gauges, margins=8))
        self.why = browser(110)
        left.addWidget(card(text("Чому такий висновок", "h2"), self.why, margins=14))
        self.last = None
        self.ai = AIBox(win, lambda: None if self.last is None else __import__(
            "voiceguard.ai", fromlist=["ai"]).facts_from_verdict(self.last))
        left.addWidget(self.ai)
        self.said = browser(130)
        left.addWidget(card(text("Що сказано — підозрілі фрази підсвічено", "h2"), self.said, margins=14))
        self.steps = Steps()
        left.addWidget(card(text("Хід перевірки", "h2"), self.steps, margins=14))
        self.details = QGridLayout()
        self.details.setHorizontalSpacing(18)
        self.details.setVerticalSpacing(6)
        det_card = card(text("Подробиці", "h2"), self.details, margins=14)
        left.addWidget(det_card)
        self.notes = text("", "muted")
        self.notes.setStyleSheet(f"color: {T['warn']};")
        left.addWidget(self.notes)
        left.addStretch(1)
        lw = QWidget()
        lw.setLayout(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(lw)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(450)
        body.addWidget(scroll)

        right = QVBoxLayout()
        right.setSpacing(14)
        self.wave = WaveView(64)
        self.spec = SpectrogramView()
        self.timeline = TimelineView()
        right.addWidget(card(text("Сигнал", "h2"), self.wave, margins=12, expand=self.wave), 2)
        right.addWidget(card(text("Спектрограма — що «чує» система", "h2"), self.spec, margins=12,
                             expand=self.spec), 3)
        tl_title = text("Підозрілість у часі — де детектор бачить ознаки підробки  ⓘ", "h2")
        tl_title.setToolTip("Запис оцінюється вікнами по 3 с з кроком 0,5 с.\nВище пунктиру — схоже на справжній "
                            "голос, нижче — ознаки синтезу.\nСірі смуги — майже без мовлення. Наведіть мишу на графік.")
        self.timeline.setToolTip("")
        right.addWidget(card(tl_title, self.timeline, margins=12, expand=self.timeline), 3)
        self._play_timer = QTimer(self, interval=30, timeout=self._tick_play)
        self._play_t0, self._play_dur = 0.0, 0.0
        body.addLayout(right, 1)
        root.addLayout(body, 1)
        self._fill_details(None)

    # дані
    def set_enabled(self, on: bool):
        for w in (self.btn_open, self.btn_play, self.btn_check, self.rec, self.speaker, self.chan):
            w.setEnabled(on)

    def refresh_speakers(self, names: list[str]):
        cur = self.speaker.currentText()
        self.speaker.clear()
        self.speaker.addItem("Не перевіряти мовця")
        self.speaker.addItems(names)
        if cur in names:
            self.speaker.setCurrentText(cur)

    def _chan_key(self) -> str:
        return CHANNEL_CHOICES[self.chan.currentText()]

    def _set_signal(self, x: np.ndarray, source: str):
        self.signal, self.source = x.astype(np.float32), source
        self.src.setText(f"{source} · {len(x) / SR:.1f} с")
        self.verdict.reset()
        self.verdict.title.setText("Запис готовий")
        self.verdict.sub.setText("Натисніть «Перевірити». Можна вибрати канал зв’язку, щоб побачити, як він змінює голос.")
        self.g_cm.set_value(None)
        self.g_asv.set_value(None)
        self.g_risk.set_value(None)
        self.why.setHtml("")
        self.said.setHtml("")
        self.last = None
        self.ai.reset()
        self.timeline.set_timeline(None)
        self.steps.reset(self.speaker.currentIndex() > 0, self.win.settings["transcribe"])
        self._fill_details(None)
        self._preview_channel()

    def _preview_channel(self):
        if self.signal is None:
            return
        key = self._chan_key()
        task = Task(channel.CHANNELS[key], self.signal)
        task.signals.done.connect(lambda y: (self.wave.set_signal(y), self.spec.set_signal(y)))
        self.win.pool.start(task)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Аудіофайл", "", "Аудіо (*.wav *.flac *.ogg *.mp3);;Усі (*.*)")
        if path:
            t = Task(audio.load, path)
            t.signals.done.connect(lambda x: self._set_signal(x, Path(path).name))
            t.signals.failed.connect(lambda e: self.win.toast.show_msg(f"Не вдалося прочитати файл: {e}", "bad"))
            self.win.pool.start(t)

    def _toggle_record(self):
        r = self.win.recorder
        if r.active:
            r.stop()
            return
        try:
            r.finished.disconnect()
            r.tick.disconnect()
        except (RuntimeError, TypeError):
            pass
        r.finished.connect(self._recorded)
        r.tick.connect(self._rec_tick)
        try:
            r.start(self.win.settings["record_seconds"])
        except Exception as e:                                   # noqa: BLE001
            self.win.toast.show_msg(f"Мікрофон недоступний: {e}", "bad")
            return
        self.rec.set_recording(True)
        self.wave.start_live(r.view)
        self.verdict.reset()
        self.verdict.title.setText("Запис…")
        self.verdict.sub.setText("Говоріть звичайним голосом. Натисніть кнопку ще раз, щоб зупинити раніше.")

    def _rec_tick(self, left: float, level: float):
        self.rec.remaining = left
        self.rec.update()
        self.level.set_level(level)
        self.wave.update()

    def _recorded(self, x: np.ndarray):
        self.rec.set_recording(False)
        self.level.set_level(0.0)
        if len(x) < SR // 2:
            self.win.toast.show_msg("Запис закороткий.", "warn")
            return
        self._set_signal(x, "мікрофон")

    def _play(self):
        if self.signal is None:
            return
        import sounddevice as sd

        task = Task(channel.CHANNELS[self._chan_key()], self.signal)

        def go(y):
            sd.stop()
            sd.play(audio.normalize(y, 0.8), SR)
            self._play_t0, self._play_dur = time.perf_counter(), len(y) / SR
            self._play_timer.start()

        task.signals.done.connect(go)
        self.win.pool.start(task)

    def _tick_play(self):
        """Курсор відтворення біжить по сигналу, спектрограмі та графіку підозрілості."""
        frac = (time.perf_counter() - self._play_t0) / max(self._play_dur, 1e-9)
        if frac >= 1.0:
            self._play_timer.stop()
            frac = None
        for w in (self.wave, self.spec, self.timeline):
            w.playhead = frac
            w.update()

    def _check(self):
        if self.signal is None:
            self.win.toast.show_msg("Спершу запишіть фразу або відкрийте файл.", "warn")
            return
        spk = self.speaker.currentText() if self.speaker.currentIndex() > 0 else None
        key, chan_name = self._chan_key(), self.chan.currentText()
        tr, lang = self.win.settings["transcribe"], self.win.settings["language"]
        self.steps.reset(spk is not None, tr)
        self.set_enabled(False)
        vg = self.win.vg

        def work(progress):
            y = channel.CHANNELS[key](self.signal)
            return y, vg.check(y, spk, source=f"{self.source} · {chan_name}", progress=progress,
                               transcribe=tr, language=lang)

        t = Task(work, with_progress=True)
        t.signals.progress.connect(self.steps.advance)
        t.signals.done.connect(self._show)
        t.signals.failed.connect(lambda e: (self.set_enabled(True), self.win.toast.show_msg(f"Помилка: {e}", "bad")))
        self.win.pool.start(t)

    def _show(self, res):
        y, v = res
        self.set_enabled(True)
        self.steps.times = v.timings
        self.steps.advance("done")
        self.wave.set_signal(y)
        self.spec.set_signal(y)
        self.timeline.set_timeline(v.timeline)
        if not v.cm_genuine:
            voice = "Голос має ознаки синтезу або перетворення."
        elif v.speaker and not v.asv_match:
            voice = f"Голос справжній, але не збігається з «{v.speaker}»."
        elif v.speaker:
            voice = f"Справжній голос мовця «{v.speaker}»."
        else:
            voice = "Голос без ознак підробки."
        self.last = v
        self.ai.reset()
        r = v.risk
        kind, title = RISK_TITLES[r["level"]]
        # підзаголовок: головна ознака змісту (назва схеми, якщо впізнано) + стан голосу — без повторів
        content = v.content or {}
        scheme = (content.get("combos") or list((content.get("tactics") or {}).values()) or [""])[0]
        self.verdict.show_verdict(kind, f"{title} · {100 * r['score']:.0f} %",
                                  (f"{scheme}. " if scheme else "") + voice)
        self.g_risk.set_value(r["score"], f"рівень: {r['level']}")
        self.why.setHtml(reasons_html(r))
        if v.transcript:
            links = v.content.get("links", [])
            extra = "".join(f"<p style='margin:6px 0 0 0;'>🔗 {l['value']} — ризик {100 * l['risk']:.0f} %</p>"
                            for l in links if l["kind"] == "url")
            self.said.setHtml(transcript_html(v.transcript, set(v.content.get("tactics", {}))) + extra)
        else:
            self.said.setHtml(f"<span style='color:{T['muted']};'>Розпізнавання мовлення вимкнено в налаштуваннях.</span>")
        self.g_cm.set_value(v.cm_confidence, f"оцінка {v.cm_score:+.2f}\nпоріг {v.cm_threshold:+.2f}")
        if v.speaker:
            rel = 1.0 / (1.0 + math.exp(-(v.asv_score - v.asv_threshold) / 0.3))
            self.g_asv.set_value(rel, f"оцінка {v.asv_score:+.2f}\nпоріг {v.asv_threshold:+.2f}")
        else:
            self.g_asv.set_value(None, "мовця не вибрано")
        self._fill_details(v)
        self.notes.setText("\n".join("⚠ " + n for n in v.notes))
        self.win.toast.show_msg(f"Перевірено за {v.seconds:.1f} с", kind)

    def _fill_details(self, v):
        while self.details.count():
            w = self.details.takeAt(0).widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        if v is None:
            rows = [("Тривалість", "—"), ("Мовлення", "—"), ("Смуга частот", "—"), ("Сигнал / шум", "—"),
                    ("Детектор", self.win.vg.detector.name if self.win.vg else "завантаження…")]
        else:
            a = v.analysis
            band = f"{a['bandwidth_hz'] / 1000:.1f} кГц · " + ("вузькосмуговий (телефон / мобільний)"
                                                                if a["narrowband"] else "широкосмуговий")
            rows = [("Тривалість", f"{a['duration']:.1f} с"), ("Мовлення", f"{a['speech']:.1f} с"),
                    ("Смуга частот", band), ("Сигнал / шум", f"≈ {a['snr_db']:.0f} дБ"),
                    ("Детектор", v.detector), ("Час обробки", f"{v.seconds:.1f} с")]
            if v.language:
                rows.insert(2, ("Мова", {"uk": "українська", "ru": "російська", "en": "англійська"}.get(v.language, v.language)))
        for i, (k, val) in enumerate(rows):
            self.details.addWidget(text(k, "muted", wrap=False), i, 0, Qt.AlignTop)
            self.details.addWidget(text(val), i, 1)
        self.details.setColumnStretch(1, 1)


class SpeakersPage(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.clips: list[np.ndarray] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        root.addWidget(text("Мовці", "h1"))
        body = QHBoxLayout()
        body.setSpacing(14)

        self.list = QVBoxLayout()
        self.list.setSpacing(8)
        holder = QWidget()
        holder.setLayout(self.list)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        body.addWidget(card(text("Зареєстровані", "h2"), scroll, margins=14, expand=scroll), 1)

        self.name = QLineEdit()
        self.name.setPlaceholderText("Ім’я мовця")
        self.rec = RecordButton(52)
        self.rec.clicked.connect(self._toggle)
        self.wave = WaveView(64)
        self.wave.placeholder = "Натисніть мікрофон"
        self.level = LevelMeter()
        self.progress = Gauge("Мовлення для моделі", 150)
        self.chips = QListWidget()
        self.chips.setMaximumHeight(120)
        add_files = QPushButton(f"{ICONS['add']}   Додати файли…")
        add_files.clicked.connect(self._add_files)
        clear = QPushButton("Очистити")
        clear.clicked.connect(self._clear)
        self.btn_enroll = QPushButton("Зареєструвати")
        self.btn_enroll.setObjectName("primary")
        self.btn_enroll.clicked.connect(self._enroll)
        rec_row = QHBoxLayout()
        rec_row.addWidget(self.rec)
        col = QVBoxLayout()
        col.addWidget(self.wave)
        col.addWidget(self.level)
        rec_row.addLayout(col, 1)
        top = QHBoxLayout()
        top.addLayout(rec_row, 1)
        top.addWidget(self.progress)
        btns = QHBoxLayout()
        btns.addWidget(add_files)
        btns.addWidget(clear)
        btns.addStretch(1)
        btns.addWidget(self.btn_enroll)
        hint = text("Запишіть 3–5 фраз по 4–6 с — разом щонайменше 15 с мовлення. Записуйте лише себе або людей, "
                    "які дали на це згоду.", "muted")
        body.addWidget(card(text("Новий мовець", "h2"), hint, self.name, top, text("Фрази", "muted"), self.chips, btns,
                            margins=16), 1)
        root.addLayout(body, 1)
        self._update_progress()

    def refresh(self, names: list[str]):
        while self.list.count():
            it = self.list.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if not names:
            self.list.addWidget(text("Ще немає жодного мовця. Зареєструйте першого праворуч.", "muted"))
        for n in names:
            info = self.win.vg.verifier.info(n)
            row = QFrame()
            row.setObjectName("card")
            lay = QHBoxLayout(row)
            lay.setContentsMargins(12, 10, 12, 10)
            lay.addWidget(Avatar(n))
            col = QVBoxLayout()
            col.addWidget(text(n, "h2", wrap=False))
            col.addWidget(text(f"{info['n_recordings']} фраз · {info['speech_seconds']:.0f} с мовлення · "
                               f"{time.strftime('%d.%m.%Y', time.localtime(info['created']))}", "muted", wrap=False))
            lay.addLayout(col, 1)
            de = QPushButton(ICONS["delete"])
            de.setFont(QFont(ICON_FONT, 11))
            de.setToolTip("Видалити")
            de.setFixedWidth(40)
            de.clicked.connect(lambda _=False, name=n: self._delete(name))
            lay.addWidget(de)
            self.list.addWidget(row)
        self.list.addStretch(1)

    def _toggle(self):
        r = self.win.recorder
        if r.active:
            r.stop()
            return
        try:
            r.finished.disconnect()
            r.tick.disconnect()
        except (RuntimeError, TypeError):
            pass
        r.finished.connect(self._recorded)
        r.tick.connect(lambda left, lev: (setattr(self.rec, "remaining", left), self.rec.update(),
                                          self.level.set_level(lev), self.wave.update()))
        try:
            r.start(self.win.settings["record_seconds"])
        except Exception as e:                                   # noqa: BLE001
            self.win.toast.show_msg(f"Мікрофон недоступний: {e}", "bad")
            return
        self.rec.set_recording(True)
        self.wave.start_live(r.view)

    def _recorded(self, x):
        self.rec.set_recording(False)
        self.level.set_level(0)
        self.wave.set_signal(x)
        self._add([x], "мікрофон")

    def _add(self, xs, source):
        for x in xs:
            self.clips.append(x.astype(np.float32))
            QListWidgetItem(f"Фраза {len(self.clips)} · {source} · {len(x) / SR:.1f} с", self.chips)
        self._update_progress()

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Фрази мовця", "", "Аудіо (*.wav *.flac *.ogg *.mp3)")
        if paths:
            t = Task(lambda: [audio.load(p) for p in paths])
            t.signals.done.connect(lambda xs: self._add(xs, "файл"))
            self.win.pool.start(t)

    def _clear(self):
        self.clips = []
        self.chips.clear()
        self.wave.set_signal(None)
        self._update_progress()

    def _update_progress(self):
        total = sum(len(x) for x in self.clips) / SR
        self.progress.set_value(min(total / 15.0, 1.0) * 0.999 if total else None, f"{total:.0f} / 15 с")

    def _enroll(self):
        name = self.name.text().strip()
        if not name or any(c in name for c in '\\/:*?"<>|'):
            self.win.toast.show_msg("Введіть ім’я без символів \\ / : * ? \" < > |", "warn")
            return
        if not self.clips:
            self.win.toast.show_msg("Додайте хоча б одну фразу.", "warn")
            return
        t = Task(self.win.vg.verifier.enroll, name, list(self.clips))

        def done(speech):
            self._clear()
            self.name.clear()
            self.win.speakers_changed()
            note = "" if speech >= 15 else " Мовлення замало — точність буде нижчою."
            self.win.toast.show_msg(f"Мовця «{name}» зареєстровано ({speech:.0f} с мовлення).{note}",
                                    "good" if speech >= 15 else "warn")

        t.signals.done.connect(done)
        self.win.pool.start(t)

    def _delete(self, name):
        if QMessageBox.question(self, "VoiceGuard", f"Видалити мовця «{name}»?") == QMessageBox.Yes:
            self.win.vg.verifier.delete(name)
            self.win.speakers_changed()
            self.win.toast.show_msg(f"Мовця «{name}» видалено.")


def verdict_of(v: dict) -> tuple[str, str]:
    if v.get("text_only"):
        lvl = v["risk"]["level"]
        return {"високий": ("Текст: високий ризик", "bad"), "середній": ("Текст: середній ризик", "warn"),
                "низький": ("Текст: без ознак", "good")}[lvl]
    if v.get("risk") and v["risk"].get("level") in ("високий", "середній") and v.get("cm_genuine") \
            and not (v.get("speaker") and not v.get("asv_match")):
        return ("Шахрайський зміст", "bad") if v["risk"]["level"] == "високий" else ("Підозрілий зміст", "warn")
    if not v["cm_genuine"]:
        return "Підробка", "bad"
    if v["speaker"] and not v["asv_match"]:
        return "Інший мовець", "bad"
    return ("Прийнято", "good") if v["speaker"] else ("Справжній", "good")


class _BatchSignals(QObject):
    row = Signal(object)
    finished = Signal()


class BatchTask(QRunnable):
    """Перевіряє файли по одному й надсилає результат кожного рядком таблиці."""

    def __init__(self, vg, paths: list[str]):
        super().__init__()
        self.setAutoDelete(False)
        self.vg, self.paths, self.stop = vg, paths, False
        self.signals = _BatchSignals()

    def run(self):
        for path in self.paths:
            if self.stop:
                break
            try:
                x = audio.load(path)
                v = self.vg.check(x, None, source=f"пакет · {Path(path).name}")
                self.signals.row.emit((path, v, None))
            except Exception as e:                               # noqa: BLE001
                self.signals.row.emit((path, None, str(e)))
        self.signals.finished.emit()


class Progress(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(10)
        self.value, self._shown, self._anim = 0.0, 0.0, None

    def set_value(self, v: float):
        self._anim = anim(self, self._shown, v, 350, self._step)
        self._anim.start()

    def _step(self, v):
        self._shown = float(v)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(qc("track"))
        p.drawRoundedRect(self.rect(), 5, 5)
        p.setBrush(qc("accent"))
        p.drawRoundedRect(QRectF(0, 0, self.width() * self._shown, self.height()), 5, 5)


class BatchPage(QWidget):
    """Пакетна перевірка теки з записами (наприклад, записів дзвінків) і зведений звіт."""

    EXT = (".wav", ".flac", ".ogg", ".mp3")

    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.task: BatchTask | None = None
        self.rows: list = []
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        head = QHBoxLayout()
        head.addWidget(text("Пакетна перевірка", "h1"))
        head.addStretch(1)
        self.btn_folder = QPushButton(f"{ICONS['batch']}   Вибрати теку…")
        self.btn_folder.setObjectName("primary")
        self.btn_folder.clicked.connect(self._choose)
        self.btn_stop = QPushButton(f"{ICONS['stop']}   Зупинити")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        export = QPushButton(f"{ICONS['export']}   Експорт CSV")
        export.clicked.connect(self._export)
        for w in (self.btn_folder, self.btn_stop, export):
            head.addWidget(w)
        root.addLayout(head)
        tiles = QHBoxLayout()
        self.t_all, self.t_ok = StatTile("файлів перевірено", "accent"), StatTile("справжній голос", "good")
        self.t_fake, self.t_part = StatTile("підробка", "bad"), StatTile("підозрілі фрагменти", "warn")
        for t in (self.t_all, self.t_ok, self.t_fake, self.t_part):
            tiles.addWidget(t)
        root.addLayout(tiles)
        self.status = text("Виберіть теку з аудіофайлами (WAV, FLAC, OGG, MP3). Перевіряються лише файли в самій теці.",
                           "muted")
        self.progress = Progress()
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Файл", "Тривалість", "Рішення", "Справжність",
                                              "Найпідозріліший момент", "Смуга"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 6):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        root.addWidget(card(self.status, self.progress, self.table, margins=14, expand=self.table), 1)

    def _choose(self):
        if not self.win.vg:
            return
        folder = QFileDialog.getExistingDirectory(self, "Тека з записами")
        if not folder:
            return
        paths = sorted(str(f) for f in Path(folder).iterdir() if f.suffix.lower() in self.EXT)
        if not paths:
            self.win.toast.show_msg("У теці немає аудіофайлів.", "warn")
            return
        self.start(paths)

    def start(self, paths: list[str]):
        self.rows = []
        self.table.setRowCount(0)
        self.total = len(paths)
        for t in (self.t_all, self.t_ok, self.t_fake, self.t_part):
            t.value.setText("0")
        self.progress.set_value(0)
        self.task = BatchTask(self.win.vg, paths)
        self.task.signals.row.connect(self._row)
        self.task.signals.finished.connect(self._finished)
        self.btn_folder.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText(f"Перевірка {self.total} файлів…")
        self.win.pool.start(self.task)

    def _stop(self):
        if self.task:
            self.task.stop = True

    def _row(self, res):
        path, v, err = res
        self.rows.append(res)
        i = self.table.rowCount()
        self.table.insertRow(i)
        if err:
            cells, kind = [Path(path).name, "—", "помилка", err[:60], "—", "—"], "warn"
        else:
            tl = v.timeline
            idx = [k for k, sp in enumerate(tl["speech"]) if sp] or list(range(len(tl["t"])))
            worst = min(idx, key=lambda k: tl["confidence"][k]) if idx else None
            partial = v.cm_genuine and worst is not None and tl["confidence"][worst] < 0.5
            label, kind = (("Справжній, є підозрілий фрагмент", "warn") if partial else ("Справжній", "good")) \
                if v.cm_genuine else ("Підробка", "bad")
            cells = [Path(path).name, f"{v.duration:.1f} с", label, f"{100 * v.cm_confidence:.0f} %",
                     "—" if worst is None else f"{tl['t'][worst]:.1f} с ({100 * tl['confidence'][worst]:.0f} %)",
                     f"{v.analysis['bandwidth_hz'] / 1000:.1f} кГц"]
        for j, c in enumerate(cells):
            it = QTableWidgetItem(c)
            if j == 2:
                it.setForeground(qc(kind))
                it.setFont(ui_font(9, QFont.DemiBold))
            self.table.setItem(i, j, it)
        self.table.setRowHeight(i, 32)
        self.table.scrollToBottom()
        ok = [r for r in self.rows if r[1] is not None]
        self.t_all.value.setText(str(len(self.rows)))
        self.t_ok.value.setText(str(sum(r[1].cm_genuine for r in ok)))
        self.t_fake.value.setText(str(sum(not r[1].cm_genuine for r in ok)))
        self.t_part.value.setText(str(sum(1 for r in ok if r[1].cm_genuine and any(
            c < 0.5 for c, sp in zip(r[1].timeline["confidence"], r[1].timeline["speech"]) if sp))))
        self.progress.set_value(len(self.rows) / max(self.total, 1))
        self.status.setText(f"Перевірено {len(self.rows)} з {self.total}…")

    def _finished(self):
        self.btn_folder.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText(f"Готово: перевірено {len(self.rows)} файлів. Деталі кожного — у «Журналі».")
        self.win.toast.show_msg(f"Пакетну перевірку завершено ({len(self.rows)} файлів).", "good")
        self.task = None

    def _export(self):
        if not self.rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Звіт пакетної перевірки", "voiceguard_batch.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["файл", "тривалість, с", "справжній", "впевненість", "мін. впевненість у часі",
                        "момент, с", "смуга, Гц", "помилка"])
            for p_, v, err in self.rows:
                if err:
                    w.writerow([p_, "", "", "", "", "", "", err])
                    continue
                tl = v.timeline
                k = int(np.argmin(tl["confidence"])) if tl["confidence"] else None
                w.writerow([p_, f"{v.duration:.2f}", int(v.cm_genuine), f"{v.cm_confidence:.3f}",
                            "" if k is None else f"{tl['confidence'][k]:.3f}", "" if k is None else tl["t"][k],
                            f"{v.analysis['bandwidth_hz']:.0f}", ""])
        self.win.toast.show_msg(f"Збережено: {Path(path).name}", "good")


class TextPage(QWidget):
    """Перевірка повідомлення без голосу: SMS, Telegram, Viber, email."""

    EXAMPLES = [
        ("Виплата «від Дії»", "Вітаємо! Вам нараховано одноразову виплату 6500 грн від держави. Щоб отримати кошти, перейдіть "
                              "на diia-vyplata.xyz/form і введіть номер картки, термін дії та CVV. Пропозиція діє лише сьогодні!"),
        ("«Служба безпеки банку»", "Шановний клієнте ПриватБанку! По вашій картці зафіксовано підозрілу операцію. Терміново "
                                   "зателефонуйте в службу безпеки за номером +380 93 123 45 67 і продиктуйте код з SMS."),
        ("Звичайне повідомлення", "Привіт! Я буду вдома о сьомій, купи, будь ласка, хліба та молока. Ключі у сусідки."),
    ]

    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        root.addWidget(text("Перевірка повідомлення", "h1"))
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Вставте текст SMS, повідомлення з Telegram чи Viber, листа…")
        self.input.setMinimumHeight(70)
        self.input.setMaximumHeight(110)
        self.input.setObjectName("vgBrowser")         # той самий стиль, що й поля з текстом (змінюється з темою)
        btns = QHBoxLayout()
        for name, sample in self.EXAMPLES:
            b = QPushButton(f"Приклад: {name}")
            b.clicked.connect(lambda _=False, t=sample: self.input.setPlainText(t))
            btns.addWidget(b)
        btns.addStretch(1)
        go = QPushButton("Перевірити")
        go.setObjectName("primary")
        go.setMinimumWidth(150)
        go.clicked.connect(self._check)
        btns.addWidget(go)
        root.addWidget(card(self.input, btns, margins=14))
        body = QHBoxLayout()
        body.setSpacing(14)
        left = QVBoxLayout()
        self.verdict = VerdictCard()
        self.verdict.title.setText("Вставте текст і натисніть «Перевірити»")
        self.verdict.sub.setText("Перевіряються тактики шахраїв, посилання, номери карток і телефонів.")
        left.addWidget(self.verdict)
        self.g = Gauge("Ризик шахрайства", 116, risk=True)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.g)
        row.addStretch(1)
        left.addWidget(card(row, margins=10))
        self.last = None
        self.ai = AIBox(win, lambda: None if self.last is None else __import__(
            "voiceguard.ai", fromlist=["ai"]).facts_from_text(self.last))
        left.addWidget(self.ai, 1)
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(450)
        body.addWidget(lw)
        right = QVBoxLayout()
        self.why = browser(140)
        self.marked = browser(140)
        right.addWidget(card(text("Чому такий висновок", "h2"), self.why, margins=14, expand=self.why), 1)
        right.addWidget(card(text("Текст — підозрілі фрагменти підсвічено", "h2"), self.marked, margins=14,
                             expand=self.marked), 1)
        body.addLayout(right, 1)
        root.addLayout(body, 1)

    def _check(self):
        t = self.input.toPlainText().strip()
        if not t or not self.win.vg:
            self.win.toast.show_msg("Вставте текст повідомлення.", "warn")
            return
        res = self.win.vg.check_text(t, source="повідомлення")
        self.last = res
        self.ai.reset()
        r = res["risk"]
        kind, title = RISK_TITLES[r["level"]]
        self.verdict.show_verdict(kind, f"{title} · {100 * r['score']:.0f} %",
                                  r["reasons"][0] if r["reasons"] else "Тактик шахраїв і небезпечних посилань не знайдено.")
        self.g.set_value(r["score"], f"рівень: {r['level']}")
        self.why.setHtml(reasons_html(r))
        self.marked.setHtml(highlight_html(t, set(res["content"]["tactics"])).replace(chr(10), "<br>"))


class JournalPage(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        head = QHBoxLayout()
        head.addWidget(text("Журнал перевірок", "h1"))
        head.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Пошук за записом, мовцем…")
        self.search.setFixedWidth(280)
        self.search.textChanged.connect(self.refresh)
        self.filter = QComboBox()
        self.filter.addItems(["Усі рішення", "Прийнято / справжній", "Підробка", "Інший мовець"])
        self.filter.currentIndexChanged.connect(self.refresh)
        export = QPushButton(f"{ICONS['export']}   Експорт CSV")
        export.clicked.connect(self._export)
        for w in (self.search, self.filter, export):
            head.addWidget(w)
        root.addLayout(head)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Час", "Запис · канал", "Мовець", "Рішення", "Справжність",
                                              "Збіг", "Смуга"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in (0, 2, 3, 4, 5, 6):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        root.addWidget(card(self.table, margins=6, expand=self.table), 1)
        self.rows = []

    def refresh(self):
        if not self.win.vg:
            return
        q, f = self.search.text().lower(), self.filter.currentIndex()
        self.rows = []
        for t, src, v in self.win.vg.journal(1000):
            label, kind = verdict_of(v)
            if f == 1 and kind != "good" or f == 2 and label != "Підробка" or f == 3 and label != "Інший мовець":
                continue
            if q and q not in (src + " " + (v["speaker"] or "")).lower():
                continue
            self.rows.append((t, src, v, label, kind))
        self.table.setRowCount(len(self.rows))
        for i, (t, src, v, label, kind) in enumerate(self.rows):
            if v.get("text_only"):
                cells = [time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(t)), src, "—", label,
                         f"ризик {100 * v['risk']['score']:.0f} %", "—", "—"]
                for j, c in enumerate(cells):
                    it = QTableWidgetItem(c)
                    if j == 3:
                        it.setForeground(qc(kind))
                        it.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
                    self.table.setItem(i, j, it)
                self.table.setRowHeight(i, 34)
                continue
            band = v.get("analysis", {}).get("bandwidth_hz")
            cells = [time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(t)), src, v["speaker"] or "—", label,
                     f"{100 * v['cm_confidence']:.0f} %", "—" if v["asv_score"] is None else f"{v['asv_score']:+.2f}",
                     "—" if band is None else f"{band / 1000:.1f} кГц"]
            for j, c in enumerate(cells):
                it = QTableWidgetItem(c)
                if j == 3:
                    it.setForeground(qc(kind))
                    it.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
                self.table.setItem(i, j, it)
            self.table.setRowHeight(i, 34)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Експорт журналу", "voiceguard_journal.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["час", "запис", "мовець", "рішення", "оцінка детектора", "поріг детектора", "оцінка мовця",
                        "поріг мовця", "смуга, Гц", "мовлення, с"])
            for t, src, v, label, _ in self.rows:
                a = v.get("analysis", {})
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)), src, v["speaker"] or "", label,
                            f"{v['cm_score']:.4f}", f"{v['cm_threshold']:.4f}",
                            "" if v["asv_score"] is None else f"{v['asv_score']:.4f}",
                            "" if v["asv_threshold"] is None else f"{v['asv_threshold']:.4f}",
                            f"{a.get('bandwidth_hz', 0):.0f}", f"{a.get('speech', 0):.2f}"])
        self.win.toast.show_msg(f"Збережено: {Path(path).name}", "good")


class StatsPage(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        root.addWidget(text("Статистика і якість моделей", "h1"))
        tiles = QHBoxLayout()
        self.t_all, self.t_ok = StatTile("усього перевірок", "accent"), StatTile("прийнято / справжній", "good")
        self.t_fake, self.t_other = StatTile("виявлено підробок", "bad"), StatTile("інший мовець", "warn")
        for t in (self.t_all, self.t_ok, self.t_fake, self.t_other):
            tiles.addWidget(t)
        root.addLayout(tiles)
        body = QHBoxLayout()
        body.setSpacing(14)
        self.eer = BarChart("%")
        self.eer_note = text("", "muted")
        body.addWidget(card(text("Помилка детектора (EER) за каналами зв’язку", "h2"), self.eer_note, self.eer,
                            margins=16), 3)
        self.days = BarChart("")
        body.addWidget(card(text("Перевірки за останні 7 днів", "h2"), self.days, margins=16), 2)
        root.addLayout(body)
        root.addStretch(1)

    def refresh(self):
        if not self.win.vg:
            return
        rows = self.win.vg.journal(100000)
        labels = [verdict_of(v)[0] for _, _, v in rows]
        self.t_all.set_count(len(rows))
        self.t_ok.set_count(sum(l in ("Прийнято", "Справжній") for l in labels))
        self.t_fake.set_count(labels.count("Підробка"))
        self.t_other.set_count(labels.count("Інший мовець"))
        today = time.time() // 86400
        counts = [sum(1 for t, _, _ in rows if t // 86400 == today - d) for d in range(6, -1, -1)]
        wd = ["пн", "вт", "ср", "чт", "пт", "сб", "нд"]

        def day(d):
            lt = time.localtime((today - d) * 86400 + 43200)
            return f"{wd[lt.tm_wday]} {lt.tm_mday:02d}.{lt.tm_mon:02d}" + (" (сьогодні)" if d == 0 else "")

        self.days.set_rows([(day(d), float(c), "accent") for d, c in zip(range(6, -1, -1), counts)])
        res = self.win.model_report()
        if res:
            name, eer = res
            nice = {"clean": "чистий запис", "opus32": "Opus 32 кбіт/с", "opus16": "Opus 16 кбіт/с",
                    "opus8": "Opus 8 кбіт/с", "telephone": "телефон G.711", "amrnb": "мобільний AMR-NB",
                    "amrwb": "HD Voice AMR-WB", "gsm": "GSM", "noise20": "шум 20 дБ", "noise10": "шум 10 дБ"}
            self.eer.set_rows([(nice.get(c, c), 100 * v, "good" if v < 0.1 else "warn" if v < 0.3 else "bad")
                               for c, v in eer.items()], vmax=50)
            self.eer_note.setText(f"{name}. Набір eval ASVspoof 2019 LA (нові мовці й частина нових атак). "
                                  "Менше — краще; 50 % — випадкове вгадування.")


class SettingsPage(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        root.addWidget(text("Налаштування", "h1"))
        self.theme = QComboBox()
        self.theme.addItems(["Темна", "Світла"])
        self.theme.currentIndexChanged.connect(lambda i: win.apply_theme("dark" if i == 0 else "light"))
        self.mic = QComboBox()
        self.mic_ids: list[int | None] = []
        self.mic.currentIndexChanged.connect(lambda i: setattr(win.recorder, "device", self.mic_ids[i] if
                                                               0 <= i < len(self.mic_ids) else None))
        self.secs = QSpinBox()
        self.secs.setRange(3, 15)
        self.secs.setValue(5)
        self.secs.setSuffix(" с")
        self.secs.valueChanged.connect(lambda v: win.settings.__setitem__("record_seconds", v))
        g = QGridLayout()
        g.setHorizontalSpacing(16)
        g.addWidget(text("Тема", "muted"), 0, 0)
        g.addWidget(self.theme, 0, 1)
        g.addWidget(text("Мікрофон", "muted"), 1, 0)
        g.addWidget(self.mic, 1, 1)
        g.addWidget(text("Тривалість запису", "muted"), 2, 0)
        g.addWidget(self.secs, 2, 1)
        self.tr = QCheckBox("Розпізнавати мовлення й аналізувати зміст (Whisper; +5–20 с на перевірку)")
        self.tr.setChecked(True)
        self.tr.toggled.connect(lambda on: win.settings.__setitem__("transcribe", on))
        self.lang = QComboBox()
        self.lang.addItems(["Українська", "Визначати автоматично"])
        self.lang.currentIndexChanged.connect(lambda i: win.settings.__setitem__("language", "uk" if i == 0 else None))
        g.addWidget(self.tr, 3, 0, 1, 2)
        g.addWidget(text("Мова розмови", "muted"), 4, 0)
        g.addWidget(self.lang, 4, 1)
        self.model = QLineEdit(win.settings["ai_model"])
        self.model.editingFinished.connect(lambda: win.settings.__setitem__("ai_model", self.model.text().strip()))
        g.addWidget(text("Модель ШІ-аналітика (Ollama)", "muted"), 5, 0)
        g.addWidget(self.model, 5, 1)
        g.setColumnStretch(1, 1)
        root.addWidget(card(text("Інтерфейс і запис", "h2"), g, margins=16))

        self.cm_strict = QComboBox()
        self.cm_strict.addItems(["м'яка", "звичайна", "сувора"])
        self.cm_strict.setCurrentText("звичайна")
        self.cm_strict.currentTextChanged.connect(self._cm_changed)
        self.cm_note = text("", "muted")
        root.addWidget(card(text("Суворість детектора підробки", "h2"), self.cm_strict, self.cm_note, margins=16))
        self.asv = QSlider(Qt.Horizontal)
        self.asv.setRange(-50, 200)
        self.asv_val = text("", wrap=False)
        self.asv.valueChanged.connect(self._asv_changed)
        row = QHBoxLayout()
        row.addWidget(self.asv, 1)
        row.addWidget(self.asv_val)
        root.addWidget(card(text("Суворість перевірки мовця", "h2"),
                            text("Поріг за замовчуванням визначено на дикторах ASVspoof 2019 (англійська мова). Для "
                                 "інших мов і мікрофонів його варто підібрати: вище — менше чужих проходить, але "
                                 "частіше відхиляються власники.", "muted"), row, margins=16))
        self.about = text("", "muted")
        root.addWidget(card(text("Про систему", "h2"), self.about, margins=16))
        root.addStretch(1)

    def refresh(self):
        import sounddevice as sd

        self.mic.blockSignals(True)
        self.mic.clear()
        self.mic_ids = [None]
        self.mic.addItem("Системний за замовчуванням")
        try:
            hostapi = sd.default.hostapi
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and d["hostapi"] == hostapi:
                    self.mic_ids.append(i)
                    self.mic.addItem(d["name"])
        except Exception:                                        # noqa: BLE001
            pass
        self.mic.blockSignals(False)
        if self.win.vg:
            self._cm_changed(self.cm_strict.currentText())
            self.asv.setValue(round(100 * self.win.vg.asv_threshold))
            d = self.win.vg.detector
            self.about.setText(
                f"Детектор підробки: {d.name}; поріг {d.threshold:+.3f} (суворість «{d.strictness}», обрано на dev "
                "ASVspoof 2019 LA, усі канали).\nВерифікація мовця: GMM-UBM, 256 компонент, MFCC + Δ + ΔΔ, MAP-адаптація.\n"
                "Рішення: запис приймається, лише якщо голос справжній І збігається з мовцем.\n"
                "Обмеження: детектор навчено на ASVspoof 2019 LA"
                + (" і доповнено українськими записами (Common Voice, синтезатор Piper)" if "україн" in d.name else "")
                + "; нові методи синтезу можуть бути не виявлені. Програма — допоміжний інструмент, а не доказ.")

    NOTES = {"м'яка": "Майже не відхиляє справжніх людей, але на нових методах синтезу пропускає 20–30 % підробок.",
             "звичайна": "Баланс: на нових методах синтезу пропускає 10–17 % підробок, відхиляє 0,5–5 % справжніх.",
             "сувора": "Пропускає 6–11 % підробок, але відхиляє 2–5 % справжніх (у мобільних дзвінках до 11–18 %)."}

    def _cm_changed(self, name: str):
        if self.win.vg:
            self.win.vg.detector.strictness = name
        self.cm_note.setText(self.NOTES[name] + " Оцінки — на eval ASVspoof 2019 LA для детектора WavLM.")

    def _asv_changed(self, v):
        self.asv_val.setText(f"{v / 100:+.2f}")
        if self.win.vg:
            self.win.vg.asv_threshold = v / 100


# --------------------------------------------------------------------------- головне вікно
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiceGuard — захист голосової верифікації")
        geo = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1400, int(geo.width() * 0.92)), min(1000, int(geo.height() * 0.92)))
        self.setMinimumSize(1100, 720)
        self.vg = None
        self.pool = QThreadPool.globalInstance()
        self.recorder = Recorder()
        self.settings = {"record_seconds": 5, "transcribe": True, "language": "uk", "ai_model": "gemma3:4b"}
        self.theme = "dark"

        central = QWidget()
        self.setCentralWidget(central)
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        side = QFrame()
        side.setObjectName("side")
        side.setFixedWidth(230)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(14, 18, 14, 14)
        sl.setSpacing(4)
        brand = QHBoxLayout()
        logo = icon_label("shield", 20, "accent")
        brand.addWidget(logo)
        name = QLabel("VoiceGuard")
        name.setFont(QFont("Segoe UI", 15, QFont.DemiBold))
        brand.addWidget(name)
        brand.addStretch(1)
        sl.addLayout(brand)
        sl.addWidget(text("Захист від підробки голосу", "muted"))
        sl.addSpacing(18)
        self.stack = QStackedWidget()
        self.pages = [CheckPage(self), TextPage(self), SpeakersPage(self), JournalPage(self), StatsPage(self),
                      BatchPage(self), SettingsPage(self)]
        self.nav = []
        for i, (key, label) in enumerate([("check", "Перевірка голосу"), ("text", "Перевірка тексту"),
                                          ("speakers", "Мовці"), ("journal", "Журнал"),
                                          ("stats", "Статистика"), ("batch", "Пакетна перевірка"),
                                          ("settings", "Налаштування")]):
            b = QPushButton(f"{ICONS[key]}     {label}")
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setFont(ui_font(10.5))
            b.clicked.connect(lambda _=False, idx=i: self.go(idx))
            sl.addWidget(b)
            self.nav.append(b)
            self.stack.addWidget(self.pages[i])
        sl.addStretch(1)
        self.status = text("Завантаження моделей…", "muted")
        sl.addWidget(self.status)
        lay.addWidget(side)
        lay.addWidget(self.stack, 1)
        self.toast = Toast(self)
        self._fx = None
        self.nav[0].setChecked(True)
        self.apply_theme("dark")
        self.pages[0].set_enabled(False)
        t = Task(self._load)
        t.signals.done.connect(self._loaded)
        t.signals.failed.connect(lambda e: (self.status.setText(f"Помилка: {e}"),
                                            QMessageBox.critical(self, "VoiceGuard", e)))
        self.pool.start(t)

    @staticmethod
    def _load():
        from .system import VoiceGuard

        vg = VoiceGuard()
        if vg.detector.kind == "ssl":
            vg.detector.score(1e-3 * np.random.default_rng(0).normal(size=SR).astype(np.float32))
        try:                                    # розпізнавання мовлення — прогріти заздалегідь, щоб перша
            from . import asr                   # перевірка не чекала завантаження моделі (~15–20 с)

            asr.transcribe(1e-3 * np.random.default_rng(1).normal(size=SR).astype(np.float32))
        except Exception:                       # noqa: BLE001 — без faster-whisper програма працює без змісту
            pass
        return vg

    def _loaded(self, vg):
        self.vg = vg
        self.pages[0].set_enabled(True)
        self.status.setText(f"● Готово\n{vg.detector.name}")
        self.status.setStyleSheet(f"color: {T['good']};")
        self.speakers_changed()
        self.pages[6].refresh()
        self.pages[0]._fill_details(None)
        self.toast.show_msg("Моделі завантажено. Можна перевіряти голос.", "good")

    def speakers_changed(self):
        names = self.vg.verifier.speakers()
        self.pages[0].refresh_speakers(names)
        self.pages[2].refresh(names)

    def model_report(self):
        import json

        from .system import GMM_TAG, ROOT

        if self.vg and self.vg.detector.kind == "ssl" and (ROOT / "results" / "cm_ssl.json").exists():
            r = json.loads((ROOT / "results" / "cm_ssl.json").read_text(encoding="utf-8"))
            ch = r["chosen"]
            return self.vg.detector.name, r["eval"][ch]["eer"]
        path = ROOT / "results" / f"cm_gmm_{GMM_TAG}.json"
        if path.exists():
            r = json.loads(path.read_text(encoding="utf-8"))
            return "LFCC 0–4 кГц + аугментація, GMM", {c: v["eer"] for c, v in r["eval"].items()}
        return None

    def go(self, idx: int):
        for i, b in enumerate(self.nav):
            b.setChecked(i == idx)
        page = self.pages[idx]
        if hasattr(page, "refresh") and idx in (3, 4, 6):
            page.refresh()
        self.stack.setCurrentIndex(idx)
        fx = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(fx)
        self._fx = anim(page, 0, 1, 260, lambda v: fx.setOpacity(float(v)))
        self._fx.finished.connect(lambda: page.setGraphicsEffect(None))
        self._fx.start()

    def apply_theme(self, name: str):
        T.clear()
        T.update(THEMES[name])
        self.theme = name
        QApplication.instance().setStyleSheet(stylesheet())
        for w in self.findChildren(QWidget):
            w.update()
        if self.vg:
            self.status.setStyleSheet(f"color: {T['good']};")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.toast.isVisible():
            self.toast.move(self.width() - self.toast.width() - 24, 20)


def main():
    import os
    import sys

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setApplicationName("VoiceGuard")
    app.setFont(ui_font(10))
    if ICON_FONT not in QFontDatabase.families():
        ICONS.update({k: v for k, v in zip(ICONS, "●" * len(ICONS))})
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
