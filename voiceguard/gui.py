"""Десктопна програма VoiceGuard (Tkinter).

Вкладки: «Перевірка» (файл або мікрофон → чи справжній голос і чи це
заявлений мовець), «Мовці» (реєстрація), «Журнал», «Про систему».
Важкі операції (моделі, кодеки, запис) виконуються у фоновому потоці,
інтерфейс оновлюється через after().
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from . import audio, channel

SR = audio.SAMPLE_RATE
CHANNEL_CHOICES = {
    "без змін": "clean",
    "голосове Telegram/Viber (Opus 16 кбіт/с)": "opus16",
    "мобільний дзвінок (AMR-NB)": "amrnb",
    "HD Voice / VoLTE (AMR-WB)": "amrwb",
    "стаціонарний телефон (G.711)": "telephone",
    "шум 20 дБ": "noise20",
}
GOOD, BAD, WARN, INK, MUTED = "#1f7a3a", "#b3261e", "#9a6700", "#1d1d1b", "#6b6b66"


class Meter(tk.Canvas):
    """Шкала 0..1 з позначкою порогу посередині: зелена вище порогу, червона нижче (і підпис поруч)."""

    def __init__(self, master, width: int = 330, height: int = 16):
        super().__init__(master, width=width, height=height, highlightthickness=0, bg="#e6e6e3")
        self.w, self.h = width, height
        self.set(None)

    def set(self, value: float | None):
        self.delete("all")
        if value is not None:
            self.create_rectangle(0, 0, self.w * value, self.h, fill=GOOD if value >= 0.5 else BAD, width=0)
        self.create_line(self.w / 2, 0, self.w / 2, self.h, fill=INK, width=2)     # поріг


class App(tk.Tk):
    def __init__(self):
        try:                                                    # чіткий текст за масштабу Windows 125–150 %
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:                                       # noqa: BLE001
            pass
        super().__init__()
        self.title("VoiceGuard — перевірка голосу")
        w, h = min(1180, self.winfo_screenwidth() - 60), min(760, self.winfo_screenheight() - 100)
        self.geometry(f"{w}x{h}+20+10")
        self.minsize(860, 560)
        self.vg = None
        self.signal: np.ndarray | None = None
        self.source = ""
        self.enroll_clips: list[np.ndarray] = []
        self._style()
        self._inbox: queue.Queue = queue.Queue()     # результати фонових потоків → головний потік
        self.after(50, self._poll)
        self.status = tk.StringVar(value="Завантаження моделей…")
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        self.tab_check, self.tab_spk, self.tab_log, self.tab_about = (ttk.Frame(nb, padding=12) for _ in range(4))
        for tab, text in ((self.tab_check, "Перевірка"), (self.tab_spk, "Мовці"), (self.tab_log, "Журнал"),
                          (self.tab_about, "Про систему")):
            nb.add(tab, text=text)
        nb.bind("<<NotebookTabChanged>>", lambda e: self._on_tab(nb.index("current")))
        ttk.Label(self, textvariable=self.status, foreground=MUTED, anchor="w").pack(fill="x", padx=12, pady=6)
        self._build_check()
        self._build_speakers()
        self._build_log()
        self._build_about()
        self._busy(True)
        threading.Thread(target=self._load, daemon=True).start()

    # ------------------------------------------------------------------ загальне
    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("vista")
        except tk.TclError:
            st.theme_use("clam")
        st.configure("Verdict.TLabel", font=("Segoe UI", 20, "bold"))
        st.configure("Head.TLabel", font=("Segoe UI", 11, "bold"))

    def _load(self):
        try:
            from .system import VoiceGuard

            vg = VoiceGuard()
            if vg.detector.kind == "ssl":                        # прогрів моделі WavLM
                vg.detector.score(np.zeros(SR, np.float32) + 1e-3 * np.random.default_rng(0).normal(size=SR))
            self._inbox.put((lambda _: self._set_vg(vg), None))
        except Exception as e:                                   # noqa: BLE001
            msg = str(e)
            self._inbox.put((lambda _: (self.status.set(f"Помилка завантаження: {msg}"),
                                        messagebox.showerror("VoiceGuard", msg)), None))

    def _set_vg(self, vg):
        self.vg = vg
        self._loaded()

    def _poll(self):
        """Головний потік: виконує колбеки, які поклали фонові потоки (Tkinter не потокобезпечний)."""
        try:
            while True:
                fn, arg = self._inbox.get_nowait()
                fn(arg)
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _loaded(self):
        self._busy(False)
        self.status.set(f"Готово. Детектор: {self.vg.detector.name}")
        self._refresh_speakers()
        self._refresh_about()

    def _busy(self, on: bool):
        for b in getattr(self, "_actions", []):
            b.configure(state="disabled" if on else "normal")
        self.configure(cursor="watch" if on else "")

    def _run(self, work, done, message: str):
        """work() — у фоновому потоці; done(result) — в інтерфейсі."""
        self._busy(True)
        self.status.set(message)

        def target():
            try:
                res = work()
                self._inbox.put((lambda r: (self._busy(False), done(r)), res))
            except Exception as e:                               # noqa: BLE001
                msg = str(e)
                self._inbox.put((lambda _: (self._busy(False), self.status.set(f"Помилка: {msg}"),
                                            messagebox.showerror("VoiceGuard", msg)), None))

        threading.Thread(target=target, daemon=True).start()

    def _record(self, seconds: float, done):
        import sounddevice as sd

        def work():
            rec = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            return rec[:, 0]

        self._run(work, done, f"Запис {seconds:.0f} с… говоріть звичайним голосом")

    @staticmethod
    def _play(x: np.ndarray):
        import sounddevice as sd

        sd.stop()
        sd.play(audio.normalize(x, 0.8), SR)

    def _on_tab(self, idx: int):
        if idx == 2:
            self._refresh_log()

    # ------------------------------------------------------------------ Перевірка
    def _build_check(self):
        f = self.tab_check
        top = ttk.Frame(f)
        top.pack(fill="x")
        ttk.Label(top, text="Мовець:").grid(row=0, column=0, sticky="w")
        self.speaker = ttk.Combobox(top, state="readonly", width=28)
        self.speaker.grid(row=0, column=1, sticky="w", padx=(6, 18))
        ttk.Label(top, text="Канал зв’язку:").grid(row=0, column=2, sticky="w")
        self.chan = ttk.Combobox(top, state="readonly", width=40, values=list(CHANNEL_CHOICES))
        self.chan.current(0)
        self.chan.grid(row=0, column=3, sticky="w", padx=6)
        btns = ttk.Frame(f)
        btns.pack(fill="x", pady=10)
        b_open = ttk.Button(btns, text="Відкрити файл…", command=self._open)
        b_rec = ttk.Button(btns, text="Записати 5 с", command=lambda: self._record(5, self._got_signal("мікрофон")))
        b_play = ttk.Button(btns, text="Відтворити",
                            command=lambda: self.signal is not None and self._play(self._channelled(self._chan_key())))
        b_check = ttk.Button(btns, text="Перевірити", command=self._check)
        for b in (b_open, b_rec, b_play, b_check):
            b.pack(side="left", padx=(0, 8))
        self._actions = [b_open, b_rec, b_play, b_check]
        self.src_label = ttk.Label(btns, text="Запис не вибрано", foreground=MUTED)
        self.src_label.pack(side="left", padx=10)

        body = ttk.Frame(f)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, width=360)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        self.verdict = ttk.Label(left, text="—", style="Verdict.TLabel", wraplength=340)
        self.verdict.pack(anchor="w", pady=(4, 2))
        self.reason = ttk.Label(left, text="Виберіть або запишіть фразу 3–5 с і натисніть «Перевірити».",
                                wraplength=340, foreground=MUTED)
        self.reason.pack(anchor="w", pady=(0, 14))
        ttk.Label(left, text="Справжність голосу", style="Head.TLabel").pack(anchor="w")
        self.cm_bar = Meter(left)
        self.cm_bar.pack(anchor="w", pady=2)
        self.cm_text = ttk.Label(left, text="Риска посередині — поріг рішення.", foreground=MUTED, wraplength=340)
        self.cm_text.pack(anchor="w", pady=(0, 12))
        ttk.Label(left, text="Збіг із голосом мовця", style="Head.TLabel").pack(anchor="w")
        self.asv_bar = Meter(left)
        self.asv_bar.pack(anchor="w", pady=2)
        self.asv_text = ttk.Label(left, text="", foreground=MUTED, wraplength=340)
        self.asv_text.pack(anchor="w", pady=(0, 12))
        self.notes = ttk.Label(left, text="", foreground=WARN, wraplength=340)
        self.notes.pack(anchor="w")

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(6, 4.2), dpi=100)
        self.ax_wave, self.ax_spec = self.fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2.2]})
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.1, hspace=0.35)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self._draw(None)

    def _got_signal(self, source: str):
        def done(x: np.ndarray):
            self.signal, self.source = x.astype(np.float32), source
            self.src_label.configure(text=f"{source}: {len(x) / SR:.1f} с", foreground=INK)
            self.status.set("Запис готовий. Натисніть «Перевірити».")
            self._draw(self.signal)
        return done

    def _open(self):
        path = filedialog.askopenfilename(title="Аудіофайл", filetypes=[("Аудіо", "*.wav *.flac *.ogg *.mp3"), ("Усі", "*.*")])
        if path:
            self._run(lambda: audio.load(path), self._got_signal(Path(path).name), "Читання файлу…")

    def _chan_key(self) -> str:
        return CHANNEL_CHOICES[self.chan.get()]

    def _channelled(self, key: str) -> np.ndarray:
        """Запис після вибраного каналу (key зчитується в головному потоці)."""
        return channel.CHANNELS[key](self.signal)

    def _check(self):
        if self.signal is None:
            messagebox.showinfo("VoiceGuard", "Спершу відкрийте файл або запишіть фразу.")
            return
        spk = self.speaker.get()
        spk = None if spk in ("", "(без верифікації — лише справжність)") else spk
        chan, key = self.chan.get(), self._chan_key()

        def work():
            x = self._channelled(key)
            return x, self.vg.check(x, spk, source=f"{self.source} | {chan}")

        self._run(work, self._show, "Аналіз…")

    def _show(self, res):
        x, v = res
        self._draw(x)
        if not v.cm_genuine:
            text, color, why = "✗ ПІДРОБКА ГОЛОСУ", BAD, "Ознаки синтезу або перетворення голосу."
        elif v.speaker and not v.asv_match:
            text, color, why = "✗ ІНШИЙ МОВЕЦЬ", BAD, f"Голос справжній, але не збігається з «{v.speaker}»."
        elif v.speaker:
            text, color, why = "✓ ПРИЙНЯТО", GOOD, f"Справжній голос мовця «{v.speaker}»."
        else:
            text, color, why = "✓ СПРАВЖНІЙ ГОЛОС", GOOD, "Ознак підробки не виявлено (мовця не перевіряли)."
        self.verdict.configure(text=text, foreground=color)
        self.reason.configure(text=why, foreground=INK)
        self.cm_bar.set(v.cm_confidence)
        self.cm_text.configure(text=f"Оцінка {v.cm_score:.2f} при порозі {v.cm_threshold:.2f} "
                                    f"(впевненість «справжній» {100 * v.cm_confidence:.0f} %)")
        if v.speaker:
            rel = 1.0 / (1.0 + np.exp(-(v.asv_score - v.asv_threshold) / 0.3))
            self.asv_bar.set(rel)
            self.asv_text.configure(text=f"Оцінка {v.asv_score:.2f} при порозі {v.asv_threshold:.2f}")
        else:
            self.asv_bar.set(None)
            self.asv_text.configure(text="Мовця не вибрано.")
        self.notes.configure(text="\n".join(v.notes))
        self.status.set(f"Готово за {v.seconds:.2f} с · детектор: {v.detector}")

    def _draw(self, x: np.ndarray | None):
        for ax in (self.ax_wave, self.ax_spec):
            ax.clear()
            ax.tick_params(labelsize=8, colors=MUTED)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        self.ax_wave.set_title("Сигнал", fontsize=9, loc="left", color=INK)
        self.ax_spec.set_title("Спектрограма (детектор використовує 0–4 кГц)", fontsize=9, loc="left", color=INK)
        if x is not None and len(x):
            t = np.arange(len(x)) / SR
            self.ax_wave.plot(t, x, color="#2a78d6", linewidth=0.5)
            self.ax_wave.set_xlim(0, t[-1])
            pxx, _, _, im = self.ax_spec.specgram(x, NFFT=512, Fs=SR, noverlap=352, cmap="magma")
            top = 10 * np.log10(pxx.max() + 1e-20)
            im.set_clim(top - 70, top)          # фіксований діапазон 70 дБ: залишки нижче — чорні
            self.ax_spec.axhline(4000, color="white", linewidth=0.8, linestyle="--")
            self.ax_spec.set_ylabel("Гц", fontsize=8)
            self.ax_spec.set_xlabel("с", fontsize=8)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ Мовці
    def _build_speakers(self):
        f = self.tab_spk
        left = ttk.Frame(f)
        left.pack(side="left", fill="y", padx=(0, 20))
        ttk.Label(left, text="Зареєстровані мовці", style="Head.TLabel").pack(anchor="w")
        self.spk_list = tk.Listbox(left, width=32, height=18)
        self.spk_list.pack(anchor="w", pady=6)
        ttk.Button(left, text="Видалити вибраного", command=self._delete_speaker).pack(anchor="w")

        right = ttk.Frame(f)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Новий мовець", style="Head.TLabel").pack(anchor="w")
        ttk.Label(right, foreground=MUTED, wraplength=520,
                  text="Запишіть або додайте 3–5 фраз по 4–6 с (разом щонайменше 15 с мовлення). "
                       "Записуйте лише себе або людей, які дали на це згоду.").pack(anchor="w", pady=(2, 10))
        row = ttk.Frame(right)
        row.pack(anchor="w")
        ttk.Label(row, text="Ім’я:").pack(side="left")
        self.new_name = ttk.Entry(row, width=30)
        self.new_name.pack(side="left", padx=6)
        row2 = ttk.Frame(right)
        row2.pack(anchor="w", pady=10)
        b1 = ttk.Button(row2, text="Записати фразу (5 с)", command=lambda: self._record(5, self._add_clip("мікрофон")))
        b2 = ttk.Button(row2, text="Додати файли…", command=self._add_files)
        b3 = ttk.Button(row2, text="Очистити", command=self._clear_clips)
        b4 = ttk.Button(row2, text="Зареєструвати", command=self._enroll)
        for b in (b1, b2, b3, b4):
            b.pack(side="left", padx=(0, 8))
        self._actions += [b1, b2, b4]
        self.clips_label = ttk.Label(right, text="Фраз: 0", foreground=MUTED)
        self.clips_label.pack(anchor="w")

    def _add_clip(self, source: str):
        def done(x):
            self.enroll_clips.append(x.astype(np.float32))
            self._clips_text()
            self.status.set(f"Додано фразу ({source}).")
        return done

    def _add_files(self):
        paths = filedialog.askopenfilenames(title="Фрази мовця", filetypes=[("Аудіо", "*.wav *.flac *.ogg *.mp3")])
        if paths:
            self._run(lambda: [audio.load(p) for p in paths],
                      lambda xs: (self.enroll_clips.extend(xs), self._clips_text()), "Читання файлів…")

    def _clear_clips(self):
        self.enroll_clips = []
        self._clips_text()

    def _clips_text(self):
        total = sum(len(x) for x in self.enroll_clips) / SR
        self.clips_label.configure(text=f"Фраз: {len(self.enroll_clips)}, разом {total:.1f} с запису",
                                   foreground=INK if total >= 15 else WARN)

    def _enroll(self):
        name = self.new_name.get().strip()
        if not name or any(c in name for c in '\\/:*?"<>|'):
            messagebox.showinfo("VoiceGuard", "Введіть ім’я без символів \\ / : * ? \" < > |")
            return
        if not self.enroll_clips:
            messagebox.showinfo("VoiceGuard", "Додайте хоча б одну фразу.")
            return

        def done(speech):
            self.enroll_clips = []
            self._clips_text()
            self._refresh_speakers()
            note = "" if speech >= 15 else " Мовлення замало — точність буде нижчою."
            self.status.set(f"Мовця «{name}» зареєстровано: {speech:.1f} с мовлення.{note}")

        self._run(lambda: self.vg.verifier.enroll(name, self.enroll_clips), done, "Реєстрація…")

    def _delete_speaker(self):
        sel = self.spk_list.curselection()
        if sel and messagebox.askyesno("VoiceGuard", f"Видалити «{self.spk_list.get(sel[0])}»?"):
            self.vg.verifier.delete(self.spk_list.get(sel[0]))
            self._refresh_speakers()

    def _refresh_speakers(self):
        names = self.vg.verifier.speakers() if self.vg else []
        self.spk_list.delete(0, "end")
        for n in names:
            self.spk_list.insert("end", n)
        self.speaker.configure(values=["(без верифікації — лише справжність)"] + names)
        if not self.speaker.get():
            self.speaker.current(0)

    # ------------------------------------------------------------------ Журнал
    def _build_log(self):
        cols = ("time", "source", "speaker", "verdict", "cm", "asv")
        self.log = ttk.Treeview(self.tab_log, columns=cols, show="headings")
        for c, (title, w) in zip(cols, (("Час", 140), ("Запис і канал", 330), ("Мовець", 120), ("Рішення", 170),
                                        ("Справжність", 100), ("Збіг", 80))):
            self.log.heading(c, text=title)
            self.log.column(c, width=w, anchor="w")
        self.log.pack(fill="both", expand=True)

    def _refresh_log(self):
        if not self.vg:
            return
        self.log.delete(*self.log.get_children())
        for t, src, v in self.vg.journal():
            if not v["cm_genuine"]:
                verdict = "✗ підробка"
            elif v["speaker"] and not v["asv_match"]:
                verdict = "✗ інший мовець"
            else:
                verdict = "✓ прийнято" if v["speaker"] else "✓ справжній"
            self.log.insert("", "end", values=(time.strftime("%d.%m %H:%M:%S", time.localtime(t)), src,
                                               v["speaker"] or "—", verdict, f"{100 * v['cm_confidence']:.0f} %",
                                               "—" if v["asv_score"] is None else f"{v['asv_score']:.2f}"))

    # ------------------------------------------------------------------ Про систему
    def _build_about(self):
        f = self.tab_about
        self.about = ttk.Label(f, text="", justify="left", wraplength=900)
        self.about.pack(anchor="w")
        ttk.Label(f, text="Суворість перевірки мовця", style="Head.TLabel").pack(anchor="w", pady=(18, 2))
        ttk.Label(f, foreground=MUTED, wraplength=900,
                  text="Поріг за замовчуванням визначено на дикторах ASVspoof 2019 (англійська мова). Для інших "
                       "мов і мікрофонів його варто підібрати: вище — менше чужих проходить, але частіше "
                       "відхиляються власники.").pack(anchor="w")
        self.asv_scale = tk.Scale(f, from_=-0.5, to=2.0, resolution=0.05, orient="horizontal", length=420,
                                  command=lambda v: self._set_asv_threshold(float(v)))
        self.asv_scale.pack(anchor="w")

    def _set_asv_threshold(self, v: float):
        if self.vg:
            self.vg.asv_threshold = v

    def _refresh_about(self):
        d = self.vg.detector
        self.asv_scale.set(round(self.vg.asv_threshold, 2))
        self.about.configure(text=(
            "VoiceGuard — верифікація мовця із захистом від підробки голосу.\n\n"
            f"Детектор підробки: {d.name}\n"
            f"   поріг {d.threshold:.3f} — точка рівних помилок на dev ASVspoof 2019 LA (усі канали)\n"
            "Верифікація мовця: GMM-UBM (256 компонент, MFCC + Δ + ΔΔ, MAP-адаптація середніх)\n\n"
            "Рішення: запис приймається, лише якщо голос визнано справжнім І він збігається з мовцем.\n"
            "Обмеження: моделі навчено на англомовному наборі ASVspoof 2019 LA; нові методи синтезу голосу "
            "можуть бути не виявлені. Програма — допоміжний інструмент, а не доказ."))


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
