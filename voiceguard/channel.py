"""Моделювання каналу зв'язку: що відбувається з голосом до того, як його перевіряють.

Шахрайський дзвінок чи голосове повідомлення проходить через кодек:
  * Opus — голосові повідомлення й дзвінки в Telegram, Viber, WhatsApp;
  * G.711 μ-law — класична телефонна лінія: смуга 300–3400 Гц, 8 кГц, 8 біт;
  * AMR-NB (12,2 кбіт/с, 8 кГц) — мобільний дзвінок GSM/3G;
  * AMR-WB (12,65 кбіт/с, 16 кГц) — «HD Voice» у мережах 3G/VoLTE;
  * GSM 06.10 (13 кбіт/с, 8 кГц) — найстаріший мобільний кодек;
  * адитивний шум із заданим SNR.
Кодеки з втратами відкидають саме ті тонкі високочастотні деталі, за якими
детектори впізнають синтез, тому стійкість до каналу — окреме питання
дослідження.
"""

from __future__ import annotations

import io
import subprocess

import numpy as np
from scipy import signal

from .audio import SAMPLE_RATE, resample


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _wav_bytes(x: np.ndarray, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _run(args: list[str], data: bytes) -> bytes:
    res = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", *args],
                         input=data, capture_output=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode(errors="replace"))
    return res.stdout


def opus(x: np.ndarray, bitrate_kbps: float = 16.0, sr: int = SAMPLE_RATE, application: str = "voip") -> np.ndarray:
    """Кодування й декодування Opus (libopus) із заданим бітрейтом."""
    import soundfile as sf

    encoded = _run(["-f", "wav", "-i", "pipe:0", "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k",
                    "-application", application, "-f", "ogg", "pipe:1"], _wav_bytes(x, sr))
    decoded = _run(["-f", "ogg", "-i", "pipe:0", "-ar", str(sr), "-ac", "1", "-f", "wav", "pipe:1"], encoded)
    y, _ = sf.read(io.BytesIO(decoded), dtype="float32")
    return _align(y, len(x))


def codec(x: np.ndarray, encoder: str, fmt: str, codec_sr: int, bitrate: str | None = None,
          sr: int = SAMPLE_RATE) -> np.ndarray:
    """Кодування й декодування довільним кодеком ffmpeg на частоті codec_sr, назад до sr."""
    import soundfile as sf

    y = resample(x, sr, codec_sr) if codec_sr != sr else x
    args = ["-f", "wav", "-i", "pipe:0", "-ar", str(codec_sr), "-ac", "1", "-c:a", encoder]
    if bitrate:
        args += ["-b:a", bitrate]
    encoded = _run(args + ["-f", fmt, "pipe:1"], _wav_bytes(y, codec_sr))
    decoded = _run(["-f", fmt, "-i", "pipe:0", "-ar", str(codec_sr), "-ac", "1", "-f", "wav", "pipe:1"], encoded)
    z, _ = sf.read(io.BytesIO(decoded), dtype="float32")
    return _align(resample(z, codec_sr, sr) if codec_sr != sr else z, len(x))


def mu_law(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Квантування G.711 μ-law (8 біт) і зворотне перетворення."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0
    return (np.sign(q) * ((1.0 + mu) ** np.abs(q) - 1.0) / mu).astype(np.float32)


def telephone(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Телефонний канал: смуга 300–3400 Гц, 8 кГц, G.711 μ-law, назад до sr."""
    sos = signal.butter(6, [300.0, 3400.0], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, x)
    y = mu_law(resample(y.astype(np.float32), sr, 8000))
    return _align(resample(y, 8000, sr), len(x))


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None,
              noise: np.ndarray | None = None) -> np.ndarray:
    """Адитивний шум (білий або заданий) із відношенням сигнал/шум snr_db."""
    rng = rng or np.random.default_rng(0)
    if noise is None:
        noise = rng.normal(0.0, 1.0, len(x))
    else:
        noise = np.resize(noise, len(x))
    p_sig, p_noise = float(np.mean(x ** 2)), float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(p_sig / (p_noise * 10.0 ** (snr_db / 10.0)))
    return (x + scale * noise).astype(np.float32)


def _align(y: np.ndarray, n: int) -> np.ndarray:
    y = y[:n]
    return np.pad(y, (0, n - len(y))).astype(np.float32)


CHANNELS = {
    "clean": lambda x: x,
    "opus32": lambda x: opus(x, 32.0),
    "opus16": lambda x: opus(x, 16.0),
    "opus8": lambda x: opus(x, 8.0),
    "telephone": telephone,
    "amrnb": lambda x: codec(x, "libopencore_amrnb", "amr", 8000, "12.2k"),
    "amrwb": lambda x: codec(x, "libvo_amrwbenc", "amr", 16000, "12.65k"),
    "gsm": lambda x: codec(x, "libgsm", "gsm", 8000),
    "noise20": lambda x: add_noise(x, 20.0),
    "noise10": lambda x: add_noise(x, 10.0),
}
