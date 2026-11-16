"""Юніт-тести ядра. Запуск: python -m unittest discover -s tests"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voiceguard import audio, channel, features, metrics  # noqa: E402

SR = 16000


def tone(freq: float, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestFeatures(unittest.TestCase):
    def test_power_spectrum_peak_at_tone_frequency(self):
        spec = features.power_spectrum(tone(1000.0), features.FrameConfig(preemphasis=0.0))
        self.assertEqual(int(np.argmax(spec.mean(axis=0))), round(1000 / SR * 512))

    def test_filterbanks_cover_band_without_gaps(self):
        for bank in (features.mel_filterbank(40), features.linear_filterbank(20)):
            # нижче ~600 Гц мел-фільтри вужчі за крок ДПФ (31,25 Гц); угорі спадна половина
            # останнього (широкого) фільтра займає ~16 відліків
            covered = bank.sum(axis=0)[20:-20]
            self.assertGreater(covered.min(), 0.4)
            self.assertLessEqual(covered.max(), 1.0 + 1e-9)

    def test_deltas_of_linear_ramp_equal_slope(self):
        ramp = np.arange(50, dtype=float)[:, None] * 0.3
        np.testing.assert_allclose(features.deltas(ramp)[3:-3], 0.3, atol=1e-9)

    def test_lfcc_and_mfcc_shapes(self):
        x = tone(440.0, 1.0)
        for f in (features.lfcc(x), features.mfcc(x)):
            self.assertEqual(f.shape, (1 + (SR - 320) // 160, 60))
            self.assertTrue(np.isfinite(f).all())

    def test_cepstrum_invariant_to_gain_except_c0(self):
        # шум заповнює всі фільтри, як у мові; на чистому тоні порожні фільтри
        # визначає поправка 1e-10 у логарифмі
        x = tone(300.0) + 0.3 * tone(2500.0) + 0.05 * np.random.default_rng(2).normal(size=SR).astype(np.float32)
        a, b = features.lfcc(x)[:, 1:20], features.lfcc(0.1 * x)[:, 1:20]
        np.testing.assert_allclose(a, b, atol=1e-3)


class TestMetrics(unittest.TestCase):
    def test_eer_separated_and_overlapping(self):
        self.assertAlmostEqual(metrics.eer([2, 3, 4], [-1, 0, 1])[0], 0.0)
        rng = np.random.default_rng(0)
        same = metrics.eer(rng.normal(0, 1, 20000), rng.normal(0, 1, 20000))[0]
        self.assertAlmostEqual(same, 0.5, delta=0.02)

    def test_eer_gaussian_matches_theory(self):
        # дві нормальні з відстанню d між середніми: EER = Φ(−d/2)
        rng = np.random.default_rng(1)
        value = metrics.eer(rng.normal(2, 1, 50000), rng.normal(0, 1, 50000))[0]
        self.assertAlmostEqual(value, 0.1587, delta=0.005)

    def test_min_dcf_perfect_is_zero(self):
        self.assertAlmostEqual(metrics.min_dcf([5, 6], [0, 1]), 0.0)


class TestChannel(unittest.TestCase):
    def test_mu_law_roundtrip_error_small(self):
        x = tone(500.0)
        self.assertLess(np.abs(channel.mu_law(x) - x).max(), 0.02)

    def test_telephone_removes_high_band(self):
        x = tone(1000.0) + tone(6000.0)
        y = channel.telephone(x)
        spec = np.abs(np.fft.rfft(y[2000:-2000])) ** 2
        f = np.fft.rfftfreq(len(y) - 4000, 1 / SR)
        band = lambda lo, hi: spec[(f > lo) & (f < hi)].sum()  # noqa: E731
        self.assertLess(band(5800, 6200), 1e-3 * band(900, 1100))

    def test_noise_snr(self):
        x = tone(700.0)
        y = channel.add_noise(x, 10.0)
        snr = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
        self.assertAlmostEqual(snr, 10.0, delta=0.1)

    def test_opus_keeps_length_and_waveform(self):
        x = tone(440.0, 1.0)
        y = channel.opus(x, 32.0)
        self.assertEqual(len(y), len(x))
        # кодек має затримку — порівнюємо за максимумом взаємної кореляції
        c = np.correlate(y[4000:12000], x[4000:12000], mode="full")
        corr = c.max() / np.sqrt(np.sum(x[4000:12000] ** 2) * np.sum(y[4000:12000] ** 2))
        self.assertGreater(corr, 0.9)


    def test_mobile_codecs_keep_length_and_band(self):
        x = tone(440.0, 1.0) + tone(6000.0, 1.0, 0.2)
        for name, narrow in (("amrnb", True), ("gsm", True), ("amrwb", False)):
            y = channel.CHANNELS[name](x)
            self.assertEqual(len(y), len(x))
            spec = np.abs(np.fft.rfft(y)) ** 2
            f = np.fft.rfftfreq(len(y), 1 / SR)
            high = spec[(f > 5800) & (f < 6200)].sum() / spec[(f > 400) & (f < 480)].sum()
            # вузькосмугові кодеки (8 кГц) не можуть передати 6 кГц
            self.assertTrue(high < 1e-3 if narrow else np.isfinite(high), name)


class TestScam(unittest.TestCase):
    def test_bank_security_scheme_is_high_risk(self):
        from voiceguard import risk, scam

        r = scam.analyze("Добрий день, вас турбує служба безпеки ПриватБанку. По вашій картці підозріла операція. "
                         "Терміново продиктуйте код з SMS, інакше картку заблокують. Нікому не кажіть про цю розмову.")
        self.assertIn("codes", r.tactics)
        self.assertIn("authority", r.tactics)
        self.assertTrue(r.combos)
        self.assertGreater(risk.combine(content=r).score, 0.85)

    def test_relative_in_trouble(self):
        from voiceguard import scam

        r = scam.analyze("Мамо, це я, я потрапив у ДТП, терміново перекинь гроші на картку адвоката")
        self.assertIn("relative", r.tactics)
        self.assertIn("money", r.tactics)

    def test_ordinary_conversation_is_low_risk(self):
        from voiceguard import risk, scam

        r = scam.analyze("Привіт, я вже виходжу з роботи, купити хліба й молока? Синій пакет у шафі.")
        self.assertLess(risk.combine(content=r).score, 0.3)

    def test_mention_is_not_a_request(self):
        from voiceguard import scam

        # згадки теми без прохання — низький ризик; те саме з проханням — високий
        for t in ("Привіт, мамо! Завтра заїду до бабусі.", "Пароль від Wi-Fi змінено, новий на касі.",
                  "За даними НБУ, обсяг інвестицій у 2023 році зріс."):
            self.assertLess(scam.analyze(t).score, 0.3, t)
        self.assertGreater(scam.analyze("Назвіть, будь ласка, пароль від банківського застосунку.").score, 0.85)

    def test_phishing_url_and_official_url(self):
        from voiceguard import scam

        bad = scam.check_url("privat24-bonus.xyz/login")
        good = scam.check_url("https://www.privat24.ua")
        self.assertGreater(bad.risk, 0.6)
        self.assertEqual(good.risk, 0.0)
        found = scam.find_links("Отримайте виплату тут: bit.ly/3xYz та надішліть номер 4149 6293 1234 5678")
        self.assertEqual({f.kind for f in found}, {"url", "card"})


class TestAudio(unittest.TestCase):
    def test_trim_silence_removes_leading_and_trailing(self):
        x = np.concatenate([np.zeros(SR), tone(300.0, 0.5), np.zeros(SR)]).astype(np.float32)
        y = audio.trim_silence(x)
        self.assertLess(len(y), 0.5 * SR + 1500)
        self.assertGreater(len(y), 0.5 * SR)

    def test_resample_preserves_tone(self):
        y = audio.resample(tone(500.0), SR, 8000)
        spec = np.abs(np.fft.rfft(y))
        self.assertAlmostEqual(np.fft.rfftfreq(len(y), 1 / 8000)[np.argmax(spec)], 500.0, delta=2.0)


class TestModels(unittest.TestCase):
    def test_diag_logpdf_matches_sklearn(self):
        from voiceguard.models import _diag_gmm_logpdf, fit_gmm

        rng = np.random.default_rng(3)
        x = rng.normal(size=(500, 4))
        g = fit_gmm(x, 3)
        np.testing.assert_allclose(_diag_gmm_logpdf(x, g.weights_, g.means_, g.covariances_),
                                   g.score_samples(x), atol=1e-8)

    def test_gmm_ubm_prefers_enrolled_speaker(self):
        from voiceguard.models import GMMUBMVerifier

        rng = np.random.default_rng(4)
        centers = rng.normal(0, 2, size=(52, 5))                # мовці — різні центри ознак
        speak = lambda k, n: centers[k] + rng.normal(size=(n, 5))  # noqa: E731
        # UBM — на 50 мовцях; перевіряються двоє нових, яких в UBM не було
        ubm = GMMUBMVerifier.fit_ubm(np.concatenate([speak(k, 100) for k in range(50)]), 8)
        model = ubm.enroll([speak(50, 300)])
        self.assertGreater(ubm.score(model, speak(50, 200)), ubm.score(model, speak(51, 200)) + 0.1)

    def test_countermeasure_separates_classes(self):
        from voiceguard.models import GMMCountermeasure

        rng = np.random.default_rng(5)
        cm = GMMCountermeasure.fit(rng.normal(0, 1, (2000, 3)), rng.normal(1.5, 1, (2000, 3)), 4)
        self.assertGreater(cm.score(rng.normal(0, 1, (200, 3))), cm.score(rng.normal(1.5, 1, (200, 3))))


if __name__ == "__main__":
    unittest.main()
