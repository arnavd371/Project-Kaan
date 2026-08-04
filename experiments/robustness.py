"""Phone-like waveform degradations for a robustness ladder (no field mics)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from model.preprocess import SAMPLE_RATE, trim_and_pad


def _ensure_len(y: np.ndarray, n: int | None = None) -> np.ndarray:
    n = n or (SAMPLE_RATE * 10)
    y = np.asarray(y, dtype=np.float32).ravel()
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y))).astype(np.float32)


def add_noise_snr(y: np.ndarray, snr_db: float, rng: np.random.Generator, pink: bool = True) -> np.ndarray:
    y = _ensure_len(y)
    if pink:
        white = rng.standard_normal(len(y))
        fft = np.fft.rfft(white)
        freqs = np.arange(len(fft)) + 1.0
        fft = fft / np.sqrt(freqs)
        noise = np.fft.irfft(fft, n=len(y)).astype(np.float32)
    else:
        noise = rng.standard_normal(len(y)).astype(np.float32)
    sp = float(np.mean(y**2) + 1e-10)
    np_ = float(np.mean(noise**2) + 1e-10)
    target = sp / (10 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target / np_)
    return (y + noise).astype(np.float32)


def apply_gain(y: np.ndarray, db: float) -> np.ndarray:
    return (y * (10 ** (db / 20.0))).astype(np.float32)


def apply_clip(y: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    peak = float(np.max(np.abs(y)) + 1e-8)
    y = y / peak
    return np.clip(y, -threshold, threshold).astype(np.float32)


def phone_bandpass(y: np.ndarray, low_hz: float = 300.0, high_hz: float = 3400.0) -> np.ndarray:
    y = _ensure_len(y)
    nyq = SAMPLE_RATE / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, y).astype(np.float32)


def soft_compress(y: np.ndarray, ratio: float = 4.0, threshold: float = 0.2) -> np.ndarray:
    y = _ensure_len(y)
    mag = np.abs(y)
    out = y.copy()
    mask = mag > threshold
    over = mag[mask] - threshold
    mag_c = threshold + over / ratio
    out[mask] = np.sign(y[mask]) * mag_c
    peak = float(np.max(np.abs(out)) + 1e-8)
    return (out / peak * float(np.max(np.abs(y)) + 1e-8)).astype(np.float32)


def muffling_lpf(y: np.ndarray, cutoff_hz: float = 1200.0) -> np.ndarray:
    y = _ensure_len(y)
    b, a = butter(2, min(cutoff_hz / (SAMPLE_RATE / 2.0), 0.99), btype="low")
    return filtfilt(b, a, y).astype(np.float32)


def light_reverb(y: np.ndarray, decay: float = 0.35, delay_ms: float = 45.0) -> np.ndarray:
    y = _ensure_len(y)
    delay = int(SAMPLE_RATE * delay_ms / 1000.0)
    out = y.copy()
    if delay < len(y):
        echo = np.zeros_like(y)
        echo[delay:] = y[:-delay] * decay
        out = out + echo
    peak = float(np.max(np.abs(out)) + 1e-8)
    return (out / peak * float(np.max(np.abs(y)) + 1e-8)).astype(np.float32)


def agc(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    y = _ensure_len(y)
    rms = float(np.sqrt(np.mean(y**2) + 1e-10))
    return (y * (target_rms / rms)).astype(np.float32)


def ladder_specs() -> list[tuple[str, str]]:
    return [
        ("clean", "Identity (no degradation)"),
        ("snr20", "Pink noise SNR 20 dB"),
        ("snr10", "Pink noise SNR 10 dB"),
        ("snr5", "Pink noise SNR 5 dB"),
        ("snr0", "Pink noise SNR 0 dB"),
        ("phone_band", "Telephone band-pass 300-3400 Hz"),
        ("muffle", "Bag-muffle LPF ~1.2 kHz"),
        ("compress", "Soft compression"),
        ("clip", "Hard clip"),
        ("reverb", "Light delay reverb"),
        ("gain_-12", "Gain −12 dB then AGC"),
        ("combo_phone", "Phone band + SNR 10 + compress"),
        ("combo_hard", "Muffle + SNR 5 + clip"),
    ]


def apply_rung(name: str, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = trim_and_pad(_ensure_len(y))
    if name == "clean":
        return y
    if name == "snr20":
        return add_noise_snr(y, 20.0, rng)
    if name == "snr10":
        return add_noise_snr(y, 10.0, rng)
    if name == "snr5":
        return add_noise_snr(y, 5.0, rng)
    if name == "snr0":
        return add_noise_snr(y, 0.0, rng)
    if name == "phone_band":
        return phone_bandpass(y)
    if name == "muffle":
        return muffling_lpf(y)
    if name == "compress":
        return soft_compress(y)
    if name == "clip":
        return apply_clip(y, 0.45)
    if name == "reverb":
        return light_reverb(y)
    if name == "gain_-12":
        return agc(apply_gain(y, -12.0))
    if name == "combo_phone":
        z = phone_bandpass(y)
        z = add_noise_snr(z, 10.0, rng)
        return soft_compress(z)
    if name == "combo_hard":
        z = muffling_lpf(y)
        z = add_noise_snr(z, 5.0, rng)
        return apply_clip(z, 0.4)
    raise KeyError(name)
