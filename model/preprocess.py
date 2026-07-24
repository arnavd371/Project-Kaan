"""Audio preprocessing pipeline for GrainEar CNN model."""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Union

import librosa
import numpy as np
from scipy import ndimage

SAMPLE_RATE = 16000
DURATION_SEC = 10
N_SAMPLES = SAMPLE_RATE * DURATION_SEC
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MEL_SHAPE = (128, 128)


def load_audio(source: Union[str, Path, bytes, io.BytesIO]) -> np.ndarray:
    """Load audio at 16 kHz mono from a file path or raw bytes."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    y, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    return y


def trim_and_pad(y: np.ndarray) -> np.ndarray:
    """Trim silence and pad or crop to exactly 10 seconds."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trimmed) == 0:
        y_trimmed = y
    if len(y_trimmed) >= N_SAMPLES:
        start = (len(y_trimmed) - N_SAMPLES) // 2
        return y_trimmed[start : start + N_SAMPLES].astype(np.float32)
    pad_total = N_SAMPLES - len(y_trimmed)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y_trimmed, (pad_left, pad_right), mode="constant").astype(np.float32)


def _generate_pink_noise(length: int) -> np.ndarray:
    """Generate pink (1/f) noise of given length."""
    white = np.random.randn(length)
    fft = np.fft.rfft(white)
    freqs = np.arange(len(fft)) + 1
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=length)
    return pink.astype(np.float32)


def _add_pink_noise_at_snr(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Add pink noise at a specified signal-to-noise ratio in dB."""
    noise = _generate_pink_noise(len(y))
    signal_power = np.mean(y**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return y + noise


def augment_waveform(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply random augmentation for training (pitch shift, pink noise, time stretch)."""
    y_aug = y.copy()

    n_steps = random.uniform(-2, 2)
    y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=n_steps)

    snr_db = random.uniform(0, 10)
    y_aug = _add_pink_noise_at_snr(y_aug, snr_db)

    rate = random.choice([0.8, 1.2])
    y_aug = librosa.effects.time_stretch(y_aug, rate=rate)

    return trim_and_pad(y_aug)


def waveform_to_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """Convert waveform to normalized mel spectrogram shaped (128, 128, 1)."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    if mel_max - mel_min > 1e-8:
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = np.zeros_like(mel_db)

    zoom_factors = (MEL_SHAPE[0] / mel_norm.shape[0], MEL_SHAPE[1] / mel_norm.shape[1])
    mel_resized = ndimage.zoom(mel_norm, zoom_factors, order=1)
    mel_resized = mel_resized[: MEL_SHAPE[0], : MEL_SHAPE[1]]

    if mel_resized.shape[0] < MEL_SHAPE[0] or mel_resized.shape[1] < MEL_SHAPE[1]:
        mel_resized = np.pad(
            mel_resized,
            (
                (0, MEL_SHAPE[0] - mel_resized.shape[0]),
                (0, MEL_SHAPE[1] - mel_resized.shape[1]),
            ),
            mode="constant",
        )

    return mel_resized.reshape(MEL_SHAPE[0], MEL_SHAPE[1], 1).astype(np.float32)


def preprocess_audio(source: Union[str, Path, bytes, io.BytesIO]) -> np.ndarray:
    """Full pipeline: load -> trim/pad -> mel spectrogram -> (128, 128, 1)."""
    y = load_audio(source)
    y = trim_and_pad(y)
    return waveform_to_mel_spectrogram(y)


def preprocess_waveform(y: np.ndarray) -> np.ndarray:
    """Preprocess an already-loaded waveform array."""
    y = trim_and_pad(y)
    return waveform_to_mel_spectrogram(y)
