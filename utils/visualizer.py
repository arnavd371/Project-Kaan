"""Spectrogram visualization utilities for Kaan."""

from __future__ import annotations

import io
from typing import List, Union

import matplotlib

matplotlib.use("Agg")

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
LIGHT_BG = "#F7F6F2"
SURFACE = "#ffffff"
INK = "#000000"


def _load_waveform(audio_data: Union[np.ndarray, bytes, io.BytesIO], sr: int = SAMPLE_RATE) -> np.ndarray:
    if isinstance(audio_data, np.ndarray):
        return audio_data
    if isinstance(audio_data, (bytes, bytearray)):
        audio_data = io.BytesIO(audio_data)
    y, _ = librosa.load(audio_data, sr=sr, mono=True)
    return y


def plot_spectrogram(audio_data: Union[np.ndarray, bytes, io.BytesIO], sr: int = SAMPLE_RATE) -> Figure:
    """Compute and plot mel spectrogram with Physical Intelligence light styling."""
    y = _load_waveform(audio_data, sr)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    times = librosa.times_like(mel_db, sr=sr, hop_length=HOP_LENGTH)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=LIGHT_BG)
    ax.set_facecolor(SURFACE)

    img = ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="gray",
        extent=[times[0], times[-1], 0, N_MELS],
    )

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("dB", color=INK)
    cbar.ax.yaxis.set_tick_params(color=INK)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=INK)

    ax.set_xlabel("Time (seconds)", color=INK, fontfamily="monospace")
    ax.set_ylabel("Mel frequency bands", color=INK, fontfamily="monospace")
    ax.set_title(
        "Acoustic Analysis - Kaan",
        color=INK,
        fontsize=12,
        fontweight="bold",
        fontfamily="monospace",
    )
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.2)

    plt.tight_layout()
    return fig


def animate_spectrogram_frames(
    audio_data: Union[np.ndarray, bytes, io.BytesIO],
    sr: int = SAMPLE_RATE,
    window_sec: float = 1.0,
    step_sec: float = 0.25,
) -> List[Figure]:
    """Generate rolling 1-second window spectrogram frames."""
    y = _load_waveform(audio_data, sr)
    window_samples = int(window_sec * sr)
    step_samples = int(step_sec * sr)
    frames = []

    if len(y) < window_samples:
        return [plot_spectrogram(y, sr)]

    for start in range(0, len(y) - window_samples + 1, step_samples):
        window = y[start : start + window_samples]
        fig = plot_spectrogram(window, sr)
        time_offset = start / sr
        fig.axes[0].set_title(
            f"Acoustic Analysis - Kaan (t={time_offset:.1f}s)",
            color=INK,
            fontsize=12,
            fontweight="bold",
            fontfamily="monospace",
        )
        frames.append(fig)

    return frames
