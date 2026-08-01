"""Feature extractors for classical models and mel-CNN inputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.preprocess import (  # noqa: E402
    SAMPLE_RATE,
    load_audio,
    trim_and_pad,
    waveform_to_mel_spectrogram,
)

def _is_smoke_path(path: str) -> bool:
    return str(path).startswith("smoke:")


def load_waveform(path: str, class_hint: int | None = None, smoke_seed: int = 0) -> np.ndarray:
    if _is_smoke_path(path):
        parts = str(path).split(":")
        class_idx = int(parts[1]) if len(parts) > 1 else (class_hint or 0)
        sample_i = int(parts[2]) if len(parts) > 2 else 0
        rng = np.random.default_rng(smoke_seed + class_idx * 1000 + sample_i)
        n = SAMPLE_RATE * 10
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        base_freqs = [120.0, 360.0, 520.0, 780.0]
        f0 = base_freqs[class_idx % len(base_freqs)]
        y = 0.35 * np.sin(2 * np.pi * f0 * t)
        y += 0.15 * np.sin(2 * np.pi * (2 * f0) * t)
        y += 0.08 * rng.standard_normal(n)
        return y.astype(np.float32)

    y = load_audio(path)
    return trim_and_pad(y)


def extract_handcrafted(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    y = trim_and_pad(y) if len(y) != SAMPLE_RATE * 10 else y
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats: list[np.ndarray] = [
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
    ]
    for extractor in (
        librosa.feature.spectral_centroid,
        librosa.feature.spectral_bandwidth,
        librosa.feature.spectral_rolloff,
        librosa.feature.zero_crossing_rate,
        librosa.feature.rms,
    ):
        if extractor in (librosa.feature.zero_crossing_rate, librosa.feature.rms):
            feat = extractor(y=y)
        else:
            feat = extractor(y=y, sr=sr)
        feats.append(np.array([feat.mean(), feat.std()], dtype=np.float32))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.append(chroma.mean(axis=1))
    feats.append(chroma.std(axis=1))
    return np.concatenate([f.ravel() for f in feats]).astype(np.float32)


def extract_mel(y: np.ndarray) -> np.ndarray:
    return waveform_to_mel_spectrogram(trim_and_pad(y))


def build_feature_matrices(
    paths: Iterable[str],
    labels: np.ndarray | None = None,
    smoke_seed: int = 42,
    return_waveforms: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = list(paths)
    hand: list[np.ndarray] = []
    mels: list[np.ndarray] = []
    waves: list[np.ndarray] = []
    for i, path in enumerate(paths):
        class_hint = int(labels[i]) if labels is not None else None
        y = load_waveform(str(path), class_hint=class_hint, smoke_seed=smoke_seed)
        hand.append(extract_handcrafted(y))
        mels.append(extract_mel(y))
        if return_waveforms:
            waves.append(trim_and_pad(y).astype(np.float32))
    X_hand = np.stack(hand).astype(np.float32)
    X_mel = np.stack(mels).astype(np.float32)
    if return_waveforms:
        return X_hand, X_mel, np.stack(waves).astype(np.float32)
    return X_hand, X_mel


def feature_dim_handcrafted() -> int:
    return 74


def describe_approaches() -> list[dict]:
    return [
        {"id": "cnn_shallow", "family": "cnn", "features": "mel", "note": "train.py Project Kaan CNN"},
        {"id": "cnn_deep", "family": "cnn", "features": "mel", "note": "train_kaggle.py v5 deeper CNN"},
        {"id": "cnn1d", "family": "cnn", "features": "mel", "note": "1D CNN on mel time axis"},
        {"id": "yamnet_probe", "family": "pretrained", "features": "waveform", "note": "YAMNet embed + logistic"},
        {"id": "svm_rbf", "family": "classical", "features": "handcrafted"},
        {"id": "mlp", "family": "classical", "features": "handcrafted"},
        {"id": "gbdt", "family": "classical", "features": "handcrafted"},
        {"id": "rf", "family": "classical", "features": "handcrafted"},
        {"id": "extratrees", "family": "classical", "features": "handcrafted"},
        {"id": "knn", "family": "classical", "features": "handcrafted"},
        {"id": "logreg", "family": "classical", "features": "handcrafted"},
    ]
