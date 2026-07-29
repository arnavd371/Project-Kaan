"""Generate synthetic pink-noise WAV files to expand the clean class."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.preprocess import N_SAMPLES, SAMPLE_RATE, _generate_pink_noise

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "clean"
NUM_SYNTHETIC = 493
TARGET_PEAK = 0.3


def generate_synthetic_clean_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0

    for i in range(NUM_SYNTHETIC):
        out_path = DATA_DIR / f"synthetic_clean_{i:03d}.wav"
        if out_path.exists():
            skipped += 1
            continue

        pink = _generate_pink_noise(N_SAMPLES)
        peak = np.max(np.abs(pink))
        if peak > 0:
            pink = pink / peak * TARGET_PEAK

        sf.write(out_path, pink.astype(np.float32), SAMPLE_RATE)
        created += 1

    total = len(list(DATA_DIR.glob("*.wav")))
    print(f"Created {created} new files, skipped {skipped} existing.")
    print(f"Total WAV files in {DATA_DIR}: {total}")


if __name__ == "__main__":
    generate_synthetic_clean_files()
