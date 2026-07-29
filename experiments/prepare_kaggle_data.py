"""Prepare IRRI + clean WAV layout for experiments (local or Kaggle).

Mirrors the data sources used by model/train_kaggle.py:
  1. Clone IRRI Rice-Acoustic-Sensor and copy species WAVs into class folders
  2. Slice Speech Commands _background_noise_ into 10 s clean WAVs

Writes under <out>/data/{clean,rice_weevil,lesser_grain_borer,red_flour_beetle}/
so experiments.collect_wav_paths / run_benchmark can use a file-level split.

Examples:
  python -m experiments.prepare_kaggle_data --out .
  python -m experiments.prepare_kaggle_data --out /kaggle/working/Project-Kaan
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
DURATION_SEC = 10
N_SAMPLES = SAMPLE_RATE * DURATION_SEC

SPECIES_MAP = {
    "S_oryzae": "rice_weevil",
    "R_dominica": "lesser_grain_borer",
    "T_castaneum": "red_flour_beetle",
}

IRRI_REPO = "https://github.com/cbalingbing/Rice-Acoustic-Sensor"
BG_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
BG_FILES = [
    "doing_the_dishes.wav",
    "dude_miaowing.wav",
    "exercise_bike.wav",
    "running_tap.wav",
    "white_noise.wav",
]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def _window_waveform(y: np.ndarray, window_sec: float = 10.0, hop_sec: float = 2.5) -> list[np.ndarray]:
    win = int(window_sec * SAMPLE_RATE)
    hop = int(hop_sec * SAMPLE_RATE)
    windows: list[np.ndarray] = []
    start = 0
    while start + win <= len(y):
        windows.append(y[start : start + win].astype(np.float32))
        start += hop
    if not windows:
        if len(y) >= win:
            windows.append(y[:win].astype(np.float32))
        else:
            pad = win - len(y)
            windows.append(np.pad(y, (0, pad), mode="constant").astype(np.float32))
    return windows


def prepare_pest_wavs(scratch: Path, data_dir: Path) -> dict[str, int]:
    repo_dir = scratch / "Rice-Acoustic-Sensor"
    if not repo_dir.exists():
        print("Cloning IRRI Rice-Acoustic-Sensor…", flush=True)
        _run(["git", "clone", "--depth", "1", IRRI_REPO, str(repo_dir)])
    else:
        print(f"Using existing clone: {repo_dir}", flush=True)

    extract_dir = scratch / "Insect_WAVs_extracted"
    zip_path = repo_dir / "Insect_WAVs.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}")
    if not extract_dir.exists():
        print("Unzipping Insect_WAVs.zip…", flush=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        _run(["unzip", "-q", str(zip_path), "-d", str(extract_dir)])

    wav_root = extract_dir / "Insect_WAVs"
    if not wav_root.is_dir():
        # Some zips flatten differently
        candidates = list(extract_dir.rglob("S_oryzae"))
        if candidates:
            wav_root = candidates[0].parent
        else:
            raise FileNotFoundError(f"Could not find Insect_WAVs under {extract_dir}")

    counts: dict[str, int] = {}
    for species, class_name in SPECIES_MAP.items():
        dst = data_dir / class_name
        dst.mkdir(parents=True, exist_ok=True)
        src_dir = wav_root / species
        n = 0
        for f in sorted(src_dir.glob("*.wav")):
            dst_file = dst / f.name
            if not dst_file.exists():
                dst_file.write_bytes(f.read_bytes())
            n += 1
        counts[class_name] = n
        print(f"  {class_name}: {n} files", flush=True)
    return counts


def prepare_clean_wavs(scratch: Path, data_dir: Path, hop_sec: float = 2.5) -> int:
    """Write 10 s clean windows from Speech Commands background noise (train_kaggle source)."""
    import librosa

    bg_dir = scratch / "background_real"
    bg_src = bg_dir / "_background_noise_"
    if not bg_src.exists():
        bg_dir.mkdir(parents=True, exist_ok=True)
        archive = scratch / "speech_commands_v0.02.tar.gz"
        print("Downloading Speech Commands background noise…", flush=True)
        result = _run(["curl", "-fSL", "-o", str(archive), BG_URL], check=False)
        if result.returncode != 0 or not archive.exists():
            raise RuntimeError(f"Failed to download {BG_URL}")
        for member in ("./_background_noise_", "_background_noise_"):
            _run(["tar", "-xzf", str(archive), "-C", str(bg_dir), member], check=False)
            if bg_src.exists():
                break
        if not bg_src.exists():
            raise RuntimeError("Failed to extract _background_noise_")
        try:
            archive.unlink()
        except OSError:
            pass

    clean_dir = data_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for fname in BG_FILES:
        path = bg_src / fname
        if not path.exists():
            print(f"  skip missing background file: {fname}", flush=True)
            continue
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        windows = _window_waveform(y, window_sec=DURATION_SEC, hop_sec=hop_sec)
        stem = Path(fname).stem
        for i, w in enumerate(windows):
            out = clean_dir / f"bg_{stem}_{i:04d}.wav"
            if not out.exists():
                sf.write(out, w.astype(np.float32), SAMPLE_RATE)
            written += 1
        print(f"  clean from {fname}: {len(windows)} windows", flush=True)
    return written


def prepare_pink_clean(data_dir: Path, n: int = 0, seed: int = 42) -> int:
    """Optional synthetic pink-noise clean files (HOW_TO_GET_DATA.md style)."""
    if n <= 0:
        return 0
    rng = np.random.default_rng(seed)
    clean_dir = data_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for i in range(n):
        out = clean_dir / f"synthetic_clean_{i:03d}.wav"
        if out.exists():
            continue
        white = rng.standard_normal(N_SAMPLES)
        fft = np.fft.rfft(white)
        freqs = np.arange(len(fft)) + 1
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=N_SAMPLES).astype(np.float32)
        peak = float(np.max(np.abs(pink))) + 1e-8
        pink = pink / peak * 0.3
        sf.write(out, pink, SAMPLE_RATE)
        created += 1
    return created


def prepare_all(
    out_root: Path,
    scratch: Path | None = None,
    pink_extra: int = 0,
    hop_sec: float = 2.5,
) -> Path:
    out_root = out_root.resolve()
    data_dir = out_root / "data"
    scratch = (scratch or (out_root / "_scratch")).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    for c in ("clean", "rice_weevil", "lesser_grain_borer", "red_flour_beetle"):
        (data_dir / c).mkdir(parents=True, exist_ok=True)

    print(f"[prepare] out={out_root}", flush=True)
    print(f"[prepare] data={data_dir}", flush=True)
    print(f"[prepare] scratch={scratch}", flush=True)

    print("\n[1/2] Pest WAVs (IRRI)…", flush=True)
    pest_counts = prepare_pest_wavs(scratch, data_dir)

    print("\n[2/2] Clean WAVs (Speech Commands background)…", flush=True)
    n_clean = prepare_clean_wavs(scratch, data_dir, hop_sec=hop_sec)
    n_pink = prepare_pink_clean(data_dir, n=pink_extra)
    if n_pink:
        print(f"  + {n_pink} synthetic pink-noise clean files", flush=True)

    print("\n[prepare] done:", flush=True)
    for name in ("clean", "rice_weevil", "lesser_grain_borer", "red_flour_beetle"):
        n = len(list((data_dir / name).glob("*.wav")))
        print(f"  {name}: {n}", flush=True)
    print(f"  pest source counts: {pest_counts}", flush=True)
    print(f"  clean windows written this run: {n_clean}", flush=True)
    return data_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare IRRI + clean WAVs for experiments")
    p.add_argument("--out", type=str, default=".", help="Project root that will contain data/")
    p.add_argument("--scratch", type=str, default="", help="Scratch dir for clone/downloads")
    p.add_argument("--pink-extra", type=int, default=0, help="Extra synthetic pink-noise clean WAVs")
    p.add_argument("--hop-sec", type=float, default=2.5, help="Hop between clean windows")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    scratch = Path(args.scratch) if args.scratch else None
    prepare_all(Path(args.out), scratch=scratch, pink_extra=args.pink_extra, hop_sec=args.hop_sec)


if __name__ == "__main__":
    main()
