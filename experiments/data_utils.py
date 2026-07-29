"""Load WAV paths and stratified file-level splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.common import CLASS_NAMES, DATA_SUBDIRS


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Prefer PROJECT_KAAN_DATA_DIR / EXPERIMENTS_DATA_DIR when set (Kaggle)."""
    import os

    override = os.environ.get("PROJECT_KAAN_DATA_DIR") or os.environ.get("EXPERIMENTS_DATA_DIR")
    if override:
        return Path(override)
    return project_root() / "data"


def collect_wav_paths(root: Path | None = None) -> Tuple[np.ndarray, np.ndarray]:
    root = root or data_dir()
    paths: list[str] = []
    labels: list[int] = []
    for class_idx, class_name in enumerate(DATA_SUBDIRS):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for wav in sorted(class_dir.glob("*.wav")):
            paths.append(str(wav.resolve()))
            labels.append(class_idx)
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int32)


def _file_bytes_hash(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.md5(p.read_bytes()).hexdigest()


def dedupe_by_file_bytes(
    paths: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep one path per unique WAV byte content. Deterministic by sorted path order."""
    order = np.argsort(paths.astype(str))
    seen: set[str] = set()
    keep_idx: list[int] = []
    removed: list[dict] = []
    for i in order:
        path = str(paths[i])
        h = _file_bytes_hash(path)
        if h is None:
            keep_idx.append(int(i))
            continue
        if h in seen:
            removed.append({"path": path, "label": int(labels[i]), "hash": h})
            continue
        seen.add(h)
        keep_idx.append(int(i))
    keep_idx = sorted(keep_idx)
    info = {
        "n_before": int(len(paths)),
        "n_after": int(len(keep_idx)),
        "n_removed": int(len(removed)),
        "removed_sample": removed[:10],
    }
    return paths[keep_idx], labels[keep_idx], info


def make_split(
    paths: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    train_paths, val_paths, y_train, y_val = train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return {
        "train_paths": train_paths,
        "val_paths": val_paths,
        "y_train": y_train,
        "y_val": y_val,
        "seed": seed,
        "test_size": test_size,
    }


def write_split_manifest(split: dict, out_path: Path) -> None:
    manifest = {
        "seed": split["seed"],
        "test_size": split["test_size"],
        "classes": CLASS_NAMES,
        "train": [
            {"path": p, "label": CLASS_NAMES[int(y)]}
            for p, y in zip(split["train_paths"], split["y_train"])
        ],
        "val": [
            {"path": p, "label": CLASS_NAMES[int(y)]}
            for p, y in zip(split["val_paths"], split["y_val"])
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def generate_smoke_paths(n_per_class: int = 24, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic WAV paths are not files; we use in-memory generation in smoke mode."""
    del seed  # seed applied later in make_split / feature synth
    paths = []
    labels = []
    for class_idx in range(len(CLASS_NAMES)):
        for i in range(n_per_class):
            paths.append(f"smoke:{class_idx}:{i}")
            labels.append(class_idx)
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int32)
