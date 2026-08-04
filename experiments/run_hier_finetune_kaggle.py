"""Lean Kaggle job: baseline CNN + fine-tuned hierarchical weevil↔borer head."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
DATA_ROOT = TEMP / "kaan_data"
OUT = WORK / "hier_finetune"


def _pip() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "librosa", "soundfile", "scikit-learn", "scipy"],
        check=True,
    )
    try:
        import tensorflow as tf

        print("TF", tf.__version__, tf.config.list_physical_devices("GPU"), flush=True)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorflow"], check=True)


def main() -> None:
    _pip()
    src = Path("/kaggle/temp/_kaan_src")
    sys.path.insert(0, str(src))
    os.environ["PROJECT_KAAN_DATA_DIR"] = str(DATA_ROOT / "data")
    os.environ["EXPERIMENTS_DATA_DIR"] = str(DATA_ROOT / "data")

    if os.environ.get("SKIP_PREPARE") != "1":
        subprocess.run(
            [sys.executable, "-m", "experiments.prepare_kaggle_data", "--out", str(DATA_ROOT)],
            check=True,
            cwd=str(src),
            env={**os.environ, "PYTHONPATH": str(src)},
        )

    from experiments.data_utils import collect_wav_paths, dedupe_by_file_bytes, make_split
    from experiments.features import build_feature_matrices
    from experiments.hierarchical import train_hierarchical_cnn
    from experiments.models import _make_spec_augment_sequence, build_cnn_deep
    from sklearn.metrics import accuracy_score, f1_score
    from tensorflow import keras

    seed = int(os.environ.get("SEED", "42"))
    epochs = int(os.environ.get("EPOCHS", "60"))
    np.random.seed(seed)

    paths, labels = collect_wav_paths()
    paths, labels, _ = dedupe_by_file_bytes(paths, labels)
    split = make_split(paths, labels, test_size=0.2, seed=seed)
    _, X_tr = build_feature_matrices(split["train_paths"], labels=split["y_train"], smoke_seed=seed)
    _, X_va = build_feature_matrices(split["val_paths"], labels=split["y_val"], smoke_seed=seed)
    y_tr, y_va = split["y_train"], split["y_val"]

    # Baseline
    model = build_cnn_deep(4)
    bs = min(32, max(4, len(X_tr)))
    y_tr_oh = keras.utils.to_categorical(y_tr, 4)
    y_va_oh = keras.utils.to_categorical(y_va, 4)
    seq = _make_spec_augment_sequence(keras, X_tr, y_tr_oh, batch_size=bs)
    steps = max(1, len(seq))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(1e-3, max(1, steps * epochs), alpha=1e-5)
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    model.fit(
        seq,
        validation_data=(X_va, y_va_oh),
        epochs=epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=15, restore_best_weights=True, monitor="val_accuracy", mode="max"
            )
        ],
        verbose=1,
    )
    base_p = model.predict(X_va, verbose=0)
    base_acc = float(accuracy_score(y_va, base_p.argmax(1)))
    print(f"baseline acc={base_acc:.4f}", flush=True)

    hier = train_hierarchical_cnn(
        X_tr, y_tr, X_va, y_va, epochs=epochs, smoke=False, base_model=model
    )
    scratch = train_hierarchical_cnn(
        X_tr, y_tr, X_va, y_va, epochs=epochs, smoke=False, base_model=None
    )

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline_acc": base_acc,
        "baseline_macro_f1": float(f1_score(y_va, base_p.argmax(1), average="macro")),
        "hierarchical_finetuned": {
            k: hier.get(k)
            for k in (
                "accuracy",
                "macro_f1",
                "weevil_borer_swap_count",
                "pair_subset_accuracy",
                "pair_specialist_alone_acc",
                "n_pair_overrides",
                "confusion",
            )
        },
        "hierarchical_scratch": {
            k: scratch.get(k)
            for k in (
                "accuracy",
                "macro_f1",
                "weevil_borer_swap_count",
                "pair_subset_accuracy",
                "confusion",
            )
        },
    }
    (OUT / "hier_finetune_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (WORK / "hier_finetune_report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# Hierarchical fine-tune report",
        "",
        f"- Baseline: **{base_acc:.2%}**",
        f"- Fine-tuned hierarchy: **{hier.get('accuracy', 0):.2%}** "
        f"(swaps={hier.get('weevil_borer_swap_count')}, pair={hier.get('pair_subset_accuracy')}, "
        f"specialist alone={hier.get('pair_specialist_alone_acc')})",
        f"- Scratch hierarchy: **{scratch.get('accuracy', 0):.2%}** "
        f"(swaps={scratch.get('weevil_borer_swap_count')})",
        "",
    ]
    (WORK / "hier_finetune_report.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)
    print("[hier-ft] DONE", flush=True)


if __name__ == "__main__":
    main()
