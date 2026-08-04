"""Advanced desk-bound suite: robustness, hierarchical/cost-sensitive, calibration, SSL.

  python -m experiments.run_advanced --smoke
  python -m experiments.run_advanced --out experiments/outputs/advanced
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.calibration import calibration_report  # noqa: E402
from experiments.common import CLASS_NAMES, REFERENCE_PAPER_VAL_ACC  # noqa: E402
from experiments.data_utils import (  # noqa: E402
    collect_wav_paths,
    dedupe_by_file_bytes,
    make_split,
    write_split_manifest,
)
from experiments.features import build_feature_matrices, extract_mel, load_waveform  # noqa: E402
from experiments.hierarchical import train_cost_sensitive_cnn, train_hierarchical_cnn  # noqa: E402
from experiments.models import (  # noqa: E402
    _make_spec_augment_sequence,
    build_cnn_deep,
    tensorflow_available,
)
from experiments.robustness import apply_rung, ladder_specs  # noqa: E402
from experiments.ssl_pretrain import run_ssl_pipeline  # noqa: E402
from model.preprocess import waveform_to_mel_spectrogram  # noqa: E402


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    if tensorflow_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


def _train_baseline(X_tr, y_tr, X_va, y_va, epochs: int, smoke: bool):
    from tensorflow import keras

    model = build_cnn_deep(n_classes=4)
    fit_epochs = 3 if smoke else epochs
    bs = min(32, max(4, len(X_tr)))
    y_tr_oh = keras.utils.to_categorical(y_tr, 4)
    y_va_oh = keras.utils.to_categorical(y_va, 4)
    seq = _make_spec_augment_sequence(keras, X_tr, y_tr_oh, batch_size=bs)
    steps = max(1, len(seq))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(1e-3, max(1, steps * fit_epochs), alpha=1e-5)
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    model.fit(
        seq,
        validation_data=(X_va, y_va_oh),
        epochs=fit_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=3 if smoke else 15, restore_best_weights=True, monitor="val_accuracy", mode="max"
            )
        ],
        verbose=1,
    )
    probs = model.predict(X_va, verbose=0)
    pred = probs.argmax(1)
    return {
        "approach_id": "cnn_deep_baseline",
        "accuracy": float(accuracy_score(y_va, pred)),
        "macro_f1": float(f1_score(y_va, pred, average="macro", zero_division=0)),
        "probs": probs.astype(np.float32),
        "y_pred": pred.astype(np.int32),
        "model": model,
    }


def _eval_robustness(model, paths_val, y_val, seed: int, smoke: bool) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    # limit files in smoke
    n = min(len(paths_val), 24 if smoke else len(paths_val))
    paths = list(paths_val[:n])
    labels = np.asarray(y_val[:n])
    for rung_id, desc in ladder_specs():
        mels = []
        for i, p in enumerate(paths):
            y = load_waveform(str(p), class_hint=int(labels[i]), smoke_seed=seed)
            y_d = apply_rung(rung_id, y, rng)
            mels.append(waveform_to_mel_spectrogram(y_d))
        X = np.stack(mels).astype(np.float32)
        probs = model.predict(X, verbose=0)
        pred = probs.argmax(1)
        rows.append(
            {
                "rung": rung_id,
                "description": desc,
                "n": int(len(labels)),
                "accuracy": float(accuracy_score(labels, pred)),
                "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            }
        )
        print(f"  [robust] {rung_id:12s} acc={rows[-1]['accuracy']:.4f}", flush=True)
    return {"rungs": rows, "seed": seed}


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k not in {"model", "models", "probs"}}
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    return obj


def run(args: argparse.Namespace) -> dict:
    if not tensorflow_available() and not args.smoke:
        raise SystemExit("TensorFlow required")
    _set_seeds(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if args.smoke:
        rng = np.random.default_rng(args.seed)
        n_tr, n_va = 48, 24
        X_hand_tr = rng.normal(size=(n_tr, 74)).astype(np.float32)
        X_mel_tr = rng.random((n_tr, 128, 128, 1), dtype=np.float32)
        X_mel_va = rng.random((n_va, 128, 128, 1), dtype=np.float32)
        y_tr = rng.integers(0, 4, n_tr).astype(np.int32)
        y_va = rng.integers(0, 4, n_va).astype(np.int32)
        # fake paths for robustness using smoke: waveforms
        paths_va = np.array([f"smoke:{int(y_va[i])}:{i}" for i in range(n_va)], dtype=object)
        X_pretrain = np.concatenate([X_mel_tr, X_mel_va], axis=0)
        dedupe = {"smoke": True}
    else:
        paths, labels = collect_wav_paths()
        if len(paths) == 0:
            raise SystemExit("No WAVs — run prepare_kaggle_data first")
        paths, labels, dedupe = dedupe_by_file_bytes(paths, labels)
        split = make_split(paths, labels, test_size=0.2, seed=args.seed)
        write_split_manifest(split, out / "split_manifest.json")
        print("Extracting features…", flush=True)
        cache = out / f"features_seed{args.seed}.npz"
        if cache.exists() and not args.no_cache:
            z = np.load(cache)
            X_mel_tr, X_mel_va = z["X_mel_train"], z["X_mel_val"]
            y_tr, y_va = z["y_train"], z["y_val"]
            paths_va = split["val_paths"]
        else:
            _, X_mel_tr = build_feature_matrices(split["train_paths"], labels=split["y_train"], smoke_seed=args.seed)
            _, X_mel_va = build_feature_matrices(split["val_paths"], labels=split["y_val"], smoke_seed=args.seed)
            y_tr, y_va = split["y_train"], split["y_val"]
            paths_va = split["val_paths"]
            np.savez_compressed(
                cache,
                X_mel_train=X_mel_tr,
                X_mel_val=X_mel_va,
                y_train=y_tr,
                y_val=y_va,
            )
        X_pretrain = np.concatenate([X_mel_tr, X_mel_va], axis=0)

    summary: dict = {
        "seed": args.seed,
        "smoke": args.smoke,
        "reference_acc": REFERENCE_PAPER_VAL_ACC,
        "dedupe": dedupe,
        "classes": CLASS_NAMES,
    }

    print("=== Baseline cnn_deep ===", flush=True)
    baseline = _train_baseline(X_mel_tr, y_tr, X_mel_va, y_va, args.epochs, args.smoke)
    summary["baseline"] = {
        "accuracy": baseline["accuracy"],
        "macro_f1": baseline["macro_f1"],
    }
    print(f"  baseline val_acc={baseline['accuracy']:.4f}", flush=True)

    print("=== Robustness ladder ===", flush=True)
    robust = _eval_robustness(baseline["model"], paths_va, y_va, args.seed, args.smoke)
    summary["robustness"] = robust
    (out / "robustness.json").write_text(json.dumps(robust, indent=2) + "\n")

    print("=== Cost-sensitive weevil/borer ===", flush=True)
    cost = train_cost_sensitive_cnn(
        X_mel_tr, y_tr, X_mel_va, y_va, epochs=args.epochs, smoke=args.smoke, pair_boost=args.pair_boost
    )
    summary["cost_sensitive"] = _jsonable(cost)
    (out / "cost_sensitive.json").write_text(json.dumps(_jsonable(cost), indent=2) + "\n")
    print(f"  cost-sensitive acc={cost.get('accuracy')} swaps={cost.get('weevil_borer_swap_count')}", flush=True)

    print("=== Hierarchical weevil/borer ===", flush=True)
    hier = train_hierarchical_cnn(X_mel_tr, y_tr, X_mel_va, y_va, epochs=args.epochs, smoke=args.smoke)
    summary["hierarchical"] = _jsonable(hier)
    (out / "hierarchical.json").write_text(json.dumps(_jsonable(hier), indent=2) + "\n")
    print(
        f"  hierarchical acc={hier.get('accuracy')} pair_subset={hier.get('pair_subset_accuracy')} "
        f"swaps={hier.get('weevil_borer_swap_count')}",
        flush=True,
    )

    print("=== Calibration + abstain (baseline) ===", flush=True)
    cal = calibration_report(y_va, baseline["probs"], treat_as_logits=False, seed=args.seed)
    summary["calibration_baseline"] = cal
    (out / "calibration_baseline.json").write_text(json.dumps(cal, indent=2) + "\n")
    if not cost.get("skipped") and "probs" in cost:
        cal_c = calibration_report(y_va, cost["probs"], seed=args.seed)
        summary["calibration_cost_sensitive"] = cal_c
        (out / "calibration_cost_sensitive.json").write_text(json.dumps(cal_c, indent=2) + "\n")

    print("=== SSL SimCLR pretrain → fine-tune ===", flush=True)
    ssl = run_ssl_pipeline(
        X_pretrain,
        X_mel_tr,
        y_tr,
        X_mel_va,
        y_va,
        pretrain_epochs=args.ssl_pretrain_epochs,
        finetune_epochs=args.epochs,
        smoke=args.smoke,
    )
    summary["ssl"] = _jsonable(ssl)
    (out / "ssl.json").write_text(json.dumps(_jsonable(ssl), indent=2) + "\n")
    print(f"  ssl finetune acc={ssl.get('accuracy')}", flush=True)

    if not ssl.get("skipped") and "probs" in ssl:
        cal_s = calibration_report(y_va, ssl["probs"], seed=args.seed)
        summary["calibration_ssl"] = cal_s

    summary["elapsed_sec"] = time.perf_counter() - t0
    (out / "advanced_summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")

    md = [
        "# Advanced suite report",
        "",
        f"- Seed: `{args.seed}` smoke={args.smoke}",
        f"- Baseline cnn_deep: **{baseline['accuracy']:.2%}** (macro F1 {baseline['macro_f1']:.4f})",
        f"- Cost-sensitive: **{cost.get('accuracy', 0):.2%}** (weevil↔borer swaps={cost.get('weevil_borer_swap_count')})",
        f"- Hierarchical: **{hier.get('accuracy', 0):.2%}** (pair subset={hier.get('pair_subset_accuracy')})",
        f"- SSL fine-tune: **{ssl.get('accuracy', 0):.2%}**",
        f"- Baseline ECE before→after T: "
        f"{cal['before']['ece']:.4f}→{cal['after']['ece']:.4f} (T={cal['after']['temperature']:.3f})",
        "",
        "## Robustness ladder (baseline)",
        "",
        "| Rung | Acc | Macro-F1 |",
        "|---|---:|---:|",
    ]
    for r in robust["rungs"]:
        md.append(f"| `{r['rung']}` | {r['accuracy']:.2%} | {r['macro_f1']:.4f} |")
    md.extend(["", f"Elapsed: {summary['elapsed_sec']:.1f}s", ""])
    (out / "advanced_report.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)

    # Copy into committed results when not smoke
    if not args.smoke and args.copy_results:
        dest = ROOT / "experiments" / "results" / "advanced"
        dest.mkdir(parents=True, exist_ok=True)
        for name in (
            "advanced_summary.json",
            "advanced_report.md",
            "robustness.json",
            "cost_sensitive.json",
            "hierarchical.json",
            "calibration_baseline.json",
            "ssl.json",
        ):
            src = out / name
            if src.exists():
                (dest / name).write_bytes(src.read_bytes())
        print(f"Copied artifacts → {dest}", flush=True)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--ssl-pretrain-epochs", type=int, default=30)
    p.add_argument("--pair-boost", type=float, default=1.8)
    p.add_argument("--out", type=str, default=str(ROOT / "experiments" / "outputs" / "advanced"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--copy-results", action="store_true", help="Copy JSON/MD into experiments/results/advanced")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
