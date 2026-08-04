"""Distill bake-off teachers into the production mel-CNN.

Teachers (default): gbdt + extratrees (handcrafted) + cnn_deep (mel).
Student: deep production CNN (cnn_deep / train_kaggle capacity) trained with
hard-label CE + soft-label KL against the mean teacher probability.

Usage:
  python -m experiments.prepare_kaggle_data --out .
  python -m model.distill
  python -m model.distill --smoke
  python -m model.export_deploy
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import CLASS_NAMES  # noqa: E402
from experiments.data_utils import (  # noqa: E402
    collect_wav_paths,
    dedupe_by_file_bytes,
    make_split,
    write_split_manifest,
)
from experiments.features import build_feature_matrices  # noqa: E402
from experiments.models import (  # noqa: E402
    _make_spec_augment_sequence,
    build_cnn_deep,
    tensorflow_available,
)

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = MODEL_DIR


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    if tensorflow_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


def _softmax_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _train_classical_teachers(
    X_hand_train: np.ndarray,
    y_train: np.ndarray,
    smoke: bool,
) -> dict[str, object]:
    teachers: dict[str, object] = {}
    gbdt = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=50 if smoke else 200,
        random_state=42,
    )
    gbdt.fit(X_hand_train, y_train)
    teachers["gbdt"] = gbdt

    et = ExtraTreesClassifier(
        n_estimators=80 if smoke else 400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    et.fit(X_hand_train, y_train)
    teachers["extratrees"] = et
    return teachers


def _train_cnn_deep_teacher(
    X_mel_train: np.ndarray,
    y_train: np.ndarray,
    X_mel_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    smoke: bool,
):
    from tensorflow import keras

    model = build_cnn_deep(n_classes=len(CLASS_NAMES))
    fit_epochs = 3 if smoke else epochs
    bs = min(32, max(4, len(X_mel_train)))
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=len(CLASS_NAMES))
    train_data = _make_spec_augment_sequence(keras, X_mel_train, y_train_cat, batch_size=bs)
    steps = max(1, len(train_data))
    lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=max(1, steps * max(fit_epochs, 1)),
        alpha=1e-5,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    cw_arr = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(cw_arr)}
    class_weight[1] *= 1.15
    class_weight[2] *= 1.25
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=3 if smoke else 15,
            restore_best_weights=True,
            monitor="val_loss",
            mode="min",
        )
    ]
    model.fit(
        train_data,
        validation_data=(X_mel_val, y_val_cat),
        epochs=fit_epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return model


def _teacher_probs(
    teachers_classical: dict[str, object],
    cnn_teacher,
    X_hand: np.ndarray,
    X_mel: np.ndarray,
) -> np.ndarray:
    stacks = []
    for name, clf in teachers_classical.items():
        p = clf.predict_proba(X_hand)
        # sklearn predict_proba columns follow classes_, not 0..C-1
        order = list(clf.classes_)
        aligned = np.zeros((len(X_hand), len(CLASS_NAMES)), dtype=np.float64)
        for j, c in enumerate(order):
            aligned[:, int(c)] = p[:, j]
        stacks.append(aligned)
        print(f"  teacher {name}: mean max-prob={aligned.max(axis=1).mean():.3f}")
    if cnn_teacher is not None:
        p = cnn_teacher.predict(X_mel, verbose=0).astype(np.float64)
        stacks.append(p)
        print(f"  teacher cnn_deep: mean max-prob={p.max(axis=1).mean():.3f}")
    soft = np.mean(np.stack(stacks, axis=0), axis=0)
    soft = soft / soft.sum(axis=1, keepdims=True).clip(min=1e-12)
    return soft.astype(np.float32)


def _build_student_softmax():
    """Deployable student matches production deep CNN capacity."""
    return build_cnn_deep(n_classes=len(CLASS_NAMES))


def _distill_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    soft_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    temperature: float,
    alpha: float,
    epochs: int,
    smoke: bool,
):
    """Train student with mixed hard + temperature-softened teacher targets via model.fit.

    y_mix = alpha * one_hot + (1-alpha) * normalize(teacher_probs ** (1/T))
    Uses the same SpecAugment path as the hard-only CNN (stable on TF/Keras 3 + GPU).
    """
    from tensorflow import keras

    fit_epochs = 3 if smoke else epochs
    bs = min(32, max(4, len(X_train)))
    n_classes = len(CLASS_NAMES)
    y_hard = keras.utils.to_categorical(y_train, num_classes=n_classes).astype(np.float32)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=n_classes).astype(np.float32)

    soft = np.clip(soft_train.astype(np.float64), 1e-8, 1.0)
    soft = np.power(soft, 1.0 / max(temperature, 1e-6))
    soft = soft / soft.sum(axis=1, keepdims=True)
    y_mix = (alpha * y_hard + (1.0 - alpha) * soft.astype(np.float32)).astype(np.float32)
    y_mix = y_mix / y_mix.sum(axis=1, keepdims=True).clip(min=1e-12)

    student = _build_student_softmax()
    train_data = _make_spec_augment_sequence(keras, X_train, y_mix, batch_size=bs)
    steps = max(1, len(train_data))
    lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=max(1, steps * max(fit_epochs, 1)),
        alpha=1e-5,
    )
    student.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=3 if smoke else 15,
            restore_best_weights=True,
            monitor="val_accuracy",
            mode="max",
        )
    ]
    hist = student.fit(
        train_data,
        validation_data=(X_val, y_val_cat),
        epochs=fit_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    history = []
    for i in range(len(hist.history.get("loss", []))):
        history.append(
            {
                "epoch": i + 1,
                "loss": float(hist.history["loss"][i]),
                "train_acc": float(hist.history.get("accuracy", [0])[i]),
                "val_acc": float(hist.history.get("val_accuracy", [0])[i]),
            }
        )
    best_val = float(max(hist.history.get("val_accuracy", [0.0])))
    return student, history, best_val


def run_distill(args: argparse.Namespace) -> dict:
    if not tensorflow_available():
        raise SystemExit("TensorFlow is required for distillation.")

    _set_seeds(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = out_dir / "distill_artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        n_train, n_val = 64, 32
        X_hand_train = rng.normal(size=(n_train, 74)).astype(np.float32)
        X_hand_val = rng.normal(size=(n_val, 74)).astype(np.float32)
        X_mel_train = rng.random((n_train, 128, 128, 1), dtype=np.float32)
        X_mel_val = rng.random((n_val, 128, 128, 1), dtype=np.float32)
        y_train = rng.integers(0, 4, size=n_train).astype(np.int32)
        y_val = rng.integers(0, 4, size=n_val).astype(np.int32)
        dedupe_info = {"smoke": True}
    else:
        paths, labels = collect_wav_paths()
        if len(paths) == 0:
            raise SystemExit(
                "No WAVs under data/. Run: python -m experiments.prepare_kaggle_data --out ."
            )
        paths, labels, dedupe_info = dedupe_by_file_bytes(paths, labels)
        split = make_split(paths, labels, test_size=0.2, seed=args.seed)
        write_split_manifest(split, report_dir / "split_manifest.json")
        print(
            f"Split seed={args.seed}: train={len(split['y_train'])} val={len(split['y_val'])} "
            f"(deduped {dedupe_info.get('n_before')}→{dedupe_info.get('n_after')})",
            flush=True,
        )
        cache_path = report_dir / f"features_seed{args.seed}.npz"
        if cache_path.exists() and not args.no_cache:
            print(f"Loading feature cache {cache_path}…", flush=True)
            z = np.load(cache_path, allow_pickle=False)
            X_hand_train = z["X_hand_train"]
            X_mel_train = z["X_mel_train"]
            X_hand_val = z["X_hand_val"]
            X_mel_val = z["X_mel_val"]
            y_train = z["y_train"]
            y_val = z["y_val"]
        else:
            print("Extracting features (cached for reuse)…", flush=True)
            X_hand_train, X_mel_train = build_feature_matrices(
                split["train_paths"], labels=split["y_train"], smoke_seed=args.seed
            )
            X_hand_val, X_mel_val = build_feature_matrices(
                split["val_paths"], labels=split["y_val"], smoke_seed=args.seed
            )
            y_train = split["y_train"]
            y_val = split["y_val"]
            np.savez_compressed(
                cache_path,
                X_hand_train=X_hand_train,
                X_mel_train=X_mel_train,
                X_hand_val=X_hand_val,
                X_mel_val=X_mel_val,
                y_train=y_train,
                y_val=y_val,
            )
            print(f"Wrote {cache_path}", flush=True)

    print("Training classical teachers…", flush=True)
    classical = _train_classical_teachers(X_hand_train, y_train, smoke=args.smoke)
    for name, clf in classical.items():
        pred = clf.predict(X_hand_val)
        print(
            f"  {name} val_acc={accuracy_score(y_val, pred):.4f} "
            f"macro_f1={f1_score(y_val, pred, average='macro'):.4f}",
            flush=True,
        )

    cnn_teacher = None
    if "cnn_deep" in args.teachers:
        print("Training cnn_deep teacher…", flush=True)
        cnn_teacher = _train_cnn_deep_teacher(
            X_mel_train, y_train, X_mel_val, y_val, epochs=args.teacher_epochs, smoke=args.smoke
        )
        pred = np.argmax(cnn_teacher.predict(X_mel_val, verbose=0), axis=1)
        print(
            f"  cnn_deep val_acc={accuracy_score(y_val, pred):.4f} "
            f"macro_f1={f1_score(y_val, pred, average='macro'):.4f}",
            flush=True,
        )
        cnn_teacher.save(report_dir / "teacher_cnn_deep.h5")

    print("Building soft labels…", flush=True)
    soft_train = _teacher_probs(classical, cnn_teacher, X_hand_train, X_mel_train)
    soft_val = _teacher_probs(classical, cnn_teacher, X_hand_val, X_mel_val)
    ensemble_pred = np.argmax(soft_val, axis=1)
    ens_acc = float(accuracy_score(y_val, ensemble_pred))
    ens_f1 = float(f1_score(y_val, ensemble_pred, average="macro"))
    print(f"  ensemble soft val_acc={ens_acc:.4f} macro_f1={ens_f1:.4f}", flush=True)

    print("Training hard-only shallow baseline…", flush=True)
    from tensorflow import keras

    baseline = _build_student_softmax()
    fit_epochs = 3 if args.smoke else args.student_epochs
    bs = min(32, max(4, len(X_mel_train)))
    y_tr_cat = keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
    y_va_cat = keras.utils.to_categorical(y_val, num_classes=len(CLASS_NAMES))
    train_seq = _make_spec_augment_sequence(keras, X_mel_train, y_tr_cat, batch_size=bs)
    baseline.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    baseline.fit(
        train_seq,
        validation_data=(X_mel_val, y_va_cat),
        epochs=fit_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=3 if args.smoke else 12,
                restore_best_weights=True,
                monitor="val_accuracy",
                mode="max",
            )
        ],
        verbose=1,
    )
    base_pred = np.argmax(baseline.predict(X_mel_val, verbose=0), axis=1)
    base_acc = float(accuracy_score(y_val, base_pred))
    base_f1 = float(f1_score(y_val, base_pred, average="macro"))
    print(f"  hard-only deep val_acc={base_acc:.4f} macro_f1={base_f1:.4f}", flush=True)

    print("Distilling into production student…", flush=True)
    deploy, history, best_val = _distill_train(
        X_mel_train,
        y_train,
        soft_train,
        X_mel_val,
        y_val,
        temperature=args.temperature,
        alpha=args.alpha,
        epochs=args.student_epochs,
        smoke=args.smoke,
    )
    dist_pred = np.argmax(deploy.predict(X_mel_val, verbose=0), axis=1)
    dist_acc = float(accuracy_score(y_val, dist_pred))
    dist_f1 = float(f1_score(y_val, dist_pred, average="macro"))
    print(f"  distilled student val_acc={dist_acc:.4f} macro_f1={dist_f1:.4f}", flush=True)
    print(classification_report(y_val, dist_pred, target_names=CLASS_NAMES, digits=4))

    ship = deploy
    shipped = "distilled"
    if dist_acc + 1e-6 < base_acc - 0.005:
        print(
            f"Distilled underperformed hard-only ({dist_acc:.4%} < {base_acc:.4%}−0.5%); "
            "shipping hard-only CNN instead.",
            flush=True,
        )
        ship = baseline
        shipped = "hard_only_fallback"
        dist_acc_report = dist_acc
        dist_f1_report = dist_f1
    else:
        dist_acc_report = dist_acc
        dist_f1_report = dist_f1

    h5_path = out_dir / ("project-kaan_model.smoke.h5" if args.smoke else "project-kaan_model.h5")
    if h5_path.exists() and not args.smoke:
        backup = out_dir / "project-kaan_model.pre_distill.h5"
        shutil.copy2(h5_path, backup)
        print(f"Backed up previous H5 → {backup}", flush=True)
    ship.save(h5_path)
    print(f"Saved ({shipped}) → {h5_path}", flush=True)

    report = {
        "seed": args.seed,
        "smoke": args.smoke,
        "teachers": args.teachers,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "ensemble_val_acc": ens_acc,
        "ensemble_val_macro_f1": ens_f1,
        "hard_only_deep_val_acc": base_acc,
        "hard_only_deep_val_macro_f1": base_f1,
        "hard_only_shallow_val_acc": base_acc,
        "hard_only_shallow_val_macro_f1": base_f1,
        "distilled_val_acc": dist_acc_report,
        "distilled_val_macro_f1": dist_f1_report,
        "shipped": shipped,
        "shipped_val_acc": float(accuracy_score(y_val, np.argmax(ship.predict(X_mel_val, verbose=0), axis=1))),
        "best_monitored_val_acc": best_val,
        "n_params": int(ship.count_params()),
        "dedupe": dedupe_info,
        "elapsed_sec": time.perf_counter() - t0,
        "history": history,
        "h5": str(h5_path),
    }
    (report_dir / "distill_report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# Distillation report",
        "",
        f"- Seed: `{args.seed}`",
        f"- Teachers: `{', '.join(args.teachers)}`",
        f"- T={args.temperature}, alpha(hard)={args.alpha}",
        f"- Ensemble val acc: **{ens_acc:.4%}** (macro F1 {ens_f1:.4f})",
        f"- Hard-only deep val acc: **{base_acc:.4%}** (macro F1 {base_f1:.4f})",
        f"- Distilled student val acc: **{dist_acc_report:.4%}** (macro F1 {dist_f1_report:.4f})",
        f"- Shipped: **{shipped}** (val acc {report['shipped_val_acc']:.4%})",
        f"- Params: {report['n_params']}",
        f"- Elapsed: {report['elapsed_sec']:.1f}s",
        "",
        "Next: `python -m model.export_deploy`",
        "",
    ]
    (report_dir / "distill_report.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description="Distill teachers into production mel-CNN")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Tiny synthetic run")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5, help="Hard-label mix weight")
    p.add_argument("--teacher-epochs", type=int, default=60)
    p.add_argument("--student-epochs", type=int, default=60)
    p.add_argument(
        "--teachers",
        type=str,
        default="gbdt,extratrees,cnn_deep",
        help="Comma list; cnn_deep optional",
    )
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--no-cache", action="store_true", help="Ignore feature cache")
    args = p.parse_args()
    args.teachers = [t.strip() for t in args.teachers.split(",") if t.strip()]
    run_distill(args)


if __name__ == "__main__":
    main()
