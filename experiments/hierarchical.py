"""Weevil ↔ lesser-grain-borer focused heads: cost-sensitive + hierarchical."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from experiments.common import CLASS_NAMES
from experiments.models import _make_spec_augment_sequence, build_cnn_deep, tensorflow_available

# Indices: clean=0, rice_weevil=1, lesser_grain_borer=2, red_flour_beetle=3
WEEVIL, BORER = 1, 2


def _class_weight_matrix(y_train: np.ndarray, pair_boost: float = 1.8) -> dict[int, float]:
    cw = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=y_train)
    weights = {i: float(w) for i, w in enumerate(cw)}
    weights[WEEVIL] *= pair_boost
    weights[BORER] *= pair_boost
    return weights


def train_cost_sensitive_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 60,
    smoke: bool = False,
    pair_boost: float = 1.8,
) -> dict[str, Any]:
    if not tensorflow_available():
        return {"skipped": True, "reason": "no tensorflow"}
    from tensorflow import keras

    model = build_cnn_deep(n_classes=len(CLASS_NAMES))
    fit_epochs = 3 if smoke else epochs
    bs = min(32, max(4, len(X_train)))
    y_tr = keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
    y_va = keras.utils.to_categorical(y_val, num_classes=len(CLASS_NAMES))
    train_data = _make_spec_augment_sequence(keras, X_train, y_tr, batch_size=bs)
    steps = max(1, len(train_data))
    lr = keras.optimizers.schedules.CosineDecay(1e-3, max(1, steps * fit_epochs), alpha=1e-5)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    cw = _class_weight_matrix(y_train, pair_boost=pair_boost)
    model.fit(
        train_data,
        validation_data=(X_val, y_va),
        epochs=fit_epochs,
        class_weight=cw,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=3 if smoke else 15, restore_best_weights=True, monitor="val_accuracy", mode="max"
            )
        ],
        verbose=1,
    )
    probs = model.predict(X_val, verbose=0)
    pred = probs.argmax(1)
    return _pack("cost_sensitive_pair", y_val, pred, probs, model, cw)


def train_hierarchical_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 60,
    smoke: bool = False,
) -> dict[str, Any]:
    """Stage A: clean vs pest vs flour_beetle-ish coarse? 

    Practical hierarchy for the known failure mode:
      Stage 1: {clean, pest_internal, red_flour_beetle} where pest_internal = weevil∪borer
      Stage 2: weevil vs borer on pest_internal only
    Final: map back to 4-class.
    """
    if not tensorflow_available():
        return {"skipped": True, "reason": "no tensorflow"}
    from tensorflow import keras

    fit_epochs = 3 if smoke else epochs
    bs = min(32, max(4, len(X_train)))

    def to_coarse(y: np.ndarray) -> np.ndarray:
        # 0 clean, 1 internal (weevil|borer), 2 red_flour_beetle
        out = np.zeros_like(y)
        out[y == 0] = 0
        out[(y == WEEVIL) | (y == BORER)] = 1
        out[y == 3] = 2
        return out

    y_c_tr, y_c_va = to_coarse(y_train), to_coarse(y_val)
    model_c = build_cnn_deep(n_classes=3)
    y_c_tr_oh = keras.utils.to_categorical(y_c_tr, 3)
    y_c_va_oh = keras.utils.to_categorical(y_c_va, 3)
    seq_c = _make_spec_augment_sequence(keras, X_train, y_c_tr_oh, batch_size=bs)
    steps = max(1, len(seq_c))
    model_c.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(1e-3, max(1, steps * fit_epochs), alpha=1e-5)
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    model_c.fit(
        seq_c,
        validation_data=(X_val, y_c_va_oh),
        epochs=fit_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=3 if smoke else 15, restore_best_weights=True, monitor="val_accuracy", mode="max"
            )
        ],
        verbose=1,
    )

    # Stage 2: only weevil/borer samples
    mask_tr = (y_train == WEEVIL) | (y_train == BORER)
    mask_va = (y_val == WEEVIL) | (y_val == BORER)
    if mask_tr.sum() < 4:
        return {"skipped": True, "reason": "too few weevil/borer train samples"}

    y_p_tr = (y_train[mask_tr] == BORER).astype(np.int32)  # 0 weevil, 1 borer
    X_p_tr = X_train[mask_tr]
    model_p = build_cnn_deep(n_classes=2)
    # smaller head reuse architecture — fine for smoke/full
    y_p_oh = keras.utils.to_categorical(y_p_tr, 2)
    seq_p = _make_spec_augment_sequence(keras, X_p_tr, y_p_oh, batch_size=min(bs, max(4, len(X_p_tr))))
    steps_p = max(1, len(seq_p))
    model_p.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(1e-3, max(1, steps_p * fit_epochs), alpha=1e-5)
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    # val for pair if available
    val_pair = None
    if mask_va.sum() >= 2:
        y_p_va = (y_val[mask_va] == BORER).astype(np.int32)
        val_pair = (X_val[mask_va], keras.utils.to_categorical(y_p_va, 2))
    model_p.fit(
        seq_p,
        validation_data=val_pair,
        epochs=fit_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=3 if smoke else 12,
                restore_best_weights=True,
                monitor="val_accuracy" if val_pair else "loss",
                mode="max" if val_pair else "min",
            )
        ],
        verbose=1,
    )

    # Compose predictions
    coarse_p = model_c.predict(X_val, verbose=0)
    coarse = coarse_p.argmax(1)
    pair_p = model_p.predict(X_val, verbose=0)
    pair = pair_p.argmax(1)  # 0 weevil, 1 borer

    pred = np.zeros(len(y_val), dtype=np.int32)
    for i in range(len(y_val)):
        c = int(coarse[i])
        if c == 0:
            pred[i] = 0
        elif c == 2:
            pred[i] = 3
        else:
            pred[i] = BORER if pair[i] == 1 else WEEVIL

    # Soft 4-class probs for calibration: approximate
    probs = np.zeros((len(y_val), 4), dtype=np.float64)
    for i in range(len(y_val)):
        probs[i, 0] = coarse_p[i, 0]
        probs[i, 3] = coarse_p[i, 2]
        probs[i, WEEVIL] = coarse_p[i, 1] * pair_p[i, 0]
        probs[i, BORER] = coarse_p[i, 1] * pair_p[i, 1]
    probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-12)

    out = _pack("hierarchical_weevil_borer", y_val, pred, probs, None, None)
    out["models"] = {"coarse": model_c, "pair": model_p}
    # pair-only accuracy when true is weevil or borer
    if mask_va.sum():
        yt = y_val[mask_va]
        yp = pred[mask_va]
        out["pair_subset_accuracy"] = float(accuracy_score(yt, yp))
        out["pair_confusion"] = confusion_matrix(yt, yp, labels=[WEEVIL, BORER]).tolist()
    return out


def _pack(name: str, y_true, y_pred, probs, model, extra) -> dict[str, Any]:
    per = f1_score(y_true, y_pred, average=None, labels=list(range(4)), zero_division=0)
    # weevil↔borer swap count
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    swap = int(cm[WEEVIL, BORER] + cm[BORER, WEEVIL])
    return {
        "approach_id": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {CLASS_NAMES[i]: float(per[i]) for i in range(4)},
        "confusion": cm.tolist(),
        "weevil_borer_swap_count": swap,
        "probs": np.asarray(probs, dtype=np.float32),
        "y_pred": np.asarray(y_pred, dtype=np.int32),
        "y_true": np.asarray(y_true, dtype=np.int32),
        "class_weights": extra,
        "n_params": int(model.count_params()) if model is not None else None,
        "report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4),
        "skipped": False,
        "model": model,
    }
