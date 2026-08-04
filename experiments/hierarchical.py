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


def _split_encoder_and_head(model):
    """Return (encoder_to_gap_dense, n_embed) assuming cnn_deep layout ending in Dense(softmax)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    # Find the pre-softmax Dense(192) or GAP output: last Dropout input is Dense(192)
    # Robust path: clone up to the layer before final Dense
    final = model.layers[-1]
    if not isinstance(final, layers.Dense):
        raise ValueError("expected final Dense softmax")
    # Build encoder: input → layer before final Dense
    # For Functional/Sequential-like Model from build_cnn_deep:
    # ... GAP → Dense(192) → Dropout → Dense(n)
    backbone_out = model.layers[-3].output  # Dense(192) relu
    encoder = keras.Model(model.input, backbone_out, name="hier_encoder")
    return encoder, int(backbone_out.shape[-1])


def finetune_hierarchical_from_base(
    base_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 40,
    smoke: bool = False,
    pair_lr: float = 3e-4,
    unfreeze_last_blocks: bool = True,
    gate_margin: float = 0.15,
) -> dict[str, Any]:
    """Fine-tune a weevil↔borer specialist on a trained 4-class CNN backbone.

    Inference soft-fusion:
      - keep baseline probs for clean / red_flour_beetle mass
      - when baseline top class is weevil or borer (or their combined mass is high),
        reallocate that mass with the pair head
      - optional gate: only override when |p_weevil - p_borer| < gate_margin
    """
    if not tensorflow_available():
        return {"skipped": True, "reason": "no tensorflow"}
    if base_model is None:
        return {"skipped": True, "reason": "no base_model"}

    from tensorflow import keras
    from tensorflow.keras import layers

    fit_epochs = 3 if smoke else epochs
    bs = min(32, max(4, len(X_train)))
    mask_tr = (y_train == WEEVIL) | (y_train == BORER)
    mask_va = (y_val == WEEVIL) | (y_val == BORER)
    if int(mask_tr.sum()) < 4:
        return {"skipped": True, "reason": "too few weevil/borer train samples"}

    encoder, _ = _split_encoder_and_head(base_model)

    # Pair head on frozen/lightly-unfrozen encoder
    for layer in encoder.layers:
        layer.trainable = False
    if unfreeze_last_blocks:
        # Unfreeze last conv block + Dense(192)
        trainable_tail = {encoder.layers[-1].name}  # Dense 192
        # also last few conv/BN before GAP
        for layer in encoder.layers[::-1]:
            if isinstance(layer, (layers.Conv2D, layers.Dense, layers.BatchNormalization)):
                trainable_tail.add(layer.name)
            if len(trainable_tail) >= 8:
                break
        for layer in encoder.layers:
            if layer.name in trainable_tail:
                layer.trainable = True

    inp = encoder.input
    h = encoder.output
    x = layers.Dropout(0.3)(h)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    pair_out = layers.Dense(2, activation="softmax", name="pair_softmax")(x)
    pair_model = keras.Model(inp, pair_out, name="weevil_borer_specialist")

    y_p_tr = (y_train[mask_tr] == BORER).astype(np.int32)
    X_p_tr = X_train[mask_tr]
    y_p_oh = keras.utils.to_categorical(y_p_tr, 2)
    seq_p = _make_spec_augment_sequence(
        keras, X_p_tr, y_p_oh, batch_size=min(bs, max(4, len(X_p_tr)))
    )
    steps_p = max(1, len(seq_p))
    pair_model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(
                pair_lr, max(1, steps_p * fit_epochs), alpha=1e-5
            )
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=["accuracy"],
    )
    val_pair = None
    if int(mask_va.sum()) >= 2:
        y_p_va = (y_val[mask_va] == BORER).astype(np.int32)
        val_pair = (X_val[mask_va], keras.utils.to_categorical(y_p_va, 2))

    hist = pair_model.fit(
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

    base_p = base_model.predict(X_val, verbose=0).astype(np.float64)
    pair_p = pair_model.predict(X_val, verbose=0).astype(np.float64)

    # Soft fusion
    probs = base_p.copy()
    n_override = 0
    for i in range(len(y_val)):
        mass = probs[i, WEEVIL] + probs[i, BORER]
        top = int(np.argmax(probs[i]))
        margin = abs(probs[i, WEEVIL] - probs[i, BORER])
        should = top in (WEEVIL, BORER) or mass >= 0.45
        if should and margin < gate_margin:
            probs[i, WEEVIL] = mass * pair_p[i, 0]
            probs[i, BORER] = mass * pair_p[i, 1]
            n_override += 1
        elif should and margin >= gate_margin:
            # still soft-blend lightly toward specialist
            alpha = 0.35
            probs[i, WEEVIL] = (1 - alpha) * probs[i, WEEVIL] + alpha * mass * pair_p[i, 0]
            probs[i, BORER] = (1 - alpha) * probs[i, BORER] + alpha * mass * pair_p[i, 1]
            # renormalize weevil+borer to mass
            s = probs[i, WEEVIL] + probs[i, BORER]
            if s > 0:
                probs[i, WEEVIL] *= mass / s
                probs[i, BORER] *= mass / s
            n_override += 1
    probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-12)
    pred = probs.argmax(1)

    out = _pack("hierarchical_finetuned", y_val, pred, probs, pair_model, None)
    out["n_pair_overrides"] = int(n_override)
    out["gate_margin"] = float(gate_margin)
    out["pair_val_acc_hist"] = [float(x) for x in hist.history.get("val_accuracy", [])]
    out["baseline_acc_reference"] = float(accuracy_score(y_val, base_p.argmax(1)))
    out["models"] = {"base": base_model, "pair": pair_model}
    if int(mask_va.sum()):
        yt = y_val[mask_va]
        yp = pred[mask_va]
        out["pair_subset_accuracy"] = float(accuracy_score(yt, yp))
        out["pair_confusion"] = confusion_matrix(yt, yp, labels=[WEEVIL, BORER]).tolist()
        # specialist alone on true pair subset
        pair_alone = pair_p[mask_va].argmax(1)
        pair_alone_lbl = np.where(pair_alone == 1, BORER, WEEVIL)
        out["pair_specialist_alone_acc"] = float(accuracy_score(yt, pair_alone_lbl))
    return out


def train_hierarchical_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 60,
    smoke: bool = False,
    base_model=None,
) -> dict[str, Any]:
    """If base_model given, fine-tune specialist; else legacy from-scratch cascade."""
    if base_model is not None:
        return finetune_hierarchical_from_base(
            base_model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=min(epochs, 40) if not smoke else epochs,
            smoke=smoke,
        )
    return _train_hierarchical_from_scratch(
        X_train, y_train, X_val, y_val, epochs=epochs, smoke=smoke
    )


def _train_hierarchical_from_scratch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 60,
    smoke: bool = False,
) -> dict[str, Any]:
    """Legacy: two CNNs from scratch (kept for ablation)."""
    if not tensorflow_available():
        return {"skipped": True, "reason": "no tensorflow"}
    from tensorflow import keras

    fit_epochs = 3 if smoke else epochs
    bs = min(32, max(4, len(X_train)))

    def to_coarse(y: np.ndarray) -> np.ndarray:
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

    mask_tr = (y_train == WEEVIL) | (y_train == BORER)
    mask_va = (y_val == WEEVIL) | (y_val == BORER)
    if mask_tr.sum() < 4:
        return {"skipped": True, "reason": "too few weevil/borer train samples"}

    y_p_tr = (y_train[mask_tr] == BORER).astype(np.int32)
    X_p_tr = X_train[mask_tr]
    model_p = build_cnn_deep(n_classes=2)
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

    coarse_p = model_c.predict(X_val, verbose=0)
    coarse = coarse_p.argmax(1)
    pair_p = model_p.predict(X_val, verbose=0)
    pair = pair_p.argmax(1)

    pred = np.zeros(len(y_val), dtype=np.int32)
    for i in range(len(y_val)):
        c = int(coarse[i])
        if c == 0:
            pred[i] = 0
        elif c == 2:
            pred[i] = 3
        else:
            pred[i] = BORER if pair[i] == 1 else WEEVIL

    probs = np.zeros((len(y_val), 4), dtype=np.float64)
    for i in range(len(y_val)):
        probs[i, 0] = coarse_p[i, 0]
        probs[i, 3] = coarse_p[i, 2]
        probs[i, WEEVIL] = coarse_p[i, 1] * pair_p[i, 0]
        probs[i, BORER] = coarse_p[i, 1] * pair_p[i, 1]
    probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-12)

    out = _pack("hierarchical_weevil_borer", y_val, pred, probs, None, None)
    out["models"] = {"coarse": model_c, "pair": model_p}
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
