"""Self-supervised mel pretrain (SimCLR-style) on public/ambient + IRRI audio, then fine-tune.

Uses SpecAugment views of mel spectrograms — no field recordings required.
Public sources available via prepare_kaggle_data: Speech Commands ambient windows + IRRI WAVs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.common import CLASS_NAMES
from experiments.models import _spec_augment_np, build_cnn_deep, tensorflow_available
from sklearn.metrics import accuracy_score, f1_score


def _encoder_backbone(proj_dim: int = 128):
    """Deep CNN trunk ending in a projection head (no softmax)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(128, 128, 1))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.1)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.15)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    h = layers.GlobalAveragePooling2D(name="embedding")(x)
    z = layers.Dense(256, activation="relu")(h)
    z = layers.Dense(proj_dim, name="projection")(z)
    return keras.Model(inputs, [h, z], name="ssl_encoder")


def nt_xent(z1, z2, temperature: float = 0.2):
    import tensorflow as tf

    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    batch = tf.shape(z1)[0]
    z = tf.concat([z1, z2], axis=0)  # 2B, D
    sim = tf.matmul(z, z, transpose_b=True) / temperature
    logits_mask = tf.ones_like(sim) - tf.eye(2 * batch)
    sim = sim * logits_mask - 1e9 * (1.0 - logits_mask)
    labels = tf.range(batch)
    labels = tf.concat([labels + batch, labels], axis=0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim)
    return tf.reduce_mean(loss)


def pretrain_simclr(
    X_mels: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    smoke: bool = False,
) -> Any:
    if not tensorflow_available():
        raise RuntimeError("TensorFlow required")
    import tensorflow as tf
    from tensorflow import keras

    enc = _encoder_backbone()
    opt = keras.optimizers.Adam(1e-3)
    fit_epochs = 2 if smoke else epochs
    bs = min(batch_size, max(4, len(X_mels) // 2))
    n = len(X_mels)

    @tf.function
    def step(v1, v2):
        with tf.GradientTape() as tape:
            _, z1 = enc(v1, training=True)
            _, z2 = enc(v2, training=True)
            loss = nt_xent(z1, z2)
        grads = tape.gradient(loss, enc.trainable_variables)
        opt.apply_gradients(zip(grads, enc.trainable_variables))
        return loss

    history = []
    for ep in range(fit_epochs):
        order = np.random.permutation(n)
        losses = []
        for start in range(0, n - bs + 1, bs):
            idx = order[start : start + bs]
            batch = X_mels[idx]
            v1 = np.stack([_spec_augment_np(x) for x in batch]).astype(np.float32)
            v2 = np.stack([_spec_augment_np(x) for x in batch]).astype(np.float32)
            loss = step(tf.convert_to_tensor(v1), tf.convert_to_tensor(v2))
            losses.append(float(loss))
        row = {"epoch": ep + 1, "loss": float(np.mean(losses) if losses else 0.0)}
        history.append(row)
        print(f"[ssl] epoch {ep+1}/{fit_epochs} loss={row['loss']:.4f}", flush=True)
    return enc, history


def finetune_from_encoder(
    encoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 60,
    smoke: bool = False,
) -> dict[str, Any]:
    from tensorflow import keras
    from tensorflow.keras import layers
    from experiments.models import _make_spec_augment_sequence

    emb_layer = encoder.get_layer("embedding")
    inp = encoder.input
    h = emb_layer.output
    x = layers.Dense(192, activation="relu")(h)
    x = layers.Dropout(0.45)(x)
    out = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    clf = keras.Model(inp, out, name="ssl_finetuned")

    fit_epochs = 3 if smoke else epochs
    bs = min(32, max(4, len(X_train)))
    y_tr = keras.utils.to_categorical(y_train, len(CLASS_NAMES))
    y_va = keras.utils.to_categorical(y_val, len(CLASS_NAMES))
    seq = _make_spec_augment_sequence(keras, X_train, y_tr, batch_size=bs)
    steps = max(1, len(seq))
    clf.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(1e-3, max(1, steps * fit_epochs), alpha=1e-5)
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    clf.fit(
        seq,
        validation_data=(X_val, y_va),
        epochs=fit_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=3 if smoke else 15, restore_best_weights=True, monitor="val_accuracy", mode="max"
            )
        ],
        verbose=1,
    )
    probs = clf.predict(X_val, verbose=0)
    pred = probs.argmax(1)
    return {
        "approach_id": "ssl_simclr_finetune",
        "accuracy": float(accuracy_score(y_val, pred)),
        "macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "probs": probs.astype(np.float32),
        "y_pred": pred.astype(np.int32),
        "y_true": y_val.astype(np.int32),
        "n_params": int(clf.count_params()),
        "model": clf,
        "skipped": False,
    }


def run_ssl_pipeline(
    X_pretrain: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    pretrain_epochs: int = 30,
    finetune_epochs: int = 60,
    smoke: bool = False,
) -> dict[str, Any]:
    if not tensorflow_available():
        return {"skipped": True, "reason": "no tensorflow"}
    enc, hist = pretrain_simclr(X_pretrain, epochs=pretrain_epochs, smoke=smoke)
    ft = finetune_from_encoder(
        enc, X_train, y_train, X_val, y_val, epochs=finetune_epochs, smoke=smoke
    )
    ft["pretrain_history"] = hist
    ft["n_pretrain"] = int(len(X_pretrain))
    return ft
