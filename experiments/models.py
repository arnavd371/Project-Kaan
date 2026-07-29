"""Model builders and trainers for the multi-approach benchmark.

CNN builders match model/train.py and model/train_kaggle.py v5.
Classical models share the handcrafted feature matrix and split.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.common import CLASS_NAMES, REFERENCE_PAPER_VAL_ACC


@dataclass
class ApproachResult:
    approach_id: str
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion: list[list[int]]
    n_params: int | None
    train_seconds: float
    beats_reference: bool
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    notes: str = ""
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach_id": self.approach_id,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class_f1": self.per_class_f1,
            "confusion": self.confusion,
            "n_params": self.n_params,
            "train_seconds": self.train_seconds,
            "beats_reference": self.beats_reference,
            "reference_acc": REFERENCE_PAPER_VAL_ACC,
            "notes": self.notes,
            "skipped": self.skipped,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
        }


def _pack_metrics(
    approach_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_params: int | None,
    train_seconds: float,
    notes: str = "",
) -> ApproachResult:
    per = f1_score(y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist()
    return ApproachResult(
        approach_id=approach_id,
        accuracy=acc,
        macro_f1=macro,
        per_class_f1={CLASS_NAMES[i]: float(per[i]) for i in range(len(CLASS_NAMES))},
        confusion=cm,
        n_params=n_params,
        train_seconds=train_seconds,
        beats_reference=acc > REFERENCE_PAPER_VAL_ACC,
        y_true=y_true.astype(int).tolist(),
        y_pred=y_pred.astype(int).tolist(),
        notes=notes,
    )


def _skipped(approach_id: str, reason: str) -> ApproachResult:
    return ApproachResult(
        approach_id=approach_id,
        accuracy=0.0,
        macro_f1=0.0,
        per_class_f1={c: 0.0 for c in CLASS_NAMES},
        confusion=[[0] * len(CLASS_NAMES) for _ in CLASS_NAMES],
        n_params=None,
        train_seconds=0.0,
        beats_reference=False,
        notes=reason,
        skipped=True,
    )


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


# CNN architectures


def build_cnn_shallow(n_classes: int = 4):
    """Shallow mel-CNN (model/train.py)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential(
        [
            layers.Input(shape=(128, 128, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="project_kaan_cnn",
    )


def build_cnn_deep(n_classes: int = 4):
    """Deeper mel-CNN v5 (model/train_kaggle.py)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(128, 128, 1))
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.15)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(192, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="project_kaan_cnn_v5")


def _spec_augment_np(spec: np.ndarray, freq_mask: int = 16, time_mask: int = 24, n_freq: int = 2, n_time: int = 2):
    out = spec.copy()
    h, w, _ = out.shape
    for _ in range(n_freq):
        f = int(np.random.randint(0, freq_mask + 1))
        if f == 0 or f >= h:
            continue
        f0 = int(np.random.randint(0, h - f + 1))
        out[f0 : f0 + f, :, :] = 0.0
    for _ in range(n_time):
        t = int(np.random.randint(0, time_mask + 1))
        if t == 0 or t >= w:
            continue
        t0 = int(np.random.randint(0, w - t + 1))
        out[:, t0 : t0 + t, :] = 0.0
    return out


def _make_spec_augment_sequence(keras, X: np.ndarray, y_cat: np.ndarray, batch_size: int):
    class SpecAugmentSequence(keras.utils.Sequence):
        def __init__(self, x, y, batch_size=32, shuffle=True):
            self.x = x
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = np.arange(len(x))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.x) / self.batch_size))

        def __getitem__(self, idx):
            batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_x = np.stack([_spec_augment_np(self.x[i]) for i in batch_idx]).astype(np.float32)
            batch_y = self.y[batch_idx]
            return batch_x, batch_y

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)

    return SpecAugmentSequence(X, y_cat, batch_size=batch_size, shuffle=True)


def train_cnn(
    approach_id: str,
    builder: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    smoke: bool = False,
    strong: bool = True,
    recipe: dict | None = None,
) -> ApproachResult:
    """Train a mel-CNN.

    recipe keys (defaults when strong=True):
      spec_augment, class_weights, label_smoothing, cosine_lr
    strong=False forces all recipe flags off.
    """
    if not tensorflow_available():
        return _skipped(approach_id, "TensorFlow not installed; CNN skipped.")

    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow import keras

    flags = {
        "spec_augment": True,
        "class_weights": True,
        "label_smoothing": True,
        "cosine_lr": True,
    }
    if not strong or smoke:
        flags = {k: False for k in flags}
    if recipe is not None:
        flags.update({k: bool(v) for k, v in recipe.items() if k in flags})

    t0 = time.perf_counter()
    model = builder(n_classes=len(CLASS_NAMES))
    fit_epochs = 3 if smoke else epochs
    bs = min(batch_size, max(4, len(X_train)))
    history_dict: dict[str, list] = {}

    use_categorical = flags["spec_augment"] or flags["label_smoothing"] or flags["cosine_lr"] or flags["class_weights"]
    if use_categorical and not smoke:
        y_train_cat = keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=len(CLASS_NAMES))
        if flags["spec_augment"]:
            train_data = _make_spec_augment_sequence(keras, X_train, y_train_cat, batch_size=bs)
            steps = max(1, len(train_data))
        else:
            train_data = X_train
            steps = max(1, int(np.ceil(len(X_train) / bs)))

        if flags["cosine_lr"]:
            lr = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-3,
                decay_steps=max(1, steps * max(fit_epochs, 1)),
                alpha=1e-5,
            )
        else:
            lr = 1e-3

        smooth = 0.05 if flags["label_smoothing"] else 0.0
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=smooth),
            metrics=["accuracy"],
        )

        class_weight = None
        if flags["class_weights"]:
            cw_arr = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=y_train)
            class_weight = {i: float(w) for i, w in enumerate(cw_arr)}
            class_weight[1] *= 1.15
            class_weight[2] *= 1.25

        patience = 25 if approach_id == "cnn_shallow" else 15
        monitor = "val_accuracy" if approach_id == "cnn_shallow" else "val_loss"
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=patience,
                restore_best_weights=True,
                monitor=monitor,
                mode="max" if monitor == "val_accuracy" else "min",
            ),
        ]
        fit_kwargs: dict[str, Any] = {
            "validation_data": (X_val, y_val_cat),
            "epochs": fit_epochs,
            "callbacks": callbacks,
            "verbose": 1,
        }
        if class_weight is not None:
            fit_kwargs["class_weight"] = class_weight
        if flags["spec_augment"]:
            history = model.fit(train_data, **fit_kwargs)
        else:
            history = model.fit(X_train, y_train_cat, batch_size=bs, **fit_kwargs)
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        on = [k for k, v in flags.items() if v]
        note = f"mel-CNN recipe=[{','.join(on) or 'none'}] epochs≤{fit_epochs}"
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=fit_epochs,
            batch_size=bs,
            verbose=0 if smoke else 1,
        )
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        note = "smoke CNN" if smoke else "baseline mel-CNN"

    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    n_params = int(model.count_params())
    elapsed = time.perf_counter() - t0
    result = _pack_metrics(approach_id, y_val, y_pred, n_params, elapsed, notes=note)
    result._history = history_dict  # type: ignore[attr-defined]
    result._recipe = flags  # type: ignore[attr-defined]
    return result


# Classical approaches


def _count_sklearn_params(est) -> int | None:
    if hasattr(est, "coef_"):
        return int(np.size(est.coef_) + (np.size(est.intercept_) if hasattr(est, "intercept_") else 0))
    if hasattr(est, "n_features_in_"):
        # Approximate capacity for tree / MLP
        if hasattr(est, "coefs_"):
            return int(sum(np.size(c) for c in est.coefs_) + sum(np.size(b) for b in est.intercepts_))
        return None
    return None


def train_svm_rbf(X_train, y_train, X_val, y_val, smoke: bool = False) -> ApproachResult:
    t0 = time.perf_counter()
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")),
        ]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    return _pack_metrics(
        "svm_rbf",
        y_val,
        y_pred,
        None,
        time.perf_counter() - t0,
        notes="RBF SVM on MFCC+spectral vector",
    )


def train_mlp(X_train, y_train, X_val, y_val, smoke: bool = False) -> ApproachResult:
    t0 = time.perf_counter()
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    max_iter=80 if smoke else 400,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    n_params = _count_sklearn_params(clf.named_steps["mlp"])
    return _pack_metrics(
        "mlp",
        y_val,
        y_pred,
        n_params,
        time.perf_counter() - t0,
        notes="MLP on MFCC+spectral vector",
    )


def train_gbdt(X_train, y_train, X_val, y_val, smoke: bool = False) -> ApproachResult:
    t0 = time.perf_counter()
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=50 if smoke else 200,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    return _pack_metrics(
        "gbdt",
        y_val,
        y_pred,
        None,
        time.perf_counter() - t0,
        notes="HistGradientBoosting on MFCC+spectral vector",
    )


def train_logreg(X_train, y_train, X_val, y_val, smoke: bool = False) -> ApproachResult:
    t0 = time.perf_counter()
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=200 if smoke else 1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    n_params = _count_sklearn_params(clf.named_steps["lr"])
    return _pack_metrics(
        "logreg",
        y_val,
        y_pred,
        n_params,
        time.perf_counter() - t0,
        notes="Multinomial logistic regression floor",
    )


APPROACH_ORDER = [
    "cnn_shallow",
    "cnn_deep",
    "svm_rbf",
    "mlp",
    "gbdt",
    "logreg",
]


def run_all_approaches(
    X_hand_train: np.ndarray,
    X_hand_val: np.ndarray,
    X_mel_train: np.ndarray,
    X_mel_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    approaches: list[str] | None = None,
    smoke: bool = False,
    cnn_epochs: int = 60,
    cnn_strong: bool = True,
    cnn_recipe: dict | None = None,
) -> list[ApproachResult]:
    wanted = approaches or APPROACH_ORDER
    results: list[ApproachResult] = []

    for name in wanted:
        if name == "cnn_shallow":
            results.append(
                train_cnn(
                    "cnn_shallow",
                    build_cnn_shallow,
                    X_mel_train,
                    y_train,
                    X_mel_val,
                    y_val,
                    epochs=cnn_epochs,
                    smoke=smoke,
                    strong=cnn_strong,
                    recipe=cnn_recipe,
                )
            )
        elif name == "cnn_deep":
            results.append(
                train_cnn(
                    "cnn_deep",
                    build_cnn_deep,
                    X_mel_train,
                    y_train,
                    X_mel_val,
                    y_val,
                    epochs=cnn_epochs,
                    smoke=smoke,
                    strong=cnn_strong,
                    recipe=cnn_recipe,
                )
            )
        elif name == "svm_rbf":
            results.append(train_svm_rbf(X_hand_train, y_train, X_hand_val, y_val, smoke=smoke))
        elif name == "mlp":
            results.append(train_mlp(X_hand_train, y_train, X_hand_val, y_val, smoke=smoke))
        elif name == "gbdt":
            results.append(train_gbdt(X_hand_train, y_train, X_hand_val, y_val, smoke=smoke))
        elif name == "logreg":
            results.append(train_logreg(X_hand_train, y_train, X_hand_val, y_val, smoke=smoke))
        else:
            results.append(_skipped(name, f"Unknown approach: {name}"))
    return results


def format_classification_report(result: ApproachResult) -> str:
    if result.skipped or not result.y_true:
        return f"{result.approach_id}: skipped ({result.notes})"
    return classification_report(
        result.y_true,
        result.y_pred,
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    )
