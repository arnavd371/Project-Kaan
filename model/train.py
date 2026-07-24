"""Train the GrainEar CNN model on acoustic grain pest data."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Allow imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.preprocess import (  # noqa: E402
    augment_waveform,
    load_audio,
    trim_and_pad,
    waveform_to_mel_spectrogram,
)

CLASS_NAMES = ["clean", "rice_weevil", "lesser_grain_borer", "red_flour_beetle"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "grainear_model.h5"
HISTORY_PATH = MODEL_DIR / "training_history.png"


def load_dataset(augment: bool = False, augment_prob: float = 0.5):
    """Load WAV files from data/ subdirectories and convert to spectrograms."""
    spectrograms = []
    labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = DATA_DIR / class_name
        wav_files = sorted(class_dir.glob("*.wav"))
        if not wav_files:
            print(f"Warning: No WAV files found in {class_dir}")

        for wav_path in wav_files:
            y = load_audio(wav_path)
            y = trim_and_pad(y)
            if augment and np.random.random() < augment_prob:
                y = augment_waveform(y)
            spec = waveform_to_mel_spectrogram(y)
            spectrograms.append(spec)
            labels.append(class_idx)

    return np.array(spectrograms, dtype=np.float32), np.array(labels, dtype=np.int32)


def build_model() -> keras.Model:
    """Build the GrainEar CNN architecture."""
    model = keras.Sequential(
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
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ],
        name="grainear_cnn",
    )
    return model


def plot_training_history(history, save_path: Path):
    """Save loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")

    axes[0].plot(history.history["loss"], label="Train Loss", color="#e74c3c")
    axes[0].plot(history.history["val_loss"], label="Val Loss", color="#3498db")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    axes[1].plot(history.history["accuracy"], label="Train Acc", color="#2ecc71")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc", color="#f39c12")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor="#0e1117")
    plt.close(fig)
    print(f"Training history saved to {save_path}")


def main():
    print("Loading dataset...")
    X, y = load_dataset(augment=False)

    if len(X) == 0:
        print("\nError: No WAV files found in data/ subdirectories.")
        print("Please follow instructions in data/README.md to download and organize datasets.")
        sys.exit(1)

    print(f"Loaded {len(X)} samples across {len(CLASS_NAMES)} classes.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply augmentation to training set only
    X_train_aug = []
    y_train_aug = []
    for spec, label in zip(X_train, y_train):
        X_train_aug.append(spec)
        y_train_aug.append(label)
        if np.random.random() < 0.5:
            class_dir = DATA_DIR / CLASS_NAMES[label]
            wav_files = list(class_dir.glob("*.wav"))
            if wav_files:
                wav_path = np.random.choice(wav_files)
                y_wave = load_audio(wav_path)
                y_wave = augment_waveform(trim_and_pad(y_wave))
                X_train_aug.append(waveform_to_mel_spectrogram(y_wave))
                y_train_aug.append(label)

    X_train = np.array(X_train_aug, dtype=np.float32)
    y_train = np.array(y_train_aug, dtype=np.int32)

    y_train_cat = keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=len(CLASS_NAMES))

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=5, factor=0.5, monitor="val_loss", verbose=1
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nClassification Report (validation set):")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))

    best_val_acc = max(history.history["val_accuracy"])
    final_val_acc = history.history["val_accuracy"][-1]
    f1_per_class = f1_score(y_val, y_pred, average=None)
    f1_macro = f1_score(y_val, y_pred, average="macro")

    print("\n=== Final Results ===")
    print(f"Best validation accuracy:  {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Final validation accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
    print(f"Macro F1 score:            {f1_macro:.4f}")
    print("Per-class F1 scores:")
    for name, score in zip(CLASS_NAMES, f1_per_class):
        print(f"  {name:22s} {score:.4f}")

    plot_training_history(history, HISTORY_PATH)


if __name__ == "__main__":
    main()
