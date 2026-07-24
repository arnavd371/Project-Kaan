"""Self-contained Kaggle GPU training script for the Kaan / GrainEar model.

Reproduces the exact same pipeline as model/train.py + model/preprocess.py +
model/convert_tflite.py, but runs standalone on Kaggle (no local package
imports needed) so it can be pasted into a single Kaggle Notebook cell.

HOW TO USE
----------
1. Go to kaggle.com/code -> New Notebook.
2. Settings (right sidebar) -> Accelerator -> GPU T4 x2 (or any GPU).
3. Settings -> Internet -> On (needed to clone the dataset repo).
4. Paste this entire file into one cell and Run All.
5. When it finishes, download these from the Output pane (/kaggle/working/model/):
     - grainear_model.h5
     - grainear.tflite
     - training_history.png
6. Copy both files into ~/Downloads/Project Kaan/model/ on your Mac,
   overwriting the existing ones. Restart the Streamlit app to pick them up.
"""

import random
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "librosa", "soundfile"], check=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from scipy import ndimage
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "data"
MODEL_DIR = WORK_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for c in ["clean", "rice_weevil", "lesser_grain_borer", "red_flour_beetle"]:
    (DATA_DIR / c).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Fetch pest audio (IRRI Rice Acoustic Sensor, public GitHub repo)
# ---------------------------------------------------------------------------
REPO_DIR = WORK_DIR / "Rice-Acoustic-Sensor"
if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/cbalingbing/Rice-Acoustic-Sensor", str(REPO_DIR)],
        check=True,
    )

extract_dir = WORK_DIR / "Insect_WAVs_extracted"
if not extract_dir.exists():
    subprocess.run(["unzip", "-q", str(REPO_DIR / "Insect_WAVs.zip"), "-d", str(extract_dir)], check=True)

SPECIES_MAP = {
    "S_oryzae": "rice_weevil",
    "R_dominica": "lesser_grain_borer",
    "T_castaneum": "red_flour_beetle",
}
wav_root = extract_dir / "Insect_WAVs"
for species, class_name in SPECIES_MAP.items():
    dst = DATA_DIR / class_name
    for f in (wav_root / species).glob("*.wav"):
        dst_file = dst / f.name
        if not dst_file.exists():
            dst_file.write_bytes(f.read_bytes())
    print(class_name, "->", len(list(dst.glob("*.wav"))), "files")

# ---------------------------------------------------------------------------
# 2. Generate synthetic pink-noise clean class (identical to generate_clean_data.py)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION_SEC = 10
N_SAMPLES = SAMPLE_RATE * DURATION_SEC


def generate_pink_noise(length):
    white = np.random.randn(length)
    fft = np.fft.rfft(white)
    freqs = np.arange(len(fft)) + 1
    fft = fft / np.sqrt(freqs)
    return np.fft.irfft(fft, n=length).astype(np.float32)


clean_dir = DATA_DIR / "clean"
NUM_SYNTHETIC = 493
TARGET_PEAK = 0.3
for i in range(NUM_SYNTHETIC):
    out_path = clean_dir / f"synthetic_clean_{i:03d}.wav"
    if out_path.exists():
        continue
    pink = generate_pink_noise(N_SAMPLES)
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink = pink / peak * TARGET_PEAK
    sf.write(out_path, pink.astype(np.float32), SAMPLE_RATE)
print("clean ->", len(list(clean_dir.glob("*.wav"))), "files")

# ---------------------------------------------------------------------------
# 3. Preprocessing pipeline (identical to model/preprocess.py)
# ---------------------------------------------------------------------------
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MEL_SHAPE = (128, 128)


def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y


def trim_and_pad(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trimmed) == 0:
        y_trimmed = y
    if len(y_trimmed) >= N_SAMPLES:
        start = (len(y_trimmed) - N_SAMPLES) // 2
        return y_trimmed[start : start + N_SAMPLES].astype(np.float32)
    pad_total = N_SAMPLES - len(y_trimmed)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y_trimmed, (pad_left, pad_right), mode="constant").astype(np.float32)


def add_pink_noise_at_snr(y, snr_db):
    noise = generate_pink_noise(len(y))
    signal_power = np.mean(y**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return y + noise


def augment_waveform(y, sr=SAMPLE_RATE):
    y_aug = y.copy()
    n_steps = random.uniform(-2, 2)
    y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=n_steps)
    snr_db = random.uniform(0, 10)
    y_aug = add_pink_noise_at_snr(y_aug, snr_db)
    rate = random.choice([0.8, 1.2])
    y_aug = librosa.effects.time_stretch(y_aug, rate=rate)
    return trim_and_pad(y_aug)


def waveform_to_mel_spectrogram(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min) if mel_max - mel_min > 1e-8 else np.zeros_like(mel_db)
    zoom_factors = (MEL_SHAPE[0] / mel_norm.shape[0], MEL_SHAPE[1] / mel_norm.shape[1])
    mel_resized = ndimage.zoom(mel_norm, zoom_factors, order=1)[: MEL_SHAPE[0], : MEL_SHAPE[1]]
    if mel_resized.shape[0] < MEL_SHAPE[0] or mel_resized.shape[1] < MEL_SHAPE[1]:
        mel_resized = np.pad(
            mel_resized,
            ((0, MEL_SHAPE[0] - mel_resized.shape[0]), (0, MEL_SHAPE[1] - mel_resized.shape[1])),
            mode="constant",
        )
    return mel_resized.reshape(MEL_SHAPE[0], MEL_SHAPE[1], 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Load dataset
# ---------------------------------------------------------------------------
CLASS_NAMES = ["clean", "rice_weevil", "lesser_grain_borer", "red_flour_beetle"]


def load_dataset():
    spectrograms, labels = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        wav_files = sorted((DATA_DIR / class_name).glob("*.wav"))
        for wav_path in wav_files:
            y = trim_and_pad(load_audio(wav_path))
            spectrograms.append(waveform_to_mel_spectrogram(y))
            labels.append(class_idx)
    return np.array(spectrograms, dtype=np.float32), np.array(labels, dtype=np.int32)


print("Loading dataset...")
X, y = load_dataset()
print(f"Loaded {len(X)} samples across {len(CLASS_NAMES)} classes.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply augmentation to training set only
X_train_aug, y_train_aug = [], []
for spec, label in zip(X_train, y_train):
    X_train_aug.append(spec)
    y_train_aug.append(label)
    if np.random.random() < 0.5:
        wav_files = list((DATA_DIR / CLASS_NAMES[label]).glob("*.wav"))
        if wav_files:
            wav_path = np.random.choice(wav_files)
            y_wave = augment_waveform(trim_and_pad(load_audio(wav_path)))
            X_train_aug.append(waveform_to_mel_spectrogram(y_wave))
            y_train_aug.append(label)

X_train = np.array(X_train_aug, dtype=np.float32)
y_train = np.array(y_train_aug, dtype=np.int32)
y_train_cat = keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
y_val_cat = keras.utils.to_categorical(y_val, num_classes=len(CLASS_NAMES))

# ---------------------------------------------------------------------------
# 5. Build + train model (identical architecture to model/train.py)
# ---------------------------------------------------------------------------
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
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_loss", verbose=1),
]

print("\nTraining...")
history = model.fit(
    X_train, y_train_cat, validation_data=(X_val, y_val_cat), epochs=60, batch_size=32, callbacks=callbacks, verbose=1
)

MODEL_PATH = MODEL_DIR / "grainear_model.h5"
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

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss")
axes[0].legend()
axes[1].plot(history.history["accuracy"], label="Train Acc")
axes[1].plot(history.history["val_accuracy"], label="Val Acc")
axes[1].set_title("Accuracy")
axes[1].legend()
plt.tight_layout()
fig.savefig(MODEL_DIR / "training_history.png", dpi=150)
print("Training history saved.")

# ---------------------------------------------------------------------------
# 6. Convert to INT8 TFLite (identical to model/convert_tflite.py)
# ---------------------------------------------------------------------------
NUM_CALIBRATION_SAMPLES = 100


def representative_dataset_gen():
    indices = np.random.choice(len(X), min(NUM_CALIBRATION_SAMPLES, len(X)), replace=False)
    for idx in indices:
        yield [np.expand_dims(X[idx].astype(np.float32), axis=0)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

TFLITE_PATH = MODEL_DIR / "grainear.tflite"
TFLITE_PATH.write_bytes(tflite_model)
print(f"TFLite saved to {TFLITE_PATH} ({TFLITE_PATH.stat().st_size / 1024:.1f} KB)")

print("\nALL DONE. Download grainear_model.h5 and grainear.tflite from /kaggle/working/model/")
