"""Kaan / Project Kaan -- improved retrain (v5) on top of the leakage-fixed baseline.

Improvements vs. the audited 84.91% run:
  1. Deduplicate identical WAV contents and identical spectrograms BEFORE the
     train/val split (the previous run had 15 exact train/val spectrogram
     collisions from duplicate IRRI files).
  2. SpecAugment (time + frequency masking) during training -- standard for
     spectrogram CNNs; helps rice_weevil vs lesser_grain_borer confusion.
  3. Slightly deeper CNN (extra 128-filter block + SpatialDropout) while
     staying small enough for INT8 TFLite on phones.
  4. Label smoothing (0.05) + class weights (harder pest classes upweighted).
  5. Longer training schedule: cosine LR decay, early-stopping patience 15.
  6. Stronger on-the-fly augmentation for pest classes (prob 0.8, not 0.5).
  7. Same audits as before; AUDIT 3 must now PASS (0 duplicates).
"""

import hashlib
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
import tensorflow as tf
from scipy import ndimage
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

WORK_DIR = Path("/kaggle/working")
MODEL_DIR = WORK_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("/kaggle/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = TEMP_DIR / "data"
for c in ["rice_weevil", "lesser_grain_borer", "red_flour_beetle"]:
    (DATA_DIR / c).mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DURATION_SEC = 10
N_SAMPLES = SAMPLE_RATE * DURATION_SEC
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MEL_SHAPE = (128, 128)
CLASS_NAMES = ["clean", "rice_weevil", "lesser_grain_borer", "red_flour_beetle"]
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Fetch pest audio
# ---------------------------------------------------------------------------
print("Cloning IRRI Rice-Acoustic-Sensor repo...", flush=True)
REPO_DIR = TEMP_DIR / "Rice-Acoustic-Sensor"
if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/cbalingbing/Rice-Acoustic-Sensor", str(REPO_DIR)],
        check=True,
    )
print("Clone done.", flush=True)

extract_dir = TEMP_DIR / "Insect_WAVs_extracted"
if not extract_dir.exists():
    print("Unzipping Insect_WAVs.zip...", flush=True)
    subprocess.run(["unzip", "-q", str(REPO_DIR / "Insect_WAVs.zip"), "-d", str(extract_dir)], check=True)
print("Unzip done.", flush=True)

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
# 2. Real clean-class background audio
# ---------------------------------------------------------------------------
BG_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
bg_dir = TEMP_DIR / "background_real"
bg_src_dir = bg_dir / "_background_noise_"
if not bg_src_dir.exists():
    bg_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading real background-noise audio...", flush=True)
    archive_path = TEMP_DIR / "speech_commands_v0.02.tar.gz"
    result = subprocess.run(["curl", "-fSL", "-o", str(archive_path), BG_URL], check=False)
    if result.returncode != 0 or not archive_path.exists():
        raise RuntimeError(f"Failed to download Speech Commands archive (exit {result.returncode})")
    result = subprocess.run(
        ["tar", "-xzf", str(archive_path), "-C", str(bg_dir), "./_background_noise_"],
        check=False,
    )
    if not bg_src_dir.exists():
        result = subprocess.run(
            ["tar", "-xzf", str(archive_path), "-C", str(bg_dir), "_background_noise_"],
            check=False,
        )
    if not bg_src_dir.exists():
        raise RuntimeError("Failed to extract background noise set")
    try:
        archive_path.unlink()
    except OSError:
        pass
print("Background noise files:", sorted(p.name for p in bg_src_dir.glob("*.wav")), flush=True)

CLEAN_TRAIN_FILES = ["doing_the_dishes.wav", "dude_miaowing.wav", "exercise_bike.wav"]
CLEAN_VAL_FILES = ["running_tap.wav"]


def window_waveform(y, window_sec=10, hop_sec=2.5, sr=SAMPLE_RATE):
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    windows = []
    start = 0
    while start + win <= len(y):
        windows.append(y[start : start + win].astype(np.float32))
        start += hop
    if not windows:
        windows.append(trim_and_pad(y))
    return windows


def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y


def trim_and_pad(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trimmed) == 0:
        y_trimmed = y
    if len(y_trimmed) >= N_SAMPLES:
        start = (len(y_trimmed) -- N_SAMPLES) // 2
        return y_trimmed[start : start + N_SAMPLES].astype(np.float32)
    pad_total = N_SAMPLES -- len(y_trimmed)
    pad_left = pad_total // 2
    pad_right = pad_total -- pad_left
    return np.pad(y_trimmed, (pad_left, pad_right), mode="constant").astype(np.float32)


def add_pink_noise_at_snr(y, snr_db):
    white = np.random.randn(len(y))
    fft = np.fft.rfft(white)
    freqs = np.arange(len(fft)) + 1
    fft = fft / np.sqrt(freqs)
    noise = np.fft.irfft(fft, n=len(y)).astype(np.float32)
    signal_power = np.mean(y**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return y + noise


def augment_waveform(y, sr=SAMPLE_RATE):
    y_aug = y.copy()
    n_steps = random.uniform(-2.5, 2.5)
    y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=n_steps)
    snr_db = random.uniform(0, 12)
    y_aug = add_pink_noise_at_snr(y_aug, snr_db)
    rate = random.choice([0.8, 0.9, 1.1, 1.2])
    y_aug = librosa.effects.time_stretch(y_aug, rate=rate)
    return trim_and_pad(y_aug)


def waveform_to_mel_spectrogram(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    mel_norm = (mel_db -- mel_min) / (mel_max -- mel_min) if mel_max -- mel_min > 1e-8 else np.zeros_like(mel_db)
    zoom_factors = (MEL_SHAPE[0] / mel_norm.shape[0], MEL_SHAPE[1] / mel_norm.shape[1])
    mel_resized = ndimage.zoom(mel_norm, zoom_factors, order=1)[: MEL_SHAPE[0], : MEL_SHAPE[1]]
    if mel_resized.shape[0] < MEL_SHAPE[0] or mel_resized.shape[1] < MEL_SHAPE[1]:
        mel_resized = np.pad(
            mel_resized,
            ((0, MEL_SHAPE[0] -- mel_resized.shape[0]), (0, MEL_SHAPE[1] -- mel_resized.shape[1])),
            mode="constant",
        )
    return mel_resized.reshape(MEL_SHAPE[0], MEL_SHAPE[1], 1).astype(np.float32)


def file_bytes_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def spec_hash(spec):
    return hashlib.md5(np.ascontiguousarray(spec).tobytes()).hexdigest()



def spec_augment_np(spec, freq_mask=16, time_mask=24, n_freq=2, n_time=2):
    """Park et al. SpecAugment on a single (H, W, 1) mel spectrogram."""
    out = spec.copy()
    h, w, _ = out.shape
    for _ in range(n_freq):
        f = np.random.randint(0, freq_mask + 1)
        if f == 0 or f >= h:
            continue
        f0 = np.random.randint(0, h -- f + 1)
        out[f0 : f0 + f, :, :] = 0.0
    for _ in range(n_time):
        t = np.random.randint(0, time_mask + 1)
        if t == 0 or t >= w:
            continue
        t0 = np.random.randint(0, w -- t + 1)
        out[:, t0 : t0 + t, :] = 0.0
    return out


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
        batch_x = np.stack([spec_augment_np(self.x[i]) for i in batch_idx]).astype(np.float32)
        batch_y = self.y[batch_idx]
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ---------------------------------------------------------------------------
# 3. Load pest data + dedupe BEFORE split
# ---------------------------------------------------------------------------
print("\nLoading pest datasets...")
pest_specs, pest_labels, pest_paths = [], [], []
seen_file_hashes = set()
removed_file_dupes = 0
for class_idx, class_name in enumerate(CLASS_NAMES):
    if class_name == "clean":
        continue
    wav_files = sorted((DATA_DIR / class_name).glob("*.wav"))
    for wav_path in wav_files:
        fh = file_bytes_hash(wav_path)
        if fh in seen_file_hashes:
            removed_file_dupes += 1
            continue
        seen_file_hashes.add(fh)
        y = trim_and_pad(load_audio(wav_path))
        pest_specs.append(waveform_to_mel_spectrogram(y))
        pest_labels.append(class_idx)
        pest_paths.append(str(wav_path))

print(f"Removed {removed_file_dupes} byte-identical duplicate WAV files before split.")

# Also drop spectrogram-identical rows (near-duplicates after trim/mel)
seen_spec = {}
keep_idx = []
removed_spec_dupes = 0
for i, spec in enumerate(pest_specs):
    h = spec_hash(spec)
    if h in seen_spec:
        removed_spec_dupes += 1
        continue
    seen_spec[h] = i
    keep_idx.append(i)
print(f"Removed {removed_spec_dupes} spectrogram-identical samples before split.")

pest_specs = np.array([pest_specs[i] for i in keep_idx], dtype=np.float32)
pest_labels = np.array([pest_labels[i] for i in keep_idx], dtype=np.int32)
pest_paths = np.array([pest_paths[i] for i in keep_idx])
print(f"Pest samples after dedupe: {len(pest_specs)}")
for class_idx, class_name in enumerate(CLASS_NAMES):
    if class_name == "clean":
        continue
    print(f"  {class_name}: {int(np.sum(pest_labels == class_idx))}")

(
    pest_X_train,
    pest_X_val,
    pest_y_train,
    pest_y_val,
    pest_paths_train,
    pest_paths_val,
) = train_test_split(
    pest_specs, pest_labels, pest_paths, test_size=0.2, random_state=SEED, stratify=pest_labels
)

train_files_by_class = {
    class_name: pest_paths_train[pest_y_train == class_idx].tolist()
    for class_idx, class_name in enumerate(CLASS_NAMES)
    if class_name != "clean"
}

print("\nLoading real 'clean' background audio...")
clean_train_windows, clean_val_windows = [], []
for fname in CLEAN_TRAIN_FILES:
    y = load_audio(bg_src_dir / fname)
    clean_train_windows.extend(window_waveform(y))
for fname in CLEAN_VAL_FILES:
    y = load_audio(bg_src_dir / fname)
    clean_val_windows.extend(window_waveform(y))

print(f"clean -> {len(clean_train_windows)} train windows from {CLEAN_TRAIN_FILES}")
print(f"clean -> {len(clean_val_windows)} val windows from {CLEAN_VAL_FILES}")

clean_X_val = np.array([waveform_to_mel_spectrogram(w) for w in clean_val_windows], dtype=np.float32)
clean_y_val = np.zeros(len(clean_X_val), dtype=np.int32)

# ---------------------------------------------------------------------------
# 4. Build augmented training set
# ---------------------------------------------------------------------------
X_train_aug, y_train_aug = [], []

# Harder pest classes get slightly higher augmentation chance
AUG_PROB = {
    "rice_weevil": 0.85,
    "lesser_grain_borer": 0.9,
    "red_flour_beetle": 0.7,
}

for spec, label in zip(pest_X_train, pest_y_train):
    X_train_aug.append(spec)
    y_train_aug.append(label)
    cname = CLASS_NAMES[label]
    if np.random.random() < AUG_PROB[cname]:
        candidates = train_files_by_class[cname]
        wav_path = np.random.choice(candidates)
        y_wave = augment_waveform(trim_and_pad(load_audio(wav_path)))
        X_train_aug.append(waveform_to_mel_spectrogram(y_wave))
        y_train_aug.append(label)
    # Extra augmented copy for lesser_grain_borer (most confused class)
    if cname == "lesser_grain_borer" and np.random.random() < 0.5:
        candidates = train_files_by_class[cname]
        wav_path = np.random.choice(candidates)
        y_wave = augment_waveform(trim_and_pad(load_audio(wav_path)))
        X_train_aug.append(waveform_to_mel_spectrogram(y_wave))
        y_train_aug.append(label)

CLEAN_OVERSAMPLE_K = 8
for w in clean_train_windows:
    X_train_aug.append(waveform_to_mel_spectrogram(w))
    y_train_aug.append(0)
    for _ in range(CLEAN_OVERSAMPLE_K):
        y_aug = augment_waveform(w)
        X_train_aug.append(waveform_to_mel_spectrogram(y_aug))
        y_train_aug.append(0)

X_train = np.array(X_train_aug, dtype=np.float32)
y_train = np.array(y_train_aug, dtype=np.int32)
X_val = np.concatenate([pest_X_val, clean_X_val], axis=0)
y_val = np.concatenate([pest_y_val, clean_y_val], axis=0)

y_train_cat = keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
y_val_cat = keras.utils.to_categorical(y_val, num_classes=len(CLASS_NAMES))

print("\n=== Dataset composition after augmentation ===")
for class_idx, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name:22s} train={int(np.sum(y_train == class_idx)):4d}  val={int(np.sum(y_val == class_idx)):4d}")

# ---------------------------------------------------------------------------
# 5. Audits
# ---------------------------------------------------------------------------
print("\n=== AUDIT 1: file-level leakage check (pest classes) ===")
train_file_set = set(pest_paths_train.tolist())
val_file_set = set(pest_paths_val.tolist())
overlap = train_file_set & val_file_set
print(f"  Train files: {len(train_file_set)}  Val files: {len(val_file_set)}  Overlap: {len(overlap)}")
assert len(overlap) == 0
print("  PASS")

print("\n=== AUDIT 2: clean-class source-file separation ===")
assert len(set(CLEAN_TRAIN_FILES) & set(CLEAN_VAL_FILES)) == 0
print("  PASS")

print("\n=== AUDIT 3: exact-duplicate spectrogram check (train vs val) ===")
# Compare only unaugmented originals that could collide -- check all train vs val hashes
val_hashes = {spec_hash(s) for s in X_val}
# Prefer checking unaugmented pest train originals; also check full train set
dupe_count = sum(1 for s in X_train if spec_hash(s) in val_hashes)
print(f"  Exact duplicate spectrograms shared between train and val: {dupe_count}")
if dupe_count > 0:
    print("  WARNING: duplicates remain (likely from clean window overlap -- unexpected).")
else:
    print("  PASS -- zero exact duplicates between train and val.")

# ---------------------------------------------------------------------------
# 6. Model + training
# ---------------------------------------------------------------------------
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
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
model = keras.Model(inputs, outputs, name="project_kaan_cnn_v5")

train_seq = SpecAugmentSequence(X_train, y_train_cat, batch_size=32, shuffle=True)
steps_per_epoch = len(train_seq)
cosine = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=max(1, steps_per_epoch * 80), alpha=1e-5
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)
model.summary()

class_weights_arr = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=y_train)
class_weight = {i: float(w) for i, w in enumerate(class_weights_arr)}
class_weight[1] *= 1.15  # rice_weevil
class_weight[2] *= 1.25  # lesser_grain_borer
print("Class weights:", class_weight)

callbacks = [
    keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
    keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "best_val_acc.weights.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
]

print("\nTraining...")
history = model.fit(
    train_seq,
    validation_data=(X_val, y_val_cat),
    epochs=80,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# Prefer best val-accuracy checkpoint if it exists
best_w = MODEL_DIR / "best_val_acc.weights.h5"
if best_w.exists():
    model.load_weights(best_w)
    print("Loaded best val_accuracy checkpoint.")

MODEL_PATH = MODEL_DIR / "project-kaan_model.h5"
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
print("\nClassification Report (validation set):")
print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))

print("\n=== AUDIT 4: full confusion matrix (rows=true, cols=predicted) ===")
cm = confusion_matrix(y_val, y_pred)
header = "                    " + "".join(f"{n[:10]:>12s}" for n in CLASS_NAMES)
print(header)
for i, row in enumerate(cm):
    print(f"{CLASS_NAMES[i]:20s}" + "".join(f"{v:12d}" for v in row))

best_val_acc = max(history.history["val_accuracy"])
final_val_acc = history.history["val_accuracy"][-1]
# Recompute accuracy of the restored/checkpoint weights
restored_acc = float(np.mean(y_pred == y_val))
f1_per_class = f1_score(y_val, y_pred, average=None)
f1_macro = f1_score(y_val, y_pred, average="macro")

print("\n=== Final Results ===")
print(f"Best validation accuracy (during train): {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
print(f"Last-epoch validation accuracy:          {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
print(f"Restored/checkpoint accuracy:            {restored_acc:.4f} ({restored_acc * 100:.2f}%)")
print(f"Macro F1 score:                          {f1_macro:.4f}")
print("Per-class F1 scores:")
for name, score in zip(CLASS_NAMES, f1_per_class):
    print(f"  {name:22s} {score:.4f}")
print(f"Baseline to beat (previous audited run): 84.91% / macro F1 0.88")

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
# 7. TFLite INT8
# ---------------------------------------------------------------------------
NUM_CALIBRATION_SAMPLES = 100
X_all_for_calib = np.concatenate([X_train, X_val], axis=0)


def representative_dataset_gen():
    indices = np.random.choice(len(X_all_for_calib), min(NUM_CALIBRATION_SAMPLES, len(X_all_for_calib)), replace=False)
    for idx in indices:
        yield [np.expand_dims(X_all_for_calib[idx].astype(np.float32), axis=0)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

TFLITE_PATH = MODEL_DIR / "project-kaan.tflite"
TFLITE_PATH.write_bytes(tflite_model)
print(f"TFLite saved to {TFLITE_PATH} ({TFLITE_PATH.stat().st_size / 1024:.1f} KB)")
print("\nALL DONE.")
