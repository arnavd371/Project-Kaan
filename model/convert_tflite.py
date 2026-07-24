"""Convert trained GrainEar Keras model to INT8 quantized TFLite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.train import load_dataset  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent
H5_PATH = MODEL_DIR / "grainear_model.h5"
TFLITE_PATH = MODEL_DIR / "grainear.tflite"
NUM_CALIBRATION_SAMPLES = 100


def representative_dataset_gen():
    """Yield spectrograms for INT8 calibration."""
    try:
        X, _ = load_dataset(augment=False)
        if len(X) > 0:
            indices = np.random.choice(len(X), min(NUM_CALIBRATION_SAMPLES, len(X)), replace=False)
            for idx in indices:
                sample = X[idx].astype(np.float32)
                yield [np.expand_dims(sample, axis=0)]
            return
    except Exception:
        pass

    print("Warning: Using synthetic data for calibration (no training data found).")
    for _ in range(NUM_CALIBRATION_SAMPLES):
        sample = np.random.rand(1, 128, 128, 1).astype(np.float32)
        yield [sample]


def convert_model():
    if not H5_PATH.exists():
        print(f"Error: Model file not found at {H5_PATH}")
        print("Run python model/train.py first.")
        sys.exit(1)

    h5_size_kb = H5_PATH.stat().st_size / 1024
    print(f"H5 model size: {h5_size_kb:.1f} KB")

    model = keras.models.load_model(H5_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    TFLITE_PATH.write_bytes(tflite_model)
    tflite_size_kb = TFLITE_PATH.stat().st_size / 1024
    print(f"TFLite model size: {tflite_size_kb:.1f} KB")
    print(f"Compression ratio: {h5_size_kb / tflite_size_kb:.2f}x")
    print(f"Saved to {TFLITE_PATH}")


if __name__ == "__main__":
    convert_model()
