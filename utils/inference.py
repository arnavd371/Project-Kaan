"""TFLite inference engine with demo-mode fallback."""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Dict, Union

import librosa
import numpy as np
import tensorflow as tf

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.preprocess import load_audio, preprocess_audio, trim_and_pad

CONFIDENCE_THRESHOLD = 0.60

CLASS_NAMES = ["clean", "rice_weevil", "lesser_grain_borer", "red_flour_beetle"]

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "grainear.tflite"

DEMO_RMS_THRESHOLD = 0.02
DEMO_CENTROID_LOW = 300
DEMO_CENTROID_HIGH = 500


def estimate_severity(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=20)

    # RMS energy - correlates with insect population density
    # per Balingbing et al. 2024 (Computers & Electronics in Agriculture)
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))

    # Impulse rate - number of high-energy frames per second
    # insects produce distinct impulses when feeding/moving
    threshold = np.mean(rms) + 1.5 * np.std(rms)
    impulse_frames = np.sum(rms > threshold)
    impulse_rate = impulse_frames / (len(y) / sr)

    # Map to three-tier severity scale
    # thresholds derived from RMS distribution in training data
    if mean_rms < 0.015 or impulse_rate < 2:
        return {
            "level": "Early",
            "color": "#f59e0b",
            "symbol": "🟡",
            "message": "Early stage infestation. Insects present but population is low.",
            "action": "Act within 2 weeks. Sun-dry grain and add neem leaves as preventive measure.",
            "urgency": 1,
        }
    elif mean_rms < 0.04 or impulse_rate < 8:
        return {
            "level": "Moderate",
            "color": "#f97316",
            "symbol": "🟠",
            "message": "Moderate infestation. Active insect activity detected.",
            "action": "Act within 3 days. Move grain to hermetic storage immediately.",
            "urgency": 2,
        }
    else:
        return {
            "level": "Severe",
            "color": "#ef4444",
            "symbol": "🔴",
            "message": "Severe infestation. Heavy insect activity detected.",
            "action": "Act immediately. Inspect outer grain layer, discard heavily damaged portions, contact KVK today.",
            "urgency": 3,
        }


class GrainEarPredictor:
    """Load TFLite model and run pest detection on audio input."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or MODEL_PATH
        self.demo_mode = not self.model_path.exists()
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        if self.demo_mode:
            warnings.warn("Model not found - running in DEMO MODE", stacklevel=2)
            print("Model not found - running in DEMO MODE")
        else:
            self._load_model()

    def _load_model(self):
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _quantize_input(self, spectrogram: np.ndarray) -> np.ndarray:
        detail = self.input_details[0]
        scale, zero_point = detail["quantization"]
        if scale == 0:
            return spectrogram.astype(detail["dtype"])
        quantized = spectrogram / scale + zero_point
        return np.clip(quantized, 0, 255).astype(detail["dtype"])

    def _dequantize_output(self, output: np.ndarray) -> np.ndarray:
        detail = self.output_details[0]
        scale, zero_point = detail["quantization"]
        return (output.astype(np.float32) - zero_point) * scale

    def _demo_predict(self, y: np.ndarray) -> Dict[str, Any]:
        """Rule-based heuristic when no trained model is available."""
        rms = float(np.sqrt(np.mean(y**2)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=16000)))

        if rms > DEMO_RMS_THRESHOLD and DEMO_CENTROID_LOW <= centroid <= DEMO_CENTROID_HIGH:
            predicted_class = "rice_weevil"
            confidence = 0.72
            all_scores = {
                "clean": 0.10,
                "rice_weevil": 0.72,
                "lesser_grain_borer": 0.10,
                "red_flour_beetle": 0.08,
            }
        else:
            predicted_class = "clean"
            confidence = 0.85
            all_scores = {
                "clean": 0.85,
                "rice_weevil": 0.05,
                "lesser_grain_borer": 0.05,
                "red_flour_beetle": 0.05,
            }

        return {
            "class": predicted_class,
            "confidence": confidence,
            "confident": confidence > CONFIDENCE_THRESHOLD,
            "all_scores": all_scores,
        }

    def predict(self, audio_source: Union[str, Path, bytes, io.BytesIO]) -> Dict[str, Any]:
        """
        Run prediction on audio bytes or file path.

        Returns dict with class, confidence, confident flag, and all_scores.
        """
        if self.demo_mode:
            y = load_audio(audio_source)
            y = trim_and_pad(y)
            return self._demo_predict(y)

        spectrogram = preprocess_audio(audio_source)
        input_data = np.expand_dims(spectrogram, axis=0)

        if self.input_details[0]["dtype"] == np.uint8:
            input_data = self._quantize_input(input_data)
        else:
            input_data = input_data.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        if self.output_details[0]["dtype"] == np.uint8:
            probabilities = self._dequantize_output(output[0])
        else:
            probabilities = output[0].astype(np.float32)

        probabilities = np.clip(probabilities, 0, None)
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()

        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])

        all_scores = {
            CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
        }

        return {
            "class": CLASS_NAMES[predicted_idx],
            "confidence": confidence,
            "confident": confidence > CONFIDENCE_THRESHOLD,
            "all_scores": all_scores,
        }
