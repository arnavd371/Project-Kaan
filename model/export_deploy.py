"""Export distilled (or any) Keras H5 → INT8 TFLite + ONNX for web/native.

Also patches web/src/lib/model.ts quantization constants to match the new TFLite scales.

  python -m model.export_deploy
  python -m model.export_deploy --h5 model/project-kaan_model.h5
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.train import load_dataset  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent
WEB_MODEL = ROOT / "web" / "public" / "model"
MODEL_TS = ROOT / "web" / "src" / "lib" / "model.ts"
NUM_CALIBRATION_SAMPLES = 100


def _representative(h5_hint_data: bool = True):
    def gen():
        if h5_hint_data:
            try:
                X, _ = load_dataset(augment=False)
                if len(X) > 0:
                    idx = np.random.choice(
                        len(X), min(NUM_CALIBRATION_SAMPLES, len(X)), replace=False
                    )
                    for i in idx:
                        yield [np.expand_dims(X[i].astype(np.float32), axis=0)]
                    return
            except Exception as e:
                print(f"Calibration data load failed ({e}); using synthetic.", flush=True)
        for _ in range(NUM_CALIBRATION_SAMPLES):
            yield [np.random.rand(1, 128, 128, 1).astype(np.float32)]

    return gen


def convert_tflite(h5_path: Path, tflite_path: Path) -> dict:
    model = keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative()
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    blob = converter.convert()
    tflite_path.write_bytes(blob)
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    meta = {
        "input_name": inp["name"],
        "output_name": out["name"],
        "input_scale": float(in_scale),
        "input_zero_point": int(in_zp),
        "output_scale": float(out_scale),
        "output_zero_point": int(out_zp),
        "tflite_kb": tflite_path.stat().st_size / 1024,
        "h5_kb": h5_path.stat().st_size / 1024,
    }
    print(
        f"TFLite {meta['tflite_kb']:.1f} KB  in={meta['input_name']} "
        f"scale={meta['input_scale']}  out_scale={meta['output_scale']}",
        flush=True,
    )
    return meta


def convert_onnx(tflite_path: Path, onnx_path: Path) -> None:
    try:
        import tf2onnx  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tf2onnx", "onnx"])
    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--tflite",
        str(tflite_path),
        "--output",
        str(onnx_path),
        "--opset",
        "13",
    ]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"tf2onnx failed with code {r.returncode}")
    print(f"ONNX → {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)", flush=True)


def patch_model_ts(meta: dict) -> None:
    if not MODEL_TS.exists():
        print(f"Skip model.ts patch (missing {MODEL_TS})")
        return
    text = MODEL_TS.read_text(encoding="utf-8")
    replacements = {
        r"const INPUT_SCALE = [^;]+;": f"const INPUT_SCALE = {meta['input_scale']};",
        r"const INPUT_ZERO_POINT = [^;]+;": f"const INPUT_ZERO_POINT = {meta['input_zero_point']};",
        r"const OUTPUT_SCALE = [^;]+;": f"const OUTPUT_SCALE = {meta['output_scale']};",
        r"const OUTPUT_ZERO_POINT = [^;]+;": f"const OUTPUT_ZERO_POINT = {meta['output_zero_point']};",
        r'const INPUT_NAME = "[^"]+";': f'const INPUT_NAME = "{meta["input_name"]}";',
    }
    new = text
    for pat, rep in replacements.items():
        new2, n = re.subn(pat, rep, new, count=1)
        if n != 1:
            print(f"Warning: failed to patch pattern {pat!r}")
        new = new2
    if new != text:
        MODEL_TS.write_text(new, encoding="utf-8")
        print(f"Patched {MODEL_TS}", flush=True)
    else:
        print("model.ts already up to date", flush=True)


def sync_web_copies(tflite_path: Path, onnx_path: Path) -> None:
    WEB_MODEL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tflite_path, WEB_MODEL / "project-kaan.tflite")
    shutil.copy2(onnx_path, WEB_MODEL / "project-kaan.onnx")
    extras = [
        ROOT / "web" / "android" / "app" / "src" / "main" / "assets" / "public" / "model",
        ROOT / "web" / "ios" / "App" / "App" / "public" / "model",
    ]
    for d in extras:
        if d.is_dir():
            shutil.copy2(tflite_path, d / "project-kaan.tflite")
            shutil.copy2(onnx_path, d / "project-kaan.onnx")
            print(f"Synced → {d}", flush=True)
    print(f"Synced → {WEB_MODEL}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=Path, default=MODEL_DIR / "project-kaan_model.h5")
    p.add_argument("--tflite", type=Path, default=MODEL_DIR / "project-kaan.tflite")
    p.add_argument("--onnx", type=Path, default=WEB_MODEL / "project-kaan.onnx")
    p.add_argument("--skip-web-patch", action="store_true")
    args = p.parse_args()
    if not args.h5.exists():
        raise SystemExit(f"Missing {args.h5}; run python -m model.distill first")

    meta = convert_tflite(args.h5, args.tflite)
    convert_onnx(args.tflite, args.onnx)
    if not args.skip_web_patch:
        patch_model_ts(meta)
    sync_web_copies(args.tflite, args.onnx)
    meta_path = MODEL_DIR / "distill_artifacts" / "export_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print("Export complete.", flush=True)


if __name__ == "__main__":
    main()
