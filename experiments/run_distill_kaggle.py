"""Kaggle entrypoint: prepare IRRI+clean WAVs, distill teachers → production CNN, export.

Env:
  SEED              default 42
  TEACHER_EPOCHS    default 60
  STUDENT_EPOCHS    default 60
  TEMPERATURE       default 3.0
  ALPHA             default 0.3
  TEACHERS          default gbdt,extratrees,cnn_deep
  SKIP_PREPARE      set 1 to reuse existing data
  SKIP_EXPORT       set 1 to skip TFLite/ONNX
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
PROJECT = WORK / "Project-Kaan"
DATA_ROOT = TEMP / "kaan_data"
OUT_MODEL = WORK / "model"
OUT_REPORT = WORK / "distill_artifacts"


def _pip_install() -> None:
    pkgs = [
        "librosa",
        "soundfile",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "pandas",
        "joblib",
        "tf2onnx",
        "onnx",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
    try:
        import tensorflow as tf  # noqa: F401

        print("TensorFlow already available:", tf.__version__, flush=True)
        print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorflow"], check=True)


def _ensure_src_on_path() -> Path:
    embed = Path("/kaggle/temp/_kaan_src")
    if (embed / "model" / "distill.py").exists():
        sys.path.insert(0, str(embed))
        return embed
    if (PROJECT / "model" / "distill.py").exists():
        sys.path.insert(0, str(PROJECT))
        return PROJECT
    raise SystemExit("distill.py not found — embed extract failed")


def _prepare(src_root: Path) -> Path:
    data_dir = DATA_ROOT / "data"
    if os.environ.get("SKIP_PREPARE") == "1" and any(data_dir.glob("*/*.wav")):
        print(f"[distill] reuse data at {data_dir}", flush=True)
        return data_dir
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "experiments.prepare_kaggle_data",
        "--out",
        str(DATA_ROOT),
    ]
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(src_root), env=env)
    return data_dir


def main() -> None:
    _pip_install()
    src = _ensure_src_on_path()
    print(f"[distill] src={src}", flush=True)

    data_dir = _prepare(src)
    os.environ["PROJECT_KAAN_DATA_DIR"] = str(data_dir)
    os.environ["EXPERIMENTS_DATA_DIR"] = str(data_dir)

    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)

    seed = os.environ.get("SEED", "42")
    teacher_epochs = os.environ.get("TEACHER_EPOCHS", "60")
    student_epochs = os.environ.get("STUDENT_EPOCHS", "60")
    temperature = os.environ.get("TEMPERATURE", "3.0")
    alpha = os.environ.get("ALPHA", "0.3")
    teachers = os.environ.get("TEACHERS", "gbdt,extratrees,cnn_deep")

    distill_cmd = [
        sys.executable,
        "-m",
        "model.distill",
        "--seed",
        seed,
        "--teacher-epochs",
        teacher_epochs,
        "--student-epochs",
        student_epochs,
        "--temperature",
        temperature,
        "--alpha",
        alpha,
        "--teachers",
        teachers,
        "--out-dir",
        str(OUT_MODEL),
    ]
    print("+", " ".join(distill_cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(distill_cmd, check=True, cwd=str(src), env=env)

    art = OUT_MODEL / "distill_artifacts"
    if art.is_dir():
        for f in art.iterdir():
            if f.is_file() and f.suffix in {".json", ".md"}:
                shutil.copy2(f, OUT_REPORT / f.name)
                shutil.copy2(f, WORK / f.name)

    if os.environ.get("SKIP_EXPORT") == "1":
        print("[distill] SKIP_EXPORT=1", flush=True)
        return

    h5 = OUT_MODEL / "project-kaan_model.h5"
    tflite = OUT_MODEL / "project-kaan.tflite"
    onnx = WORK / "project-kaan.onnx"
    export_cmd = [
        sys.executable,
        "-m",
        "model.export_deploy",
        "--h5",
        str(h5),
        "--tflite",
        str(tflite),
        "--onnx",
        str(onnx),
        "--skip-web-patch",
    ]
    print("+", " ".join(export_cmd), flush=True)
    link = src / "data"
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_dir():
            try:
                if link.is_symlink():
                    link.unlink()
                else:
                    shutil.rmtree(link)
            except Exception:
                pass
    try:
        link.symlink_to(data_dir)
    except Exception:
        if not (src / "data").exists():
            shutil.copytree(data_dir, src / "data", dirs_exist_ok=True)

    subprocess.run(export_cmd, check=True, cwd=str(src), env=env)

    for src_f, name in (
        (h5, "project-kaan_model.h5"),
        (tflite, "project-kaan.tflite"),
        (onnx, "project-kaan.onnx"),
    ):
        if src_f.exists():
            dest = WORK / name
            if src_f.resolve() != dest.resolve():
                shutil.copy2(src_f, dest)
            print(f"[distill] output {dest} ({dest.stat().st_size / 1024:.1f} KB)", flush=True)

    print("[distill] DONE", flush=True)


if __name__ == "__main__":
    main()
