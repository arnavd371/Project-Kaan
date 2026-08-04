"""Kaggle entry: prepare data + run advanced suite + build results page."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
DATA_ROOT = TEMP / "kaan_data"
OUT = WORK / "experiments" / "outputs" / "advanced"


def _pip() -> None:
    pkgs = ["librosa", "soundfile", "scikit-learn", "matplotlib", "scipy", "pandas", "joblib"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
    try:
        import tensorflow as tf

        print("TF", tf.__version__, "GPUs", tf.config.list_physical_devices("GPU"), flush=True)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorflow"], check=True)


def main() -> None:
    _pip()
    src = Path("/kaggle/temp/_kaan_src")
    if not (src / "experiments" / "run_advanced.py").exists():
        raise SystemExit("embedded src missing run_advanced.py")
    sys.path.insert(0, str(src))

    data_dir = DATA_ROOT / "data"
    if os.environ.get("SKIP_PREPARE") != "1":
        cmd = [sys.executable, "-m", "experiments.prepare_kaggle_data", "--out", str(DATA_ROOT)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src)
        subprocess.run(cmd, check=True, cwd=str(src), env=env)
    os.environ["PROJECT_KAAN_DATA_DIR"] = str(data_dir)
    os.environ["EXPERIMENTS_DATA_DIR"] = str(data_dir)

    epochs = os.environ.get("EPOCHS", "60")
    ssl_ep = os.environ.get("SSL_PRETRAIN_EPOCHS", "30")
    seed = os.environ.get("SEED", "42")
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "experiments.run_advanced",
        "--seed",
        seed,
        "--epochs",
        epochs,
        "--ssl-pretrain-epochs",
        ssl_ep,
        "--out",
        str(OUT),
        "--copy-results",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src)
    # copy-results writes under src/experiments/results/advanced — also mirror to WORK
    subprocess.run(cmd, check=True, cwd=str(src), env=env)

    # Build results page using committed + new advanced artifacts
    # Ensure bake-off stats exist in src results (from embed)
    subprocess.run(
        [sys.executable, "-m", "experiments.build_results_page"],
        check=True,
        cwd=str(src),
        env=env,
    )

    # Flatten key outputs for Kaggle UI
    for name in (
        "advanced_report.md",
        "advanced_summary.json",
        "robustness.json",
        "cost_sensitive.json",
        "hierarchical.json",
        "calibration_baseline.json",
        "ssl.json",
    ):
        p = OUT / name
        if p.exists():
            shutil.copy2(p, WORK / name)
    html = src / "experiments" / "results" / "index.html"
    if html.exists():
        shutil.copy2(html, WORK / "index.html")
        dest = WORK / "experiments" / "results"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(html, dest / "index.html")
        adv = src / "experiments" / "results" / "advanced"
        if adv.is_dir():
            shutil.copytree(adv, dest / "advanced", dirs_exist_ok=True)
    print("[advanced] DONE", flush=True)


if __name__ == "__main__":
    main()
