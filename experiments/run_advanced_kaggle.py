"""Kaggle entry: prepare data + multi-seed advanced suite + aggregate CIs + results page."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
DATA_ROOT = TEMP / "kaan_data"
OUT = WORK / "experiments" / "outputs" / "advanced_multiseed"


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
    if not (src / "experiments" / "run_advanced_multiseed.py").exists():
        raise SystemExit("embedded src missing run_advanced_multiseed.py")
    sys.path.insert(0, str(src))

    data_dir = DATA_ROOT / "data"
    if os.environ.get("SKIP_PREPARE") != "1":
        cmd = [sys.executable, "-m", "experiments.prepare_kaggle_data", "--out", str(DATA_ROOT)]
        env = {**os.environ, "PYTHONPATH": str(src)}
        subprocess.run(cmd, check=True, cwd=str(src), env=env)
    os.environ["PROJECT_KAAN_DATA_DIR"] = str(data_dir)
    os.environ["EXPERIMENTS_DATA_DIR"] = str(data_dir)

    epochs = os.environ.get("EPOCHS", "60")
    ssl_ep = os.environ.get("SSL_PRETRAIN_EPOCHS", "30")
    seeds = os.environ.get("SEEDS", os.environ.get("SEED", "42,43,44"))
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "experiments.run_advanced_multiseed",
        "--seeds",
        seeds,
        "--epochs",
        epochs,
        "--ssl-pretrain-epochs",
        ssl_ep,
        "--out",
        str(OUT),
        "--copy-results",
    ]
    env = {**os.environ, "PYTHONPATH": str(src)}
    subprocess.run(cmd, check=True, cwd=str(src), env=env)

    subprocess.run(
        [sys.executable, "-m", "experiments.build_results_page"],
        check=True,
        cwd=str(src),
        env=env,
    )

    for name in ("advanced_aggregate.md", "advanced_aggregate.json"):
        p = OUT / name
        if p.exists():
            shutil.copy2(p, WORK / name)
    dest_res = src / "experiments" / "results" / "advanced_multiseed"
    if dest_res.is_dir():
        shutil.copytree(dest_res, WORK / "experiments" / "results" / "advanced_multiseed", dirs_exist_ok=True)
        for f in dest_res.glob("*"):
            if f.is_file():
                shutil.copy2(f, WORK / f.name)
    html = src / "experiments" / "results" / "index.html"
    if html.exists():
        shutil.copy2(html, WORK / "index.html")
    print("[advanced-multiseed] DONE", flush=True)


if __name__ == "__main__":
    main()
