"""Kaggle entrypoint: prepare IRRI+clean WAVs, then run multi-approach benchmark.

Env: SEED/SEEDS, CNN_EPOCHS, MODELS, PINK_EXTRA, SKIP_PREPARE, CNN_BASELINE,
AUDIT_SOFT, RUN_MODE=ablations, ABLATION.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
INPUT_CANDIDATES = [
    Path("/kaggle/input/project-kaan-experiments-code"),
    Path("/kaggle/input/project-kaan-experiments"),
    Path("/kaggle/input/datasets/arnavd371/project-kaan-experiments-code"),
    Path("/kaggle/input/datasets/arnavd371/project-kaan-experiments"),
]
PROJECT = WORK / "Project-Kaan"
DATA_ROOT = TEMP / "kaan_data"
OUT_DIR = WORK / "experiments" / "outputs" / "kaggle_run"


def _pip_install() -> None:
    pkgs = [
        "librosa",
        "soundfile",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "pandas",
        "joblib",
        "tensorflow_hub",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
        try:
        import tensorflow as tf  # noqa: F401

        print("TensorFlow already available:", tf.__version__, flush=True)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorflow"], check=True)


def _find_code_input() -> Path | None:
    for p in INPUT_CANDIDATES:
        if p.is_dir():
            return p
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for child in sorted(input_root.iterdir()):
            if not child.is_dir():
                continue
            if (
                (child / "experiments").is_dir()
                or (child / "experiments.zip").exists()
                or (child / "experiments.tar").exists()
            ):
                return child
    return None


def _materialize_from_input(src: Path, dest: Path) -> None:
    import tarfile
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    if (src / "experiments" / "run_benchmark_kaggle.py").exists():
        for name in ("experiments", "model"):
            s = src / name
            d = dest / name
            if not s.exists():
                continue
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
        return

    for name in ("experiments", "model"):
        d = dest / name
        if d.exists():
            shutil.rmtree(d)
        zpath = src / f"{name}.zip"
        tpath = src / f"{name}.tar"
        if zpath.exists():
            print(f"[code] unzip {zpath.name}", flush=True)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(dest)
        elif tpath.exists():
            print(f"[code] untar {tpath.name}", flush=True)
            with tarfile.open(tpath, "r") as tf:
                tf.extractall(dest)
        elif (src / name).is_dir():
            shutil.copytree(src / name, d)


def _sync_project_code() -> Path:
    PROJECT.mkdir(parents=True, exist_ok=True)
        local = Path(__file__).resolve().parent.parent
    if (local / "experiments").is_dir() and (local / "model" / "preprocess.py").exists():
        print(f"[code] using local tree: {local}", flush=True)
        return local
    src = _find_code_input()
    if src is None:
        raise SystemExit(
            "No experiments code found. Use the embedded Kaggle kernel or attach "
            "'arnavd371/project-kaan-experiments-code'."
        )

    print(f"[code] syncing from {src} → {PROJECT}", flush=True)
    _materialize_from_input(src, PROJECT)
    if not (PROJECT / "experiments" / "run_benchmark_kaggle.py").exists():
        hits = list(PROJECT.rglob("run_benchmark_kaggle.py"))
        raise SystemExit(f"experiments code missing after sync; found={hits}")
    (PROJECT / "model").mkdir(exist_ok=True)
    (PROJECT / "model" / "__init__.py").touch(exist_ok=True)
    (PROJECT / "experiments" / "__init__.py").touch(exist_ok=True)
    return PROJECT


def main() -> None:
    print("=== Project Kaan multi-approach benchmark (Kaggle) ===", flush=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    _pip_install()

    try:
        import tensorflow as tf

        print("TensorFlow:", tf.__version__, flush=True)
        print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    except Exception as e:
        print("TensorFlow import warning:", e, flush=True)

    project = _sync_project_code()
    sys.path.insert(0, str(project))
    os.chdir(project)

        if TEMP.exists():
        data_parent = DATA_ROOT
    else:
        data_parent = project
    data_dir = data_parent / "data"
    skip_prepare = os.environ.get("SKIP_PREPARE", "").strip() in {"1", "true", "True"}
    if not skip_prepare:
        from experiments.prepare_kaggle_data import prepare_all

        pink_extra = int(os.environ.get("PINK_EXTRA", "0"))
        prepare_all(
            out_root=data_parent,
            scratch=TEMP / "kaan_scratch" if TEMP.exists() else project / "_scratch",
            pink_extra=pink_extra,
        )
    else:
        print(f"[prepare] skipped; expecting WAVs under {data_dir}", flush=True)

    seed = os.environ.get("SEED", "42")
    seeds = os.environ.get("SEEDS", "").strip()
    cnn_epochs = os.environ.get("CNN_EPOCHS", "60")
    models = os.environ.get("MODELS", "").strip()
    cnn_baseline = os.environ.get("CNN_BASELINE", "").strip() in {"1", "true", "True"}
    audit_soft = os.environ.get("AUDIT_SOFT", "").strip() in {"1", "true", "True"}
    run_mode = os.environ.get("RUN_MODE", "benchmark").strip().lower()

    if run_mode in {"ablation", "ablations"}:
        abl_out = WORK / "experiments" / "outputs" / "ablations"
        cmd = [
            sys.executable,
            "-m",
            "experiments.run_ablations",
            "--data-dir",
            str(data_dir),
            "--out",
            str(abl_out),
            "--cnn-epochs",
            cnn_epochs,
        ]
        if audit_soft:
            cmd.append("--audit-soft")
        versions = os.environ.get("ABLATION", "").strip()
        if versions and versions != "all":
            cmd.extend(["--versions", versions])
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(project))
        for name in ("ablation_table.md", "ablation_table.csv", "ablation_table.json"):
            src = abl_out / name
            if src.exists():
                shutil.copy2(src, WORK / name)
                print(src.read_text(encoding="utf-8")[:4000] if name.endswith(".md") else f"copied {name}", flush=True)
        print(f"\n[done] ablations → {abl_out}", flush=True)
        return

    cmd = [
        sys.executable,
        "-m",
        "experiments.run_benchmark",
        "--data-dir",
        str(data_dir),
        "--out",
        str(OUT_DIR),
        "--cnn-epochs",
        cnn_epochs,
    ]
    if seeds:
        cmd.extend(["--seeds", seeds])
    else:
        cmd.extend(["--seed", seed])
    if models:
        cmd.extend(["--models", models])
    if cnn_baseline:
        cmd.append("--cnn-baseline")
    if audit_soft:
        cmd.append("--audit-soft")

    print("[run]", " ".join(cmd), flush=True)
    print(
        f"[cnn] strong_recipe={not cnn_baseline} epochs≤{cnn_epochs} "
        "(SpecAugment + cosine + label_smooth + class weights)",
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=str(project))

    report = OUT_DIR / "report.md"
    if report.exists():
        shutil.copy2(report, WORK / "benchmark_report.md")
        print(report.read_text(encoding="utf-8")[:4000], flush=True)
    metrics = OUT_DIR / "metrics.csv"
    if metrics.exists():
        shutil.copy2(metrics, WORK / "metrics.csv")
    for name in (
        "audit_before_training.md",
        "audit_after_training.md",
        "audit_before_training.json",
        "audit_after_training.json",
        "dedupe_report.json",
        "stats.md",
        "stats.json",
        "aggregate_metrics.json",
        "per_seed_metrics.json",
        "findings.md",
        "findings.json",
    ):
        src = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, WORK / name)
                for seed_dir in sorted(OUT_DIR.glob("seed_*")):
            s = seed_dir / name
            if s.exists():
                shutil.copy2(s, WORK / f"{seed_dir.name}_{name}")
    print(f"\n[done] outputs → {OUT_DIR}", flush=True)
    print(f"[done] also copied report/metrics/audits/findings to {WORK}", flush=True)


if __name__ == "__main__":
    main()
