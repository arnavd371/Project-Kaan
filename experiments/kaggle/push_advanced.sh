#!/usr/bin/env bash
# Push advanced suite (robustness / hierarchical / calibration / SSL / results page) to Kaggle GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD=/tmp/kaan-advanced-embed
KERNEL=/tmp/kaan-advanced-kernel

rm -rf "$BUILD"
mkdir -p "$BUILD/pkg/experiments/results" "$BUILD/pkg/model"
rsync -a --exclude '__pycache__' --exclude 'outputs' --exclude 'kaggle' \
  "$ROOT/experiments/" "$BUILD/pkg/experiments/"
# Keep committed results for the HTML dashboard
rsync -a "$ROOT/experiments/results/" "$BUILD/pkg/experiments/results/"
cp "$ROOT/model/__init__.py" "$ROOT/model/preprocess.py" "$ROOT/model/generate_clean_data.py" \
   "$BUILD/pkg/model/"

export ROOT
python3 - <<'PY'
import base64, io, json, zipfile, os
from pathlib import Path

ROOT = Path(os.environ["ROOT"])
root = Path("/tmp/kaan-advanced-embed/pkg")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix != ".npz":
            zf.write(p, p.relative_to(root).as_posix())
b64 = base64.b64encode(buf.getvalue()).decode("ascii")

script = f'''"""Kaan advanced suite: robustness, hierarchical, calibration, SSL, results HTML."""
from __future__ import annotations
import base64, io, os, runpy, shutil, sys, zipfile
from pathlib import Path

WORK_SRC = Path("/kaggle/temp/_kaan_src")
_CODE_ZIP_B64 = """{b64}"""

def _extract() -> Path:
    if WORK_SRC.exists():
        shutil.rmtree(WORK_SRC)
    WORK_SRC.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(_CODE_ZIP_B64)), "r") as zf:
        zf.extractall(WORK_SRC)
    (WORK_SRC / "model" / "__init__.py").touch(exist_ok=True)
    (WORK_SRC / "experiments" / "__init__.py").touch(exist_ok=True)
    sys.path.insert(0, str(WORK_SRC))
    print(f"[embed] {{WORK_SRC}}", flush=True)
    return WORK_SRC

if __name__ == "__main__":
    root = _extract()
    os.environ.setdefault("SEED", "42")
    os.environ.setdefault("EPOCHS", "60")
    os.environ.setdefault("SSL_PRETRAIN_EPOCHS", "30")
    runpy.run_path(str(root / "experiments" / "run_advanced_kaggle.py"), run_name="__main__")
'''
out = ROOT / "experiments/kaggle/kaan-advanced-suite.py"
out.write_text(script)
print("wrote", out, "bytes", out.stat().st_size)
meta = {
    "id": "arnavd371/kaan-advanced-suite",
    "title": "kaan-advanced-suite",
    "code_file": "kaan-advanced-suite.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(ROOT / "experiments/kaggle/kernel-metadata-advanced.json").write_text(json.dumps(meta, indent=2) + "\n")
PY

rm -rf "$KERNEL"
mkdir -p "$KERNEL"
cp "$ROOT/experiments/kaggle/kernel-metadata-advanced.json" "$KERNEL/kernel-metadata.json"
cp "$ROOT/experiments/kaggle/kaan-advanced-suite.py" "$KERNEL/"
kaggle kernels push -p "$KERNEL"
echo "Open: https://www.kaggle.com/code/arnavd371/kaan-advanced-suite"
kaggle kernels status arnavd371/kaan-advanced-suite || true
