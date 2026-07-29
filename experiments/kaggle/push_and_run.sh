#!/usr/bin/env bash
# Rebuild the self-contained embedded kernel and push it to Kaggle (GPU + internet).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD=/tmp/kaan-embed
KERNEL=/tmp/kaan-bench-kernel

rm -rf "$BUILD"
mkdir -p "$BUILD/pkg/experiments" "$BUILD/pkg/model"
rsync -a --exclude '__pycache__' --exclude 'outputs' --exclude 'kaggle' \
  "$ROOT/experiments/" "$BUILD/pkg/experiments/"
cp "$ROOT/model/__init__.py" "$ROOT/model/preprocess.py" "$ROOT/model/generate_clean_data.py" \
  "$BUILD/pkg/model/"

python3 - <<PY
import base64, io, json, zipfile
from pathlib import Path

root = Path("$BUILD/pkg")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in root.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(root).as_posix())
b64 = base64.b64encode(buf.getvalue()).decode("ascii")

script = f'''"""Self-contained Kaggle kernel: Project Kaan multi-approach benchmark.

GPU + Internet required (IRRI clone + Speech Commands download).
No external code dataset required  - experiments/ + model/preprocess are embedded.
"""
from __future__ import annotations

import base64
import io
import os
import runpy
import shutil
import sys
import zipfile
from pathlib import Path

WORK_SRC = Path("/kaggle/temp/_kaan_src")
_CODE_ZIP_B64 = """{b64}"""


def _extract() -> Path:
    WORK_SRC.parent.mkdir(parents=True, exist_ok=True)
    if WORK_SRC.exists():
        shutil.rmtree(WORK_SRC)
    WORK_SRC.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(_CODE_ZIP_B64)
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        zf.extractall(WORK_SRC)
    (WORK_SRC / "model" / "__init__.py").touch(exist_ok=True)
    (WORK_SRC / "experiments" / "__init__.py").touch(exist_ok=True)
    entry = WORK_SRC / "experiments" / "run_benchmark_kaggle.py"
    if not entry.exists():
        raise SystemExit(f"embedded code missing entry: {{entry}}; tree={{list(WORK_SRC.rglob('*.py'))}}")
    sys.path.insert(0, str(WORK_SRC))
    print(f"[embed] extracted to {{WORK_SRC}} ({{len(list(WORK_SRC.rglob('*')))}} paths)", flush=True)
    return WORK_SRC


if __name__ == "__main__":
    root = _extract()
    os.environ.pop("PROJECT_KAAN_DATA_DIR", None)
    os.environ.setdefault("CNN_EPOCHS", "60")
    os.environ.setdefault("RUN_MODE", "ablations")
    os.environ.pop("SEEDS", None)
    runpy.run_path(str(root / "experiments" / "run_benchmark_kaggle.py"), run_name="__main__")
'''
out = Path("$ROOT/experiments/kaggle/kaan-multi-approach-benchmark.py")
out.write_text(script)
print("wrote", out, "bytes", out.stat().st_size)

meta = {
    "id": "arnavd371/kaan-multi-approach-benchmark",
    "title": "Kaan Multi-Approach Benchmark",
    "code_file": "kaan-multi-approach-benchmark.py",
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
Path("$ROOT/experiments/kaggle/kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\\n")
PY

rm -rf "$KERNEL"
mkdir -p "$KERNEL"
cp "$ROOT/experiments/kaggle/kernel-metadata.json" "$KERNEL/"
cp "$ROOT/experiments/kaggle/kaan-multi-approach-benchmark.py" "$KERNEL/"
kaggle kernels push -p "$KERNEL"
echo "Open: https://www.kaggle.com/code/arnavd371/kaan-multi-approach-benchmark"
kaggle kernels status arnavd371/kaan-multi-approach-benchmark || true
