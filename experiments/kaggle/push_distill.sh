#!/usr/bin/env bash
# Rebuild self-contained distill kernel and push to Kaggle (GPU T4 + internet).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD=/tmp/kaan-distill-embed
KERNEL=/tmp/kaan-distill-kernel

rm -rf "$BUILD"
mkdir -p "$BUILD/pkg/experiments" "$BUILD/pkg/model"
rsync -a --exclude '__pycache__' --exclude 'outputs' --exclude 'kaggle' \
  --exclude 'results' --exclude 'distill_artifacts' \
  "$ROOT/experiments/" "$BUILD/pkg/experiments/"
cp "$ROOT/model/__init__.py" \
   "$ROOT/model/preprocess.py" \
   "$ROOT/model/generate_clean_data.py" \
   "$ROOT/model/train.py" \
   "$ROOT/model/distill.py" \
   "$ROOT/model/export_deploy.py" \
   "$ROOT/model/convert_tflite.py" \
   "$BUILD/pkg/model/"

export ROOT
python3 - <<'PY'
import base64, io, json, zipfile, os
from pathlib import Path

ROOT = Path(os.environ["ROOT"])
root = Path("/tmp/kaan-distill-embed/pkg")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in root.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(root).as_posix())
b64 = base64.b64encode(buf.getvalue()).decode("ascii")

script = f'''"""Self-contained Kaggle kernel: Project Kaan production distillation.

GPU + Internet required (IRRI clone + Speech Commands).
Teachers: gbdt + extratrees + cnn_deep → deep mel-CNN student → INT8 TFLite + ONNX.
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
    entry = WORK_SRC / "experiments" / "run_distill_kaggle.py"
    if not entry.exists():
        raise SystemExit(f"embedded code missing entry: {{entry}}; tree={{list(WORK_SRC.rglob('*.py'))}}")
    sys.path.insert(0, str(WORK_SRC))
    print(f"[embed] extracted to {{WORK_SRC}} ({{len(list(WORK_SRC.rglob('*')))}} paths)", flush=True)
    return WORK_SRC


if __name__ == "__main__":
    root = _extract()
    os.environ.pop("PROJECT_KAAN_DATA_DIR", None)
    os.environ.setdefault("SEED", "42")
    os.environ.setdefault("TEACHER_EPOCHS", "60")
    os.environ.setdefault("STUDENT_EPOCHS", "60")
    os.environ.setdefault("TEMPERATURE", "2.0")
    os.environ.setdefault("ALPHA", "0.5")
    os.environ.setdefault("TEACHERS", "gbdt,extratrees,cnn_deep")
    runpy.run_path(str(root / "experiments" / "run_distill_kaggle.py"), run_name="__main__")
'''
out = ROOT / "experiments/kaggle/kaan-distill-production.py"
out.write_text(script)
print("wrote", out, "bytes", out.stat().st_size)

meta = {
    "id": "arnavd371/kaan-distill-production",
    "title": "kaan-distill-production",
    "code_file": "kaan-distill-production.py",
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
(ROOT / "experiments/kaggle/kernel-metadata-distill.json").write_text(json.dumps(meta, indent=2) + "\n")
PY

rm -rf "$KERNEL"
mkdir -p "$KERNEL"
cp "$ROOT/experiments/kaggle/kernel-metadata-distill.json" "$KERNEL/kernel-metadata.json"
cp "$ROOT/experiments/kaggle/kaan-distill-production.py" "$KERNEL/"
kaggle kernels push -p "$KERNEL"
echo "Open: https://www.kaggle.com/code/arnavd371/kaan-distill-production"
kaggle kernels status arnavd371/kaan-distill-production || true
