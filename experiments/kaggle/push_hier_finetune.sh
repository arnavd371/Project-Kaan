#!/usr/bin/env bash
# Push hierarchical fine-tune kernel (baseline + FT head vs scratch ablation).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD=/tmp/kaan-hier-embed
KERNEL=/tmp/kaan-hier-kernel

rm -rf "$BUILD"
mkdir -p "$BUILD/pkg/experiments" "$BUILD/pkg/model"
rsync -a --exclude '__pycache__' --exclude 'outputs' --exclude 'kaggle' --exclude 'results' \
  "$ROOT/experiments/" "$BUILD/pkg/experiments/"
cp "$ROOT/model/__init__.py" "$ROOT/model/preprocess.py" "$ROOT/model/generate_clean_data.py" \
   "$BUILD/pkg/model/"

export ROOT
python3 - <<'PY'
import base64, io, json, zipfile, os
from pathlib import Path
ROOT = Path(os.environ["ROOT"])
root = Path("/tmp/kaan-hier-embed/pkg")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in root.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(root).as_posix())
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
script = f'''"""Kaan hierarchical fine-tune: specialist head on baseline encoder."""
from __future__ import annotations
import base64, io, os, runpy, shutil, sys, zipfile
from pathlib import Path
WORK_SRC = Path("/kaggle/temp/_kaan_src")
_CODE_ZIP_B64 = """{b64}"""
def _extract():
    if WORK_SRC.exists(): shutil.rmtree(WORK_SRC)
    WORK_SRC.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(_CODE_ZIP_B64)), "r") as zf:
        zf.extractall(WORK_SRC)
    (WORK_SRC/"model"/"__init__.py").touch(exist_ok=True)
    (WORK_SRC/"experiments"/"__init__.py").touch(exist_ok=True)
    sys.path.insert(0, str(WORK_SRC))
    return WORK_SRC
if __name__ == "__main__":
    root = _extract()
    os.environ.setdefault("SEED", "42")
    os.environ.setdefault("EPOCHS", "60")
    runpy.run_path(str(root/"experiments"/"run_hier_finetune_kaggle.py"), run_name="__main__")
'''
out = ROOT / "experiments/kaggle/kaan-hier-finetune.py"
out.write_text(script)
print("wrote", out, out.stat().st_size)
meta = {
    "id": "arnavd371/kaan-hier-finetune",
    "title": "kaan-hier-finetune",
    "code_file": "kaan-hier-finetune.py",
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
(ROOT / "experiments/kaggle/kernel-metadata-hier.json").write_text(json.dumps(meta, indent=2) + "\n")
PY

rm -rf "$KERNEL" && mkdir -p "$KERNEL"
cp "$ROOT/experiments/kaggle/kernel-metadata-hier.json" "$KERNEL/kernel-metadata.json"
cp "$ROOT/experiments/kaggle/kaan-hier-finetune.py" "$KERNEL/"
kaggle kernels push -p "$KERNEL"
echo "Open: https://www.kaggle.com/code/arnavd371/kaan-hier-finetune"
kaggle kernels status arnavd371/kaan-hier-finetune || true
