# Experiments

Paper evidence pipeline for Project Kaan. The live app is under `web/`. This folder runs same-split model comparisons, ablations, audits, and figures.

## Research question

Under a leakage-aware file-level split, do multiple approaches (deep mel-CNN and classical models) exceed the cited ~84.51% reference accuracy of Balingbing et al. (2024)?

## Models compared

| ID | Method | Features |
|---|---|---|
| `cnn_shallow` | Project Kaan mel-CNN (`model/train.py`) | 128x128 mel image |
| `cnn_deep` | Deeper mel-CNN v5 (`model/train_kaggle.py`) | 128x128 mel image |
| `svm_rbf` | SVM (RBF) | MFCC + spectral summary vector |
| `mlp` | Shallow MLP | Same handcrafted vector |
| `gbdt` | HistGradientBoosting | Same handcrafted vector |
| `logreg` | Logistic regression (optional floor) | Same handcrafted vector |

The app still ships the existing INT8 / ONNX CNN only. This folder compares approaches on the same file-level split; it does not replace the production model path.

## Data layout

```
data/clean/*.wav
data/rice_weevil/*.wav
data/lesser_grain_borer/*.wav
data/red_flour_beetle/*.wav
```

See `data/HOW_TO_GET_DATA.md`. Use file-level splits only.

## Quick start

```bash
pip install -r requirements.txt
pip install tensorflow>=2.13.0   # optional; needed for CNN approaches

python -m experiments.run_benchmark --smoke
python -m experiments.run_benchmark --seed 42 --out experiments/outputs/run_seed42
python -m experiments.run_benchmark --seeds 42,43,44 --out experiments/outputs/multi
python -m experiments.run_ablations --out experiments/outputs/ablations
```

### Kaggle

Full IRRI data + GPU: see `experiments/KAGGLE.md`.

```bash
bash experiments/kaggle/push_and_run.sh
kaggle kernels status arnavd371/kaan-multi-approach-benchmark
```

Data prep matches `model/train_kaggle.py` (IRRI clone + Speech Commands clean windows).

## Outputs

- `metrics.json` / `metrics.csv`
- `stats.md` / `stats.json` / `aggregate_metrics.json` (multi-seed)
- `per_seed_metrics.json`
- `audit_before_training.*` / `audit_after_training.*`
- `ablation_table.*` (ablation runs)
- `confusion_*.png`, `fig_*.png`
- `split_manifest.json`
- `report.md`

## Workshop checklist

See `experiments/WORKSHOP_CHECKLIST.md`.

## Licence

Apache License 2.0 (same as the Project Kaan monorepo).
