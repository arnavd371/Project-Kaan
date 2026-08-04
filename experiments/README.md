# Experiments

Same-split model comparisons, audits, ablations, and figures for Project Kaan.
The live app is under `web/`.

## Layout

| Path | Role |
|---|---|
| `run_benchmark.py` | Multi-approach benchmark entry |
| `run_ablations.py` | CNN recipe ablations |
| `run_advanced.py` | Robustness / hierarchical / calibration / SSL |
| `run_advanced_kaggle.py` | Kaggle orchestrator for advanced suite |
| `run_benchmark_kaggle.py` | Kaggle orchestrator |
| `robustness.py` | Phone-like degradation ladder |
| `hierarchical.py` | Cost-sensitive + weevil↔borer hierarchy |
| `calibration.py` | Temperature scaling + abstain curves |
| `ssl_pretrain.py` | SimCLR mel pretrain → fine-tune |
| `build_results_page.py` | Static `results/index.html` dashboard |
| `models.py` | Approach builders / trainers |
| `features.py` | Mel + handcrafted + waveforms |
| `audit.py` / `stats.py` / `findings.py` / `plots.py` | Checks, stats, reports, figures |
| `prepare_kaggle_data.py` | IRRI + clean window prep |
| `kaggle/` | Embedded kernel + push script |
| `results/` | Committed multi-seed summary tables |
| `outputs/` | Local run artifacts (gitignored) |

## Research question

Under a leakage-aware file-level split, do multiple approaches exceed the cited ~84.51% reference accuracy of Balingbing et al. (2024)?

## Models

| ID | Method | Features |
|---|---|---|
| `cnn_shallow` | Mel-CNN (`model/train.py`) | 128×128 mel |
| `cnn_deep` | Mel-CNN v5 (`model/train_kaggle.py`) | 128×128 mel |
| `cnn1d` | 1D CNN on mel time | Mel sequence |
| `yamnet_probe` | Frozen YAMNet + logistic | Waveform |
| `svm_rbf` | RBF SVM | Handcrafted ~74-D |
| `mlp` | MLP | Handcrafted |
| `gbdt` | HistGradientBoosting | Handcrafted |
| `rf` | RandomForest | Handcrafted |
| `extratrees` | ExtraTrees | Handcrafted |
| `knn` | k-NN | Handcrafted |
| `logreg` | Logistic regression | Handcrafted |

The app still ships the existing INT8 / ONNX CNN only.

## Data

```
data/clean/*.wav
data/rice_weevil/*.wav
data/lesser_grain_borer/*.wav
data/red_flour_beetle/*.wav
```

See `data/HOW_TO_GET_DATA.md`. Use file-level splits only. Byte-dedupe before split.

## Run

```bash
pip install -r requirements.txt
pip install 'tensorflow>=2.13.0'   # CNNs
# optional: tensorflow_hub           # yamnet_probe

python -m experiments.run_benchmark --smoke
python -m experiments.run_benchmark --seeds 42,43,44 --out experiments/outputs/multi
python -m experiments.run_ablations --out experiments/outputs/ablations

# Advanced suite (prefer Kaggle GPU for full IRRI):
python -m experiments.run_advanced --smoke
bash experiments/kaggle/push_advanced.sh
python -m experiments.build_results_page
```

Kaggle GPU (bake-off):

```bash
bash experiments/kaggle/push_and_run.sh
```

See `KAGGLE.md`.

## Outputs

Per run directory: `metrics.*`, `stats.*`, `findings.*`, audits, plots, `split_manifest.json`, `report.md`.  
Committed summaries: `results/`.

## Licence

Apache License 2.0.
