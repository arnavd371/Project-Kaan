# Kaggle: full-data multi-approach benchmark

This is the intended path for paper numbers when local `data/` has no WAVs.

## What runs on Kaggle

1. **Data prep** (`experiments/prepare_kaggle_data.py`): same sources as `model/train_kaggle.py`:
  - Clone [IRRI Rice-Acoustic-Sensor](https://github.com/cbalingbing/Rice-Acoustic-Sensor) → pest class WAVs
  - Speech Commands `_background_noise_` → 10 s `clean/` windows
2. **Before-training audit** (`experiments/audit.py`): class counts, file-level leakage, byte-identical WAV twins, exact mel train↔val dupes, feature shapes, TF/GPU
3. **Benchmark** (`experiments.run_benchmark`): one stratified **file-level** split for:
  - `cnn_shallow`, `cnn_deep` (preserved Project Kaan CNNs, **strong recipe** by default)
  - `svm_rbf`, `mlp`, `gbdt`, `logreg`
4. **After-training audit**: prediction diversity, vs **84.51%** reference, confusion diagonals, CNN learning-curve summary
5. **Outputs** under `/kaggle/working/experiments/outputs/kaggle_run/`:
  - `metrics.json` / `metrics.csv`
  - `report.md`
  - `audit_before_training.{json,md}`, `audit_after_training.{json,md}`
  - `split_manifest.json`
  - `confusion_*.png`, `fig_*.png`
  - Copies: `/kaggle/working/benchmark_report.md`, `/kaggle/working/metrics.csv`, audit files

Reference to beat: **84.51%** (Balingbing et al.).

**Strong CNN recipe** (default, mirrors `train_kaggle.py`): SpecAugment, cosine LR, label smoothing 0.05, balanced class weights (rice_weevil×1.15, lesser_grain_borer×1.25), early stopping patience 15, `CNN_EPOCHS=60`.

Production `web/` / TFLite / ONNX paths are **not** modified.

## Files

| Path | Role |
|---|---|
| `experiments/prepare_kaggle_data.py` | Build `data/{class}/*.wav` on Kaggle or locally |
| `experiments/audit.py` | Before/after training audits |
| `experiments/run_benchmark_kaggle.py` | Kaggle orchestrator (prepare + audit + benchmark) |
| `experiments/kaggle/kaan-multi-approach-benchmark.py` | Self-contained kernel (embedded code zip) |
| `experiments/kaggle/kernel-metadata.json` | GPU + internet kernel metadata |
| `experiments/kaggle/push_and_run.sh` | Rebuild embed + `kaggle kernels push` |

Optional code dataset (not required by the self-contained kernel):
`arnavd371/project-kaan-experiments-code`

## Push / run (CLI)

Requires authenticated Kaggle CLI (`~/.kaggle/kaggle.json`).

```bash
cd "/path/to/Project Kaan"
bash experiments/kaggle/push_and_run.sh

# Monitor
kaggle kernels status arnavd371/kaan-multi-approach-benchmark

# After COMPLETE, download artifacts (avoid WAVs)
kaggle kernels output arnavd371/kaan-multi-approach-benchmark -p /tmp/kaan-bench-out \
  --file-pattern '(metrics\.(csv|json)|report\.md|benchmark_report\.md|audit_.*\.(md|json)|.*\.log|fig_.*\.png|confusion_.*\.png|split_manifest\.json)'
```

Kernel URL: https://www.kaggle.com/code/arnavd371/kaan-multi-approach-benchmark

Enable **GPU** + **Internet** (set in `kernel-metadata.json`). Internet is required to clone IRRI and download Speech Commands.

## Env knobs (kernel)

| Env | Default | Meaning |
|---|---|---|
| `SEED` | `42` | Split seed |
| `SEEDS` | empty | Comma list → multi-seed aggregate |
| `CNN_EPOCHS` | `60` | Max CNN train epochs (early stopping may stop sooner) |
| `CNN_BASELINE` | unset | `1` → disable SpecAugment/class-weights/cosine |
| `AUDIT_SOFT` | unset | `1` → do not abort on before-audit FAIL |
| `MODELS` | all | Subset, e.g. `cnn_deep,svm_rbf,gbdt` |
| `PINK_EXTRA` | `0` | Extra synthetic pink clean WAVs |
| `SKIP_PREPARE` | unset | `1` to reuse existing `data/` |

## Latest seed-42 results (kernel v4, strong CNN, COMPLETE)

Artifacts: `experiments/outputs/kaggle_run_strong/`.

| Approach | Acc | Macro F1 | Beats 84.51%? |
|---|---:|---:|:---:|
| **cnn_deep** | **97.53%** | **0.976** | yes |
| gbdt | 95.68% | 0.961 | yes |
| svm_rbf | 95.37% | 0.959 | yes |
| logreg | 94.44% | 0.955 | yes |
| cnn_shallow | 92.90% | 0.939 | yes |
| mlp | 91.05% | 0.927 | yes |

**Before audit:** file-level leakage PASS (0 path overlap); WARN: 12 byte-identical / exact-mel train↔val twins (IRRI dupes: same issue `train_kaggle.py` AUDIT 3 targets).  
**After audit:** PASS=20 FAIL=0; all six approaches beat the reference.

## Latest seed-42 results (kernel v5, strong CNN + byte-dedupe, COMPLETE)

Artifacts: `experiments/outputs/kaggle_run_deduped/`.

Byte-dedupe removed **40** identical WAV contents (1619 → 1579) before the split.

| Approach | Acc | Macro F1 | Beats 84.51%? |
|---|---:|---:|:---:|
| **gbdt** | **95.89%** | **0.967** | yes |
| cnn_deep | 95.57% | 0.965 | yes |
| svm_rbf | 95.57% | 0.965 | yes |
| logreg | 94.30% | 0.954 | yes |
| mlp | 91.77% | 0.934 | yes |
| cnn_shallow | 35.13% | 0.254 | no (early-stopped @16; unstable under SpecAugment) |

**Before audit:** PASS=6 WARN=0 FAIL=0: file leakage, byte twins, and exact-mel train↔val all **PASS**.  
**After audit:** 5/6 approaches beat the reference (`cnn_shallow` WARN).

## Latest multi-seed stats (kernel v6, seeds 42/43/44, COMPLETE)

Artifacts: `experiments/outputs/kaggle_run_stats/` (`stats.md`, `stats.json`, `per_seed_metrics.json`).

| Approach | Acc mean±std | Acc 95% CI | Seeds > 84.51% | t p (>) | CI > ref |
|---|---:|---:|---:|---:|:---:|
| gbdt | 95.36%±1.50 | [93.67, 96.52] | 3/3 | 0.0031 | yes |
| svm_rbf | 94.73%±1.20 | [93.35, 95.57] | 3/3 | 0.0023 | yes |
| cnn_deep | 94.62%±0.95 | [93.67, 95.57] | 3/3 | 0.0015 | yes |
| logreg | 94.09%±1.59 | [92.41, 95.57] | 3/3 | 0.0045 | yes |
| cnn_shallow | 93.78%±0.97 | [92.72, 94.62] | 3/3 | 0.0018 | yes |
| mlp | 92.09%±0.32 | [91.77, 92.41] | 3/3 | 0.0003 | yes |

Wilcoxon one-sided p=0.125 for all (minimum with n=3 and all positive diffs). Cite mean±std and bootstrap CIs as primary; t-tests as secondary.

## Latest CNN ablations (kernel v7, seed 42, COMPLETE)

Artifacts: `experiments/outputs/ablations/`. One GitHub release per version.

| Version | Tag | cnn_deep Acc | cnn_shallow Acc |
|---|---|---:|---:|
| full | [v1.1.1-ablate-full](https://github.com/arnavd371/Project-Kaan/releases/tag/v1.1.1-ablate-full) | 95.57% | 93.99% |
| − SpecAugment | [v1.1.2](https://github.com/arnavd371/Project-Kaan/releases/tag/v1.1.2-ablate-no-specaugment) | 96.20% | 97.15% |
| − class weights | [v1.1.3](https://github.com/arnavd371/Project-Kaan/releases/tag/v1.1.3-ablate-no-class-weights) | 96.52% | 94.62% |
| − label smoothing | [v1.1.4](https://github.com/arnavd371/Project-Kaan/releases/tag/v1.1.4-ablate-no-label-smoothing) | 96.52% | 93.99% |
| baseline | [v1.1.5](https://github.com/arnavd371/Project-Kaan/releases/tag/v1.1.5-ablate-baseline) | 7.59% | 94.94% |

Summary table: [v1.1.6-ablation-summary](https://github.com/arnavd371/Project-Kaan/releases/tag/v1.1.6-ablation-summary).  
Baseline `cnn_deep` collapse shows the full training recipe matters for the deep net on this split; single-seed ablation deltas should not be over-interpreted.

## Local prepare (optional)

```bash
python -m experiments.prepare_kaggle_data --out .
python -m experiments.run_benchmark --seed 42 --out experiments/outputs/run_seed42
```

## Existing related kernel

Production CNN retrain (not the multi-approach bake-off):

`arnavd371/kaan-grainear-acoustic-pest-cnn-training` (`model/train_kaggle.py`)
