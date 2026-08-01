# Kaan | कान

Acoustic grain pest detector for Indian farmers. Hold a phone against a storage bag, record briefly, and get an on-device class prediction with an actionable advisory in English, Hindi, Marathi, Punjabi, or Telugu.

**Live demo:** https://kaan-web.vercel.app  
**Repo:** https://github.com/arnavd371/Project-Kaan  
**Licence:** Apache License 2.0 (`LICENSE`, `NOTICE`)

## What it does

Kaan classifies storage audio into four classes:

| Class | Pest |
|---|---|
| `clean` | No pest detected |
| `rice_weevil` | *Sitophilus oryzae* |
| `lesser_grain_borer` | *Rhyzopertha dominica* |
| `red_flour_beetle` | *Tribolium castaneum* |

Audio is converted to a mel spectrogram and scored by a compact INT8 CNN (TFLite on native paths; ONNX Runtime Web in the browser). Inference runs locally. No upload is required for classification.

## Problem

India stores over 80 million tonnes of food grain. Insects cause about 1,300 crore rupees in annual storage losses (IGMRI, 2015). These pests often feed inside kernels, so damage is hard to see early. Lab gear and commercial acoustic probes are expensive for smallholders. Kaan targets a free, offline phone workflow for rural storage.

## Results

### Production CNN (audited training run)

| Metric | Value |
|---|---|
| Validation accuracy | 97.76% |
| Macro F1 | 0.98 |
| INT8 TFLite size | ~333 KB |
| Keras H5 size | ~1.3 MB |

Training data: IRRI Rice Acoustic Sensor Dataset (Balingbing et al., 2024), leakage-aware file-level splits.

### Workshop multi-approach bake-off

Same file-level split, byte-deduped IRRI pest WAVs + Speech Commands background windows for `clean`, seeds **42 / 43 / 44**. Reference line cited from Balingbing et al.: **84.51%**.

| Approach | Acc mean ± std | Seeds above 84.51% |
|---|---:|:---:|
| gbdt | 95.36% ± 1.50 | 3/3 |
| cnn_deep | 95.15% ± 0.48 | 3/3 |
| extratrees | 94.94% ± 1.10 | 3/3 |
| svm_rbf | 94.73% ± 1.20 | 3/3 |
| logreg | 94.09% ± 1.59 | 3/3 |
| rf | 93.67% ± 0.95 | 3/3 |
| cnn_shallow | 93.57% ± 1.28 | 3/3 |
| mlp | 92.09% ± 0.32 | 3/3 |
| knn | 90.30% ± 0.37 | 3/3 |
| yamnet_probe | 85.65% ± 1.83 | 2/3 |
| cnn1d | 64.77% ± 20.7 | 1/3 |

Bootstrap 95% CIs for the strong mel-CNN and handcrafted models sit above 84.51%. `yamnet_probe` is near the reference line; `cnn1d` is unstable and not competitive. Main confusions are rice weevil ↔ lesser grain borer. Audits, findings (McNemar, SNR proxy), plots, and release assets live under `experiments/` and [GitHub Releases](https://github.com/arnavd371/Project-Kaan/releases).

CNN ablation (seed 42): full strong recipe keeps `cnn_deep` near ~95%; a bare Adam / sparse-CE baseline collapsed (`cnn_deep` ~7.6%). Leave-one-out removals of SpecAugment / class weights / label smoothing are single-seed and should not be over-interpreted.

## Technical stack

| Layer | Choice |
|---|---|
| Audio | librosa; 16 kHz mono; 10 s windows |
| Features | Mel spectrogram: 128 bands, `n_fft=2048`, `hop_length=512` → 128×128×1 |
| Training | TensorFlow / Keras (`model/train.py`, `model/train_kaggle.py`) |
| Deployed model | INT8 TFLite (`model/project-kaan.tflite`); ONNX for web |
| Web app | Next.js 14 in `web/` + ONNX Runtime Web |
| Mobile shells | Capacitor (`web/android`, `web/ios`) |
| Languages | English, Hindi, Marathi, Punjabi, Telugu |
| Hosting | Vercel (static web); classification is on-device / in-browser |

## Repository layout

| Path | Role |
|---|---|
| `model/`, `utils/` | Preprocess, train, TFLite convert, inference helpers |
| `experiments/` | Same-split benchmarks, audits, multi-seed stats, CNN ablations |
| `web/` | Public Next.js site (canonical UI) |
| `web/android`, `web/ios` | Capacitor wrappers |

Copyright 2026 Arnav Dhiman.


## Run locally

Website:

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000. For Vercel, set the project Root Directory to `web`.

Python helpers / training:

```bash
pip install -r requirements.txt
```

Benchmarks (install TensorFlow if you want CNN approaches):

```bash
python -m experiments.run_benchmark --smoke
python -m experiments.run_benchmark --seeds 42,43,44 --out experiments/outputs/multi
```

Full IRRI GPU runs on Kaggle: see `experiments/KAGGLE.md`.

```bash
bash experiments/kaggle/push_and_run.sh
```

Retrain / export:

```bash
python model/train.py
# or model/train_kaggle.py on Kaggle
python model/convert_tflite.py
```

## Privacy and limits

- Classification audio stays on device / in browser for inference.
- Screening aid only, not a lab diagnosis; low-confidence outputs ask for a re-record.
- Pulse beetle and other legume pests are out of scope.
- Workshop numbers use IRRI + ambient clean windows, not a large Indian phone-mic field corpus.

## Citations

Balingbing C. et al. (2024). Application of a multi-layer CNN to classify major insect pests in stored rice detected by an acoustic device. *Computers and Electronics in Agriculture*, 225, 109297.

IGMRI (2015). Annual Report. Indian Grain Storage Management and Research Institute, Ministry of Food and Public Distribution, Government of India.

## Copyright

Copyright 2026 Arnav Dhiman. Licensed under the Apache License, Version 2.0.
