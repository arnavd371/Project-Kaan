# Kaan | कान

**Acoustic stored-grain pest detector for Indian farmers.**

Hold a phone against a storage bag, record about ten seconds of audio, and get an on-device class prediction plus an actionable advisory in English, Hindi, Marathi, Punjabi, or Telugu.

| | |
|---|---|
| **Author** | Arnav Dhiman |
| **Email** | [arnavd371@gmail.com](mailto:arnavd371@gmail.com) |
| **GitHub** | [arnavd371](https://github.com/arnavd371) |
| **Portfolio** | [arnavdportfolio.vercel.app](https://arnavdportfolio.vercel.app/) |
| **Live demo** | [kaan-web.vercel.app](https://kaan-web.vercel.app) |
| **Repository** | [arnavd371/Project-Kaan](https://github.com/arnavd371/Project-Kaan) |
| **Latest release** | [v3.0.0](https://github.com/arnavd371/Project-Kaan/releases/tag/v3.0.0) |
| **Licence** | [Apache License 2.0](LICENSE) (`LICENSE`, `NOTICE`) |

---

## Contents

1. [What it does](#what-it-does)
2. [Why it exists](#why-it-exists)
3. [How it works](#how-it-works)
4. [Classes](#classes)
5. [Results](#results)
6. [Workshop contribution framing](#workshop-contribution-framing)
7. [Limitations](#limitations)
8. [Repository layout](#repository-layout)
9. [Technical stack](#technical-stack)
10. [Data](#data)
11. [Experiments (v2 bake-off)](#experiments-v2-bake-off)
12. [Advanced suite (v3+)](#advanced-suite-v3)
13. [Run locally](#run-locally)
14. [Train and export](#train-and-export)
15. [Privacy, safety, and limits](#privacy-safety-and-limits)
16. [Author and copyright](#author-and-copyright)
17. [Cite](#cite)
18. [Acknowledgements](#acknowledgements)

---

## What it does

Kaan turns a short contact recording into one of four labels and shows a short advisory. Classification runs **on the device or in the browser**. Audio does not need to be uploaded for inference.

Typical flow:

1. Place the phone flat against a grain bag or bin.
2. Record ~10–30 seconds (canonical training window is 10 s at 16 kHz mono).
3. The client builds a mel spectrogram and runs the compact CNN (TFLite or ONNX).
4. If confidence is low, the UI asks for a quieter, closer re-record.

The shipped product path is the mel-CNN. The `experiments/` suite compares additional approaches on the same leakage-aware split; it does **not** automatically replace production weights.

---

## Why it exists

India stores large volumes of food grain. Insects cause substantial storage losses (IGMRI, 2015). Rice weevil, lesser grain borer, and red flour beetle often feed inside kernels, so early damage is hard to see. Lab acoustic systems and commercial probes are expensive for many smallholders.

Kaan targets a **free, offline phone workflow**: contact audio, on-device model, multilingual text, no special hardware beyond a smartphone.

Unaided listening can catch late, loud infestation. Early bag sounds are sparse and hard to classify by ear at scale. Kaan is a **screening aid**, not a claim that a phone microphone outranks human absolute hearing sensitivity.

---

## How it works

```
Phone on bag
    → 16 kHz mono WAV (~10 s)
    → trim silence / pad or crop
    → mel spectrogram 128 × 128 × 1
    → INT8 TFLite (native) or ONNX Runtime Web (browser)
    → 4-class softmax
    → confidence gate (threshold 0.6) → class + advisory
```

**Mel parameters:** 128 mel bands, `n_fft=2048`, `hop_length=512`, power→dB, min–max normalize, resize to 128×128×1.

**Training-time robustness (production / Kaggle CNN path):** pink noise at controlled SNR, pitch shift, time stretch, SpecAugment (time/frequency masks), class weights, label smoothing, cosine LR (strong recipe).

**Clean class:** real ambient noise windows (Speech Commands `_background_noise_`), not synthetic silence.

---

## Classes

| ID | Name | Species / meaning |
|---|---|---|
| 0 | `clean` | No pest detected (ambient / background) |
| 1 | `rice_weevil` | *Sitophilus oryzae* |
| 2 | `lesser_grain_borer` | *Rhyzopertha dominica* |
| 3 | `red_flour_beetle` | *Tribolium castaneum* |

Out of scope: pulse beetle and other legume pests; legal certification; lab diagnosis.

---

## Results

### Production CNN (shipped path)

| Metric | Value |
|---|---|
| Validation accuracy (v3 distilled, seed 42) | **97.15%** |
| Macro F1 | 0.977 |
| Hard-only deep baseline (same split) | 95.57% |
| Teacher ensemble | 95.89% |
| INT8 TFLite | ~333 KB |
| Keras H5 | ~3.7 MB |

v3 ships a deep mel-CNN distilled from `gbdt` + `extratrees` + `cnn_deep` soft labels (Kaggle T4). Report: [`experiments/results/distill/`](experiments/results/distill/). Do not conflate this with the multi-seed bake-off means below.

### Multi-approach bake-off (v2.0.0)

Same **file-level** stratified split after **byte-dedupe**, IRRI pest WAVs + Speech Commands ambient windows for `clean`, seeds **42 / 43 / 44**. Reference line: cited Balingbing et al. accuracy **84.51%** on this protocol (not a locked reimplementation).

| Approach | Acc mean ± std | Macro-F1 mean ± std | Seeds > 84.51% | CI above ref? |
|---|---:|---:|:---:|:---:|
| `gbdt` | 95.36% ± 1.50 | 96.02% ± 1.30 | 3/3 | yes |
| `cnn_deep` | 95.15% ± 0.48 | 95.86% ± 0.59 | 3/3 | yes |
| `extratrees` | 94.94% ± 1.10 | 95.67% ± 1.01 | 3/3 | yes |
| `svm_rbf` | 94.73% ± 1.20 | 95.41% ± 1.34 | 3/3 | yes |
| `logreg` | 94.09% ± 1.59 | 95.27% ± 1.31 | 3/3 | yes |
| `rf` | 93.67% ± 0.95 | 94.64% ± 0.81 | 3/3 | yes |
| `cnn_shallow` | 93.57% ± 1.28 | 94.59% ± 1.16 | 3/3 | yes |
| `mlp` | 92.09% ± 0.32 | 93.65% ± 0.27 | 3/3 | yes |
| `knn` | 90.30% ± 0.37 | 92.06% ± 0.23 | 3/3 | yes |
| `yamnet_probe` | 85.65% ± 1.83 | 88.28% ± 1.50 | 2/3 | no |
| `cnn1d` | 64.77% ± 20.7 | 69.44% ± 19.0 | 1/3 | no |

**Findings (seed 42):** best classical `gbdt` (95.89%) vs best CNN `cnn_deep` (95.57%); McNemar not significant. Main confusions: rice weevil ↔ lesser grain borer. `cnn1d` is unstable; `yamnet_probe` is near the reference line only.

Committed tables: [`experiments/results/`](experiments/results/). Release assets: [v2.0.0](https://github.com/arnavd371/Project-Kaan/releases/tag/v2.0.0).

---

## Workshop contribution framing

For climate / AI-for-good workshops, lead with **deployment constraints**, not peak lab accuracy:

1. **On-device / offline** INT8 + ONNX screening (no cloud required for inference)
2. **Robustness ladder** — phone-like degradations; treat SNR/band-pass collapse as a primary finding
3. **Calibration + abstain** — temperature scaling and coverage–accuracy curves
4. **Open bake-off + distillation** — classical ≈ deep; distilled production weights released

Accuracy vs the cited 84.51% reference is **supporting**. Draft: [`workshop/`](workshop/) (CCAI @ NeurIPS 2026 Papers track, ≤4 pages).

---

## Limitations

See **[`LIMITATIONS.md`](LIMITATIONS.md)** for domain shift, species coverage, soft reference comparison, and ethics. Short version: IRRI lab acoustics ≠ Indian phone-on-bag; the robustness ladder shows how badly phone-band and noise can hurt.

---

## Repository layout

```
Project-Kaan/
├── README.md                 # this file
├── LIMITATIONS.md            # domain shift / ethics for workshop claims
├── LICENSE                   # Apache-2.0 (copyright: Arnav Dhiman)
├── NOTICE                    # copyright owner contact + data notices
├── CITATION.cff
├── requirements.txt
├── workshop/                 # CCAI @ NeurIPS 2026 draft (≤4 pages)
├── model/                    # train, preprocess, TFLite export, weights
├── utils/                    # inference helpers
├── data/                     # how to obtain WAVs (data not always committed)
├── experiments/              # bake-off, audits, stats, findings, Kaggle
│   ├── results/              # committed multi-seed summaries
│   ├── kaggle/               # GPU kernel push scripts
│   └── outputs/              # local run artifacts (gitignored)
└── web/                      # Next.js app + Capacitor android/ios
```

| Path | Role |
|---|---|
| `model/train.py` | Shallow / production-style mel-CNN training |
| `model/train_kaggle.py` | Deeper CNN + strong recipe (Kaggle-oriented) |
| `model/distill.py` | Distill gbdt+extratrees+cnn_deep → production CNN |
| `model/export_deploy.py` | INT8 TFLite + ONNX export and web sync |
| `model/convert_tflite.py` | INT8 TFLite export (legacy entry) |
| `model/project-kaan.tflite` | Shipped INT8 weights |
| `experiments/run_benchmark.py` | Eleven-approach same-split benchmark |
| `experiments/run_ablations.py` | CNN recipe ablations |
| `experiments/findings.py` | McNemar, confusions, SNR proxy, INT8 parity |
| `web/` | Canonical UI (Vercel root directory = `web`) |

---

## Technical stack

| Layer | Choice |
|---|---|
| Audio | librosa; 16 kHz mono; 10 s windows |
| Mel features | 128 bands, `n_fft=2048`, `hop_length=512` → 128×128×1 |
| Handcrafted (~74-D) | MFCC-20 + spectral + chroma summaries |
| Training | TensorFlow / Keras |
| Deployed model | INT8 TFLite; ONNX for web |
| Web | Next.js 14 + ONNX Runtime Web |
| Mobile | Capacitor (`web/android`, `web/ios`) |
| Languages | English, Hindi, Marathi, Punjabi, Telugu |
| Hosting | Vercel (static); inference on-device / in-browser |
| Experiments compute | Kaggle GPU (T4) for full IRRI runs |

---

## Data

Expected local layout after prep:

```
data/clean/*.wav
data/rice_weevil/*.wav
data/lesser_grain_borer/*.wav
data/red_flour_beetle/*.wav
```

| Source | Role |
|---|---|
| [IRRI Rice Acoustic Sensor](https://github.com/cbalingbing/Rice-Acoustic-Sensor) (Balingbing et al., 2024) | Pest class WAVs |
| Speech Commands `_background_noise_` | Ambient windows for `clean` |

**Eval hygiene:** byte-dedupe identical files before split; stratified **file-level** train/val (not window-level); before/after training audits (`experiments/audit.py`). See `data/HOW_TO_GET_DATA.md` and `experiments/KAGGLE.md`.

---

## Experiments (v2 bake-off)

Approaches: `cnn_shallow`, `cnn_deep`, `cnn1d`, `yamnet_probe`, `svm_rbf`, `mlp`, `gbdt`, `rf`, `extratrees`, `knn`, `logreg`.

```bash
pip install -r requirements.txt
pip install 'tensorflow>=2.13.0'    # CNN approaches
# optional: tensorflow_hub          # yamnet_probe

python -m experiments.run_benchmark --smoke
python -m experiments.run_benchmark --seeds 42,43,44 --out experiments/outputs/multi
python -m experiments.run_ablations --out experiments/outputs/ablations
bash experiments/kaggle/push_and_run.sh
```

Details: [`experiments/README.md`](experiments/README.md).

---

## Advanced suite (v3+)

Desk-bound follow-ons (no field mics): phone-like **robustness ladder**, **weevil↔borer** cost-sensitive + hierarchical heads, **temperature calibration / abstain curves**, **SimCLR mel SSL** on IRRI + Speech Commands ambient → fine-tune, and a static **results dashboard**.

```bash
# Local smoke
python -m experiments.run_advanced --smoke

# Multi-seed + bootstrap CIs (prefer Kaggle GPU)
python -m experiments.run_advanced_multiseed --seeds 42,43,44 --copy-results

# Kaggle GPU — preferred
bash experiments/kaggle/push_advanced.sh
# https://www.kaggle.com/code/arnavd371/kaan-advanced-suite

# Dashboard from committed JSON/MD
python -m experiments.build_results_page
# → experiments/results/index.html
```

Artifacts: `experiments/results/advanced/` (single-seed) and `experiments/results/advanced_multiseed/` (means ± std, 95% CIs). Skips farmer-app UX (multi-clip vote, etc.).

---

## Run locally

### Website

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000. On Vercel, set **Root Directory** to `web`.

### Python helpers

```bash
pip install -r requirements.txt
```

---

## Train and export

Baseline hard-label CNN:

```bash
python model/train.py
python model/convert_tflite.py
```

**Distilled production model** (recommended): ensemble soft labels from `gbdt` + `extratrees` + `cnn_deep` into the deep mel-CNN, then INT8 TFLite + ONNX for `web/`.

```bash
# Local (CPU-heavy — prefer Kaggle GPU)
python -m experiments.prepare_kaggle_data --out .
python -m model.distill
python -m model.export_deploy

# Kaggle GPU (T4) — preferred
bash experiments/kaggle/push_distill.sh
# https://www.kaggle.com/code/arnavd371/kaan-distill-production
kaggle kernels status arnavd371/kaan-distill-production
```

Smoke (no WAVs): `python -m model.distill --smoke` (writes `project-kaan_model.smoke.h5` only).

After Kaggle completes, download `project-kaan_model.h5`, `project-kaan.tflite`, `project-kaan.onnx`, and `distill_report.md` from kernel output, then copy into `model/` and `web/public/model/` (or re-run `python -m model.export_deploy` locally from the H5).

Production weights live under `model/` (`project-kaan_model.h5`, `project-kaan.tflite`) and `web/public/model/project-kaan.onnx`.

---

## Privacy, safety, and limits

- Classification audio stays on the device / in the browser for inference.
- Screening aid only; not a lab or legal diagnosis.
- Low-confidence outputs ask for a re-record in a quieter setting.
- Phone mic quality, bag material, and background noise affect accuracy.
- Benchmark numbers use IRRI + ambient clean windows, **not** a large Indian phone-mic field corpus.
- Pulse beetle and other pests outside the four-class table are unsupported.

---

## Author and copyright

**Copyright © 2026 Arnav Dhiman.**

| Field | Value |
|---|---|
| Legal name | Arnav Dhiman |
| Email | arnavd371@gmail.com |
| GitHub | https://github.com/arnavd371 |
| Portfolio | https://arnavdportfolio.vercel.app/ |
| Project | https://github.com/arnavd371/Project-Kaan |

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The Apache appendix boilerplate in `LICENSE` is filled with the same copyright owner details.

---

## Cite

See [`CITATION.cff`](CITATION.cff). Software citation (APA-style):

> Dhiman, A. (2026). *Kaan (कान): Acoustic grain pest detector for Indian farmers* (Version 3.0.0) [Computer software]. https://github.com/arnavd371/Project-Kaan

---

## Acknowledgements

- Balingbing et al. (2024) and the IRRI Rice Acoustic Sensor Dataset.
- Google Speech Commands ambient noise subset (when used for `clean`).
- Open-source stack: TensorFlow, librosa, scikit-learn, Next.js, ONNX Runtime Web, Capacitor.

Primary research citation:

Balingbing C. et al. (2024). Application of a multi-layer CNN to classify major insect pests in stored rice detected by an acoustic device. *Computers and Electronics in Agriculture*, 225, 109297.

IGMRI (2015). Annual Report. Indian Grain Storage Management and Research Institute, Ministry of Food and Public Distribution, Government of India.
