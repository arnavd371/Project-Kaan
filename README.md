# Kaan | कान

**Kaan** (कान, Hindi/Punjabi for “ear”) is an open-source acoustic screening tool for stored-grain pests. A farmer holds a smartphone against a rice or wheat bag, records about ten seconds of contact audio, and gets a local class prediction plus a short advisory in English, Hindi, Marathi, Punjabi, or Telugu. Inference runs on-device (TFLite) or in the browser (ONNX). Audio does not need a cloud upload for classification.

It is built for Indian smallholders and extension pilots: no special probe hardware, Apache-2.0 code and weights, and a leakage-aware experiment suite so others can retrain or audit the claims.

| | |
|---|---|
| **Author** | Arnav Dhiman |
| **Email** | [arnavd371@gmail.com](mailto:arnavd371@gmail.com) |
| **GitHub** | [arnavd371](https://github.com/arnavd371) |
| **Portfolio** | [arnavdportfolio.vercel.app](https://arnavdportfolio.vercel.app/) |
| **Live demo** | [kaan-web.vercel.app](https://kaan-web.vercel.app) |
| **Repository** | [arnavd371/Project-Kaan](https://github.com/arnavd371/Project-Kaan) |
| **Latest release** | [v3.1.1](https://github.com/arnavd371/Project-Kaan/releases/tag/v3.1.1) |
| **Licence** | [Apache License 2.0](LICENSE) (see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), [`CHANGELOG.md`](CHANGELOG.md)) |
| **SPDX** | `Apache-2.0` |

---

## Contents

1. [Introduce Kaan](#introduce-kaan)
2. [Open-source specifications](#open-source-specifications)
3. [Load the trained models](#load-the-trained-models)
4. [How it works](#how-it-works)
5. [Classes](#classes)
6. [Results](#results)
7. [Workshop contribution framing](#workshop-contribution-framing)
8. [Limitations](#limitations)
9. [Repository layout](#repository-layout)
10. [Technical stack](#technical-stack)
11. [Data](#data)
12. [Experiments (v2 bake-off)](#experiments-v2-bake-off)
13. [Advanced suite (v3+)](#advanced-suite-v3)
14. [Run locally](#run-locally)
15. [Train and export](#train-and-export)
16. [Privacy, safety, and limits](#privacy-safety-and-limits)
17. [Author and copyright](#author-and-copyright)
18. [Cite](#cite)
19. [Acknowledgements](#acknowledgements)

---

## Introduce Kaan

**Problem.** Post-harvest insects (rice weevil, lesser grain borer, red flour beetle) feed inside kernels. Early damage is hard to see. Lab acoustic sensors and commercial probes are expensive for many farms.

**What Kaan does.** Contact phone audio → mel spectrogram → compact CNN → one of four labels + multilingual advisory. Low confidence triggers a re-record prompt instead of a forced pest call.

**What it is not.** Not a laboratory or legal diagnosis. Not trained on a large Indian phone-on-bag field corpus yet (IRRI contact acoustics + ambient clean windows). See [`LIMITATIONS.md`](LIMITATIONS.md).

**Why open source.** Agriculture departments, KVKs, and researchers can inspect the pipeline, load the shipped weights, retrain on local grain, and redistribute under Apache-2.0.

---

## Open-source specifications

| Spec | Detail |
|---|---|
| Licence | Apache License 2.0 (root + `web/`) |
| Copyright | © 2026 Arnav Dhiman (`arnavd371@gmail.com`) |
| SPDX | `Apache-2.0` |
| Version | see [`VERSION`](VERSION) / [`CHANGELOG.md`](CHANGELOG.md) |
| Source | https://github.com/arnavd371/Project-Kaan |
| Demo | https://kaan-web.vercel.app |
| Shipped weights | Keras H5, INT8 TFLite, ONNX (Apache-2.0 grant; see `NOTICE`) |
| Training audio | Not redistributed in-repo; IRRI + Speech Commands terms stay with publishers |
| Input | 16 kHz mono, ~10 s window (pad/crop) |
| Features | Mel 128×128×1 (`n_fft=2048`, `hop_length=512`, 128 mels, dB, min-max) |
| Output | Softmax over 4 classes; UI confidence gate 0.6 |
| Production model | Distilled deep mel-CNN (~333 KB INT8 TFLite; ~3.7 MB H5) |
| Native path | TFLite via LiteRT / TF Lite Interpreter (`utils/inference.py`) |
| Web path | ONNX Runtime Web (`web/public/model/project-kaan.onnx`) |
| Mobile shell | Capacitor (`web/android`, `web/ios`) |
| Languages | English, Hindi, Marathi, Punjabi, Telugu |

Anyone may use, modify, and redistribute the software under the Apache-2.0 conditions in [`LICENSE`](LICENSE). Keep the attribution notices in [`NOTICE`](NOTICE).

---

## Load the trained models

Shipped artifacts (also attached to GitHub releases):

| File | Role |
|---|---|
| `model/project-kaan_model.h5` | Keras float model (train / export / research) |
| `model/project-kaan.tflite` | INT8 production weights (native / Python) |
| `web/public/model/project-kaan.onnx` | Browser / ONNX Runtime |
| `web/public/model/mel_filterbank.json` | Mel filterbank used by the web preprocessor |

### Python (TFLite)

```bash
pip install -r requirements.txt
# optional: tensorflow>=2.13 if you prefer tf.lite over LiteRT
```

```python
from pathlib import Path
from utils.inference import ProjectKaanPredictor

predictor = ProjectKaanPredictor(
    model_path=Path("model/project-kaan.tflite")
)
result = predictor.predict("path/to/contact.wav")
print(result["class"], result["confidence"], result["confident"], result["all_scores"])
```

`ProjectKaanPredictor` loads INT8 TFLite, builds the mel the same way as training (`model/preprocess.py`), dequantizes softmax, and applies the 0.6 confidence gate (`confident` is true when confidence > 0.6). If the `.tflite` file is missing it falls back to a demo heuristic (not for production).

### Python (Keras H5)

```python
import numpy as np
from tensorflow import keras
from model.preprocess import preprocess_audio

model = keras.models.load_model("model/project-kaan_model.h5")
mel = preprocess_audio("path/to/contact.wav")  # float32 (128, 128, 1)
probs = model.predict(mel[None, ...], verbose=0)[0]
classes = ["clean", "rice_weevil", "lesser_grain_borer", "red_flour_beetle"]
print(classes[int(probs.argmax())], float(probs.max()))
```

### Browser / Node (ONNX)

The live app loads `/model/project-kaan.onnx` via ONNX Runtime Web (`web/src/lib/model.ts`). Mel prep must match training (`web/src/lib/mel.ts` + `mel_filterbank.json`). INT8 input/output scales in `model.ts` must match the TFLite quantization (re-synced by `python -m model.export_deploy`).

```bash
cd web && npm install && npm run dev
# open http://localhost:3000 → App records or uploads audio and runs ONNX locally
```

Standalone ONNX (Node example):

```bash
npm install onnxruntime-node
```

```js
import * as ort from "onnxruntime-node";
const session = await ort.InferenceSession.create("web/public/model/project-kaan.onnx");
// feed UINT8 NHWC mel [1,128,128,1] with the same scales as web/src/lib/model.ts
```

### Download from a release

```bash
gh release download v3.1.1 -R arnavd371/Project-Kaan -p '*.tflite' -p '*.onnx' -p '*.h5' -D ./weights
# or clone the repo; weights are already under model/ and web/public/model/
```

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

**Mel parameters:** 128 mel bands, `n_fft=2048`, `hop_length=512`, power→dB, min-max normalize, resize to 128×128×1.

**Training-time robustness (production / Kaggle CNN path):** pink noise at controlled SNR, pitch shift, time stretch, SpecAugment (time/frequency masks), class weights, label smoothing, cosine LR (strong recipe).

**Clean class:** real ambient noise windows (Speech Commands `_background_noise_`), not synthetic silence.

The shipped product path is the mel-CNN. The `experiments/` suite compares additional approaches on the same leakage-aware split; it does **not** automatically replace production weights.

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

### Advanced suite multi-seed (v3.1.0)

Seeds **42 / 43 / 44**, bootstrap 95% CI of the mean. Full tables: [`experiments/results/advanced_multiseed/`](experiments/results/advanced_multiseed/).

| Metric | Mean ± std | 95% CI |
|---|---:|---:|
| Baseline accuracy | 96.73% ± 0.37% | [96.52%, 97.15%] |
| SSL fine-tune accuracy | 96.62% ± 0.18% | [96.52%, 96.84%] |
| Cost-sensitive accuracy | 96.73% ± 0.66% | [96.20%, 97.47%] |
| Hierarchical accuracy | 94.09% ± 4.76% | [88.61%, 97.15%] |
| ECE after temperature | 0.029 ± 0.015 | [0.015, 0.046] |
| Robustness: clean | 96.73% ± 0.37% | [96.52%, 97.15%] |
| Robustness: phone band | 60.86% ± 8.35% | [51.27%, 66.46%] |
| Robustness: SNR ≤10 / hard combos | ~7.59% | (collapse) |

**Hierarchical fine-tune (seed 42, strict gate):** baseline **96.84%** → fused hierarchy **97.15%** (scratch cascade 94.30%). Report: [`experiments/results/hier_finetune/`](experiments/results/hier_finetune/).

---

## Workshop contribution framing

For climate / AI-for-good workshops, lead with **deployment constraints**, not peak lab accuracy:

1. **On-device / offline** INT8 + ONNX screening (no cloud required for inference)
2. **Robustness ladder** - phone-like degradations; treat SNR/band-pass collapse as a primary finding
3. **Calibration + abstain** - temperature scaling and coverage-accuracy curves
4. **Open bake-off + distillation** - classical ≈ deep; distilled production weights released

Accuracy vs the cited 84.51% reference is **supporting**. Draft: [`workshop/`](workshop/) (CCAI @ NeurIPS 2026 Papers track, ≤4 pages).

---

## Limitations

See **[`LIMITATIONS.md`](LIMITATIONS.md)** for domain shift, species coverage, soft reference comparison, and ethics. Short version: IRRI lab acoustics ≠ Indian phone-on-bag; the robustness ladder shows how badly phone-band and noise can hurt.

---

## Repository layout

```
Project-Kaan/
├── README.md                 # this file
├── CHANGELOG.md              # release notes
├── VERSION                   # current semver
├── LIMITATIONS.md            # domain shift / ethics for workshop claims
├── LICENSE                   # Apache-2.0 (appendix filled: Arnav Dhiman)
├── NOTICE                    # copyright owner + data + dependency notices
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
└── web/                      # Next.js app + Capacitor android/ios (Apache-2.0)
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

# Kaggle GPU - preferred
bash experiments/kaggle/push_advanced.sh
# https://www.kaggle.com/code/arnavd371/kaan-advanced-suite

# Dashboard from committed JSON/MD
python -m experiments.build_results_page
# → experiments/results/index.html
```

Artifacts: `experiments/results/advanced/` (single-seed reference) and `experiments/results/advanced_multiseed/` (v3.1 means ± std, 95% CIs). Hierarchical fine-tune: `experiments/results/hier_finetune/`. Skips farmer-app UX (multi-clip vote, etc.).

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
# Local (CPU-heavy - prefer Kaggle GPU)
python -m experiments.prepare_kaggle_data --out .
python -m model.distill
python -m model.export_deploy

# Kaggle GPU (T4) - preferred
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

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) (appendix filled with copyright-owner identity - not empty brackets) and [`NOTICE`](NOTICE) (owner contacts, data attributions, dependency notes). The `web/` client uses the same Apache-2.0 grant (`web/LICENSE`, `web/NOTICE`).

---

## Cite

See [`CITATION.cff`](CITATION.cff). Software citation (APA-style):

> Dhiman, A. (2026). *Kaan (कान): Acoustic grain pest detector for Indian farmers* (Version 3.1.1) [Computer software]. https://github.com/arnavd371/Project-Kaan

---

## Acknowledgements

- Balingbing et al. (2024) and the IRRI Rice Acoustic Sensor Dataset.
- Google Speech Commands ambient noise subset (when used for `clean`).
- Open-source stack: TensorFlow, librosa, scikit-learn, Next.js, ONNX Runtime Web, Capacitor.

Primary research citation:

Balingbing C. et al. (2024). Application of a multi-layer CNN to classify major insect pests in stored rice detected by an acoustic device. *Computers and Electronics in Agriculture*, 225, 109297.

IGMRI (2015). Annual Report. Indian Grain Storage Management and Research Institute, Ministry of Food and Public Distribution, Government of India.
