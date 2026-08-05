# Agent context: Project Kaan (कान)

Use this briefing when helping with code, papers, apps, releases, or claims about Kaan. Prefer repository files and committed results over memory.

## One-line summary

Kaan is an **open-source, offline acoustic screening tool** for stored-grain pests. A farmer presses a smartphone against a rice/wheat bag, records ~10 s of contact audio, and gets a local 4-class prediction plus a multilingual advisory. No dedicated MEMS probe or cloud upload is required for inference.

## Author and licensing

- **Author:** Arnav Dhiman (public docs: name only unless the user asks otherwise)
- **License:** Apache-2.0 (code + shipped weights). See `LICENSE`, `NOTICE`
- **SPDX:** `Apache-2.0`
- **Version:** 3.1.1 (see `VERSION`, `CHANGELOG.md`)

## Canonical links

| Resource | URL |
|---|---|
| GitHub repo | https://github.com/arnavd371/Project-Kaan |
| Live web demo | https://kaan-web.vercel.app |
| Latest release (v3.1.1) | https://github.com/arnavd371/Project-Kaan/releases/tag/v3.1.1 |
| Android APK (release sideload) | https://github.com/arnavd371/Project-Kaan/releases/download/v3.1.1/kaan-3.1.1.apk |
| Android APK (debug) | https://github.com/arnavd371/Project-Kaan/releases/download/v3.1.1/kaan-3.1.1-debug.apk |
| Android ZIP | https://github.com/arnavd371/Project-Kaan/releases/download/v3.1.1/kaan-3.1.1-android.zip |
| IRRI / Balingbing dataset context | https://github.com/cbalingbing/Rice-Acoustic-Sensor |
| Kaggle (author) | https://www.kaggle.com/arnavd371 |

Local clone (this machine, if present): `/Users/arnavdhiman/Downloads/Project Kaan`

Paper drafts (local):
- Overleaf-ready: `paper/kaan_overleaf.tex` (also `~/Downloads/kaan_overleaf.tex` / `.zip`)
- Longer draft: `paper/kaan_full_paper.tex`

## Problem and product

- **Problem:** Post-harvest insects feed inside kernels; early damage is hard to see. Lab acoustic probes help but need hardware many Indian smallholders cannot afford.
- **Product path:** Phone on bag → 16 kHz mono ~10 s → mel 128×128×1 → INT8 CNN (TFLite) or ONNX (web/Android WebView) → 4-class softmax → confidence gate **0.6** → class + advisory.
- **Languages:** English, Hindi, Marathi, Punjabi, Telugu.
- **Android:** Capacitor wrap of the same web UI (`web/android/`), package `com.arnavdhiman.kaan`. Not a separate SwiftUI app. iOS shell exists under `web/ios/` (needs full Xcode).
- **Privacy:** Classification audio stays on-device / in-browser for the shipped path.

## Classes (4-way)

| ID | Label | Species / meaning |
|---|---|---|
| 0 | `clean` | Ambient / no monitored pest (Speech Commands ambient windows, not silence) |
| 1 | `rice_weevil` | *Sitophilus oryzae* |
| 2 | `lesser_grain_borer` | *Rhyzopertha dominica* |
| 3 | `red_flour_beetle` | *Tribolium castaneum* |

Out of scope: pulse beetle / legumes; legal certification; lab diagnosis.

## Data and evaluation hygiene (critical)

- **Pest audio:** IRRI Rice Acoustic Sensor (Balingbing et al., 2024). Not Indian phone-on-bag field data.
- **Clean audio:** Google Speech Commands `_background_noise_` ambient windows.
- **Protocol:** byte-dedupe identical files → stratified **file-level** 80/20 split (not window-level).
- **Seeds:** `{42, 43, 44}` for multi-seed tables.
- **Soft reference only:** cited Balingbing accuracy **84.51%** under *our* protocol. Not a locked reimplementation of their pipeline.
- **Limitations doc:** `LIMITATIONS.md` (read before making climate / field claims).

## Features

- **Mel:** 128 mels, `n_fft=2048`, `hop_length=512`, dB, min-max, resize to 128×128×1.
- **Handcrafted ~74-D:** MFCC-20 + spectral + chroma summaries (classical models).
- **Waveform:** YAMNet probe path.

## Eleven-approach bake-off (v2)

Same split for all. Committed tables: `experiments/results/`, `stats.md`, `aggregate_metrics.json`.

| Approach | Acc mean ± std (seeds 42/43/44) | Notes |
|---|---:|---|
| `gbdt` | 95.36% ± 1.50 | Best multi-seed mean; fast teacher |
| `cnn_deep` | 95.15% ± 0.48 | Deep mel-CNN v5; distill backbone |
| `extratrees` | 94.94% ± 1.10 | Distill teacher |
| `svm_rbf` | 94.73% ± 1.20 | |
| `logreg` | 94.09% ± 1.59 | |
| `rf` | 93.67% ± 0.95 | |
| `cnn_shallow` | 93.57% ± 1.28 | ~111k params |
| `mlp` | 92.09% ± 0.32 | |
| `knn` | 90.30% ± 0.37 | |
| `yamnet_probe` | 85.65% ± 1.83 | Near soft ref; CI not above ref |
| `cnn1d` | 64.77% ± 20.7 | Unstable |

**Seed 42:** `gbdt` 95.89% vs `cnn_deep` 95.57%; McNemar n.s. Main confusion: weevil ↔ lesser grain borer.

Do **not** say “every model beat 84.51%.” `yamnet_probe` and `cnn1d` do not fully clear that line.

## Distillation / production (v3)

- **Teachers:** `gbdt` + `extratrees` + `cnn_deep`
- **Student:** `cnn_deep` architecture
- **Mix:** \(y_{mix} = \alpha y_{hard} + (1-\alpha) normalize(p^{1/T})\), defaults \(\alpha=0.5\), \(T=2\)
- **Seed 42:** ensemble 95.89%; hard-only deep 95.57%; **distilled student 97.15%** (macro-F1 0.977)
- **Ship:** INT8 TFLite ~333 KB (`model/project-kaan.tflite`), H5 ~3.7 MB, ONNX (`web/public/model/project-kaan.onnx`)
- **Code:** `model/distill.py`, `model/export_deploy.py`
- **Report:** `experiments/results/distill/`

**Do not conflate** distill seed-42 **97.15%** with bake-off multi-seed means.

## Advanced suite (v3.1)

Orchestrators: `experiments/run_advanced.py`, `run_advanced_multiseed.py`. Results: `experiments/results/advanced_multiseed/`.

| Metric | Mean ± std | 95% CI (approx) |
|---|---:|---:|
| Baseline acc | 96.73% ± 0.37% | [96.52%, 97.15%] |
| SSL fine-tune | 96.62% ± 0.18% | |
| Cost-sensitive | 96.73% ± 0.66% | |
| Hierarchical | 94.09% ± 4.76% | high variance |
| ECE after temperature | 0.029 ± 0.015 | |
| Robustness clean | 96.73% ± 0.37% | |
| Phone band 300–3400 Hz | 60.86% ± 8.35% | **primary deployment finding** |
| SNR≤10 / hard combos | ~7.59% | collapse |

**Hier fine-tune (seed 42, strict gate):** baseline 96.84% → fused 97.15% (`experiments/results/hier_finetune/`).

## Repo layout (high level)

```
Project-Kaan/
├── README.md, LIMITATIONS.md, LICENSE, NOTICE, VERSION, CHANGELOG.md
├── model/           # train, preprocess, distill, export, weights
├── experiments/     # bake-off, advanced suite, audits, results/, kaggle/
├── utils/           # ProjectKaanPredictor (TFLite)
├── data/            # how to get WAVs (audio often not committed)
├── web/             # Next.js + Capacitor android/ios, ONNX
└── paper/           # LaTeX drafts (kaan_overleaf.tex, etc.)
```

## How to load / run (pointers)

- Python TFLite: `utils/inference.py` → `ProjectKaanPredictor`
- Web: `cd web && npm install && npm run dev`
- Mobile sync: `cd web && npm run build:mobile` then `npm run android`
- Android docs: `web/MOBILE.md`
- Full IRRI training: prefer Kaggle GPU (scripts under `experiments/kaggle/`)

## Hard rules for agents

1. **Soft claim** vs Balingbing **84.51%** only; never imply locked reimplementation.
2. **Do not conflate** distill 97.15% (seed 42) with bake-off multi-seed means or field accuracy.
3. **IRRI lab ≠ Indian phone-on-bag.** Robustness phone-band / noise collapse is a first-class finding.
4. **Screening aid**, not diagnosis or legal certification.
5. **Public identity:** name only (Arnav Dhiman) unless user asks to add email/portfolio/username in docs.
6. **No em dashes** in user-facing prose if continuing project style (use commas / hyphens).
7. **Commits:** only when asked; author as Arnav Dhiman; do not invent Cursor co-authors.
8. **Secrets:** never commit tokens, keystores with real passwords, or paste live API tokens into chat.
9. Prefer editing existing design (cream `#f7f6f2`, ink UI) rather than inventing a new mobile skin.
10. Cite committed JSON/MD under `experiments/results/` when quoting numbers.

## Suggested citation (software)

Dhiman, A. (2026). *Kaan: Acoustic grain pest detector for Indian farmers* (Version 3.1.1) [Computer software]. https://github.com/arnavd371/Project-Kaan

Primary research citation for data/context:

Balingbing C. et al. (2024). Application of a multi-layer CNN to classify major insect pests in stored rice detected by an acoustic device. *Computers and Electronics in Agriculture*, 225, 109297.

## What “done” looks like for common tasks

| User ask | Likely deliverable |
|---|---|
| Paper / Overleaf | Use `paper/kaan_overleaf.tex` structure; keep soft 84.51%; fill from `experiments/results/` |
| Android | Capacitor path already exists; rebuild with `npm run android:apk` / release assets |
| Claims / README | Sync with bake-off + distill + advanced tables; link LIMITATIONS |
| Retrain | Leakage-aware split first; prefer Kaggle GPU for full IRRI |

## Quick copy-paste for a new agent chat

```
You are helping with Project Kaan (https://github.com/arnavd371/Project-Kaan),
an Apache-2.0 offline acoustic stored-grain pest screener for Indian farmers
(demo: https://kaan-web.vercel.app). Author: Arnav Dhiman. Version 3.1.1.

4 classes: clean, rice_weevil, lesser_grain_borer, red_flour_beetle.
Data: IRRI pest WAVs + Speech Commands ambient clean; byte-dedupe; file-level splits;
seeds 42/43/44. Soft ref only: Balingbing 84.51% under our protocol.

Bake-off: gbdt ~95.36% mean leads; cnn_deep ~95.15%; classical ≈ deep.
Distill (seed 42): student 97.15% → INT8 TFLite ~333KB + ONNX. Do not conflate with multi-seed means.
Advanced: baseline ~96.73%; phone-band ~60.9%; hard noise collapse ~7.6% (primary finding).
App: same web UI via Capacitor Android. Screening aid only; IRRI ≠ Indian phone field corpus.
Read LIMITATIONS.md and experiments/results/ before stating numbers.
```
