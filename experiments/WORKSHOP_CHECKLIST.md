# NeurIPS-type workshop: A-tier checklist for Project Kaan

Mark items as you complete them. Do not claim venue acceptance.

## Framing (paper story)

- [ ] Title centres the **RQ** (deeper mel-CNN for pest acoustics), not only the app
- [ ] Clear problem → method → controlled comparison → deployment note
- [ ] Contribution list: (1) open tool+weights (2) deeper mel-CNN (3) multi-baseline bake-off (4) leakage-aware eval
- [ ] Related work: Balingbing 2024, classical bioacoustics, edge ML, agri-AI (not only product blogs)
- [ ] Explicit non-claims: not a field RCT; not pulse beetle; not medical diagnosis

## Experimental rigor

- [ ] Frozen `split_manifest.json` committed or released with the paper
- [ ] File-level split; zero train/test file overlap; duplicate spectrogram audit
- [ ] ≥3 random seeds; report mean ± std for accuracy and macro F1
- [ ] Baselines: SVM, MLP, GBDT (and optional logreg) on **same** handcrafted features
- [ ] CNN trained/evaluated on **same** split as baselines
- [ ] Metrics: accuracy, macro F1, per-class F1, confusion matrices
- [ ] Bootstrap 95% CI on macro F1 (or McNemar vs best baseline)
- [ ] Ablation table: depth / SpecAugment / class weights (at least one)
- [ ] INT8 or ONNX parity note (float vs quantized gap)

## Figures (camera-ready)

- [ ] Accuracy + macro F1 bar chart across models
- [ ] Per-class F1 grouped bars
- [ ] Confusion matrix grid
- [ ] CNN learning curves
- [ ] Accuracy-efficiency (size or latency vs F1)
- [ ] Example mel spectrograms (1 per class)

## Systems / open science

- [ ] GitHub public; licence matches paper text (Apache-2.0)
- [ ] Tag **v1.0** release with TFLite/ONNX + train scripts
- [ ] One-command reproduce for tables (`python -m experiments.run_benchmark`)
- [ ] Live demo URL listed as systems artifact, secondary to results

## Ethics / impact (workshop-friendly)

- [ ] Limitations section (domain shift, class taxonomy, false negatives)
- [ ] Broader impact: food security, pesticide reduction, misuse/overtrust
- [ ] Data licence/attribution for IRRI clips

## Venue fit (be realistic)

| Venue tier | Fit today |
|---|---|
| NeurIPS / ICML / ICLR main | Low without field data + stronger novelty |
| NeurIPS workshops (ML4D, Climate, Audio, Deployed) | Plausible if checklist above is green |
| Agri / Sensors / Applied AI venues | Strong with deployment story |

## Honest gap list (remaining)

1. Short related-work section with citations in the paper draft
2. Optional: 50-100 Indian phone recordings as external test set
3. Field / KVK pilot notes for deployment claims
