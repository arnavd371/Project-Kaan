# Changelog

All notable releases of Project Kaan are listed here.

## [3.1.0] — 2026-08-04

Workshop-competitive research release on top of the v3 distilled production model.

- Multi-seed advanced suite (seeds 42/43/44) with bootstrap 95% CIs: robustness ladder, calibration, cost-sensitive and hierarchical heads, SSL fine-tune
- Hierarchical weevil↔borer fine-tune with strict top-2 fusion gate (Kaggle): **97.15%** vs **96.84%** baseline
- CCAI @ NeurIPS 2026 Papers-track draft under `workshop/` (anonymous PDF + LaTeX)
- `LIMITATIONS.md` — domain shift, soft 84.51% reference, ethics
- Contribution framing: on-device, robustness, calibration, distillation (accuracy as supporting)
- Apache-2.0 `NOTICE` expanded with full copyright-owner and third-party attributions; `web/` aligned from MIT to Apache-2.0
- Results dashboard includes multi-seed aggregate (`experiments/results/advanced_multiseed/`)

## [3.0.0] — 2026-08-04

- Distilled production CNN (gbdt + extratrees + cnn_deep → deep mel-CNN), 97.15% val (seed 42)
- TFLite / ONNX ship path; Apache-2.0 identity docs

## [2.0.0] — 2026

- Leakage-aware multi-approach bake-off, multi-seed stats, ablations

Earlier tags (`v1.x`) document experiment and ablation milestones.
