# Advanced suite report

- Seed: `42` smoke=False
- Baseline cnn_deep: **97.15%** (macro F1 0.9772)
- Cost-sensitive: **96.84%** (weevil↔borer swaps=7)
- Hierarchical: **78.16%** (pair subset=0.67)
- SSL fine-tune: **97.78%**
- Baseline ECE before→after T: 0.0442→0.0301 (T=0.500)

## Robustness ladder (baseline)

| Rung | Acc | Macro-F1 |
|---|---:|---:|
| `clean` | 97.15% | 0.9772 |
| `snr20` | 30.38% | 0.2728 |
| `snr10` | 7.59% | 0.0353 |
| `snr5` | 7.59% | 0.0353 |
| `snr0` | 7.59% | 0.0353 |
| `phone_band` | 52.22% | 0.4759 |
| `muffle` | 93.35% | 0.8713 |
| `compress` | 92.09% | 0.9005 |
| `clip` | 87.66% | 0.8355 |
| `reverb` | 93.35% | 0.9464 |
| `gain_-12` | 97.15% | 0.9772 |
| `combo_phone` | 7.59% | 0.0353 |
| `combo_hard` | 7.59% | 0.0353 |

Elapsed: 1080.6s
