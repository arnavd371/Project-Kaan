# Limitations and domain shift

Kaan is a **screening aid**, not a laboratory or legal diagnosis. The points below are intentional product and research constraints - especially for workshop-facing claims.

## Data domain

- Pest audio comes from the **IRRI Rice Acoustic Sensor** corpus (lab/contact acoustic device), not from Indian smartphone-on-bag recordings.
- The `clean` class uses **Speech Commands ambient windows**, not silence and not farm storage ambience in India.
- Species covered: rice weevil, lesser grain borer, red flour beetle only. Pulse beetle and other pests are unsupported.

## Evaluation domain

- Reported accuracies use **leakage-aware file-level splits** after byte-dedupe on the above sources.
- Comparison to Balingbing et al. **84.51%** is a **soft reference** under our protocol, not a locked reimplementation of their exact pipeline.
- Bake-off numbers are multi-seed (42/43/44). Advanced-suite citation preferred: [`experiments/results/advanced_multiseed/`](experiments/results/advanced_multiseed/) (means ± std, bootstrap 95% CIs).

## Deployment domain shift (first-class finding)

The **robustness ladder** replays validation audio under phone-like degradations (SNR, telephone band-pass, muffling, compression, clipping, reverb, combos).

Observed pattern (seed 42 baseline ~97% clean):

- Mild muffling / reverb / compress / gain: often still high 80s-90s.
- Phone band-pass and additive noise: **sharp collapse** (often &lt;50%, sometimes ~8% on hard combos).

So laboratory accuracy **does not** imply phone-mic field accuracy. Any climate-adaptation claim must keep this gap visible.

## Product / ethics

- Offline inference keeps classification audio on-device for the shipped path; that does not remove consent or misuse risks of any future logging features.
- Confidence gating / abstention should be preferred over forcing a pest label when unsure.
- Advisories are informational; extension services and local regulations prevail.

## What would close the gap

1. Labeled **phone-on-bag** recordings under Indian storage conditions (same four classes).
2. Human factors study with farmers / FPOs on false-alarm cost.
3. Tighter phone-mic augmentation matched to measured handset frequency responses.
