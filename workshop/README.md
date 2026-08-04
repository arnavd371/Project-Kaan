# Workshop submission (CCAI @ NeurIPS 2026)

Target: **Tackling Climate Change with Machine Learning** workshop. Papers track (at most 4 pages + refs), agriculture/food, ground-up / on-device theme.

| File | Role |
|---|---|
| `kaan_ccai_neurips2026.tex` | Anonymous draft (edit this; do not commit personal writeups as `.txt`) |
| `kaan_ccai_neurips2026.pdf` | Built PDF |
| `neurips_2025.sty` | NeurIPS-based style (swap for official CCAI zip when OpenReview opens) |

## Build PDF

```bash
cd workshop
pdflatex kaan_ccai_neurips2026.tex
pdflatex kaan_ccai_neurips2026.tex
```

## Blind submission checklist

- [ ] No author names / affiliations / personal contact in PDF
- [ ] Use official CCAI style zip when OpenReview opens
- [ ] Climate impact pathway explicit (adaptation + waste/mitigation)
- [ ] Limitations / domain shift section present
- [ ] Code/data availability statement (anonymized for review)
- [ ] Do not paste CCAI CFP text into LLMs (workshop rule)

## What this draft covers (successful results only)

Bake-off, distillation, multi-seed advanced suite (robustness, calibration, SSL, heads), hierarchical fine-tune that beats baseline, Apache-2.0 release. Soft 84.51% reference; phone-band/noise collapse as primary deployment finding.

See also repo root `LIMITATIONS.md`.
