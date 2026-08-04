# Workshop submission (CCAI @ NeurIPS 2026)

Target: **Tackling Climate Change with Machine Learning** workshop — Papers track (≤4 pages + refs), agriculture/food, ground-up / on-device theme.

| File | Role |
|---|---|
| `kaan_ccai_neurips2026.tex` | Anonymous 4-page draft |
| `neurips_2025.sty` | NeurIPS-based style (CCAI requires workshop template when released; this is the base) |

## Build PDF

```bash
cd workshop
pdflatex kaan_ccai_neurips2026.tex
pdflatex kaan_ccai_neurips2026.tex
```

## Blind submission checklist

- [ ] No author names / affiliations / GitHub / portfolio in PDF
- [ ] Use official CCAI style zip when OpenReview opens (~Aug 2026)
- [ ] Climate impact pathway explicit (adaptation + waste/mitigation)
- [ ] Limitations / domain shift section present
- [ ] Code/data availability statement (anonymized URL or “supplement”)
- [ ] Do not paste CCAI CFP text into LLMs (workshop rule)

## Contribution framing (lead with these)

1. On-device / offline screening under rural constraints  
2. Robustness ladder as phone-domain proxy  
3. Calibration + abstain  
4. Open bake-off + distillation  

Accuracy vs 84.51% is supporting, not the headline.

See also repo root `LIMITATIONS.md`.
