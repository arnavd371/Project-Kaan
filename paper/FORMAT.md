# Kaan paper format — 35–40 pages

Target length: **35–40 pages** (single-column, 11 pt, ~1" margins), suitable as an **extended technical report / journal-style manuscript**.  
Shorter workshop versions (4–9 pp.) can cut Sections 8–11 and the appendices.

Content source of truth: `experiments/PAPER_METHODOLOGY.md` and `experiments/results/`.

LaTeX skeleton: `paper/main.tex` + `paper/sections/*.tex`.

---

## Page budget (aim 37.5 ± 2.5)

| # | Section | Pages | What to put here |
|---:|---|---:|---|
| — | Title, authors, abstract, keywords | 0.5–1 | Abstract from methodology pack; 5–8 keywords |
| 1 | Introduction | 2.5–3 | Food security; cryptic pests; phone gap; “listen?” objection; contributions |
| 2 | Background and related work | 3.5–4 | SPI acoustics; Balingbing 2024; classical audio ML; edge ML; agri-AI; gap statement |
| 3 | Problem setting and threat model | 2–2.5 | Classes; in/out of scope; farmer workflow; soft vs hard claims |
| 4 | System design (Kaan) | 3–3.5 | Pipeline; preprocess; TFLite/ONNX; UI/languages; privacy; confidence gate |
| 5 | Data | 2.5–3 | IRRI; Speech Commands clean; windowing; byte-dedupe; audits; class counts |
| 6 | Methods | 4.5–5 | Mel + 74-D + YAMNet; 11 approaches; CNN recipe; classical hyperparams; export |
| 7 | Experimental protocol | 2.5–3 | RQ; seeds; metrics; stats; ablations; findings modules; compute |
| 8 | Results | 5–6 | Multi-seed tables; CIs; per-seed; McNemar; confusions; efficiency; SNR proxy; prod CNN |
| 9 | Ablations and analysis | 2.5–3 | Recipe ablations; baseline collapse; weevil↔borer; negative controls (cnn1d, YAMNet) |
| 10 | Discussion | 2–2.5 | Classical ≈ CNN; recipe > depth; soft reference; deploy implications |
| 11 | Limitations, ethics, broader impact | 2–2.5 | Phone-mic gap; taxonomy; overtrust; accessibility |
| 12 | Reproducibility | 1–1.5 | Commands; Kaggle; hashes; licence; artifact table |
| 13 | Conclusion | 0.75–1 | 1 paragraph restating RQ + finding + next step (phone field set) |
| A | Appendices | 5–7 | Hyperparams; extra confusions; audit checklists; spectrogram examples; UI screenshots |
| — | References | 1.5–2 | 35–60 cites typical for this length |
| | **Total** | **~35–40** | |

---

## Figure / table quota (keeps length honest)

| ID | Item | Section | Est. space |
|---|---|---|---|
| Fig 1 | System pipeline | 4 | 0.6 p |
| Fig 2 | Mel examples (4 classes) | 5–6 | 0.8 p |
| Fig 3 | Accuracy / F1 bar vs 84.51% | 8 | 0.7 p |
| Fig 4 | Per-class F1 grouped | 8 | 0.6 p |
| Fig 5 | Confusion: gbdt + cnn_deep | 8–9 | 0.8 p |
| Fig 6 | Size/time vs F1 | 8 | 0.5 p |
| Fig 7 | Ablation bars | 9 | 0.5 p |
| Fig 8 | SNR proxy curve | 8–9 | 0.4 p |
| Fig 9 | App screenshots (optional) | A | 0.5 p |
| Tab 1 | Class taxonomy | 3 | 0.3 p |
| Tab 2 | Feature summary | 6 | 0.4 p |
| Tab 3 | Approach catalogue | 6 | 0.5 p |
| Tab 4 | Multi-seed stats (primary) | 8 | 0.8 p |
| Tab 5 | Per-seed accuracies | 8 | 0.4 p |
| Tab 6 | McNemar highlights | 8–9 | 0.3 p |
| Tab 7 | Efficiency (seed 42) | 8 | 0.3 p |
| Tab 8 | Ablation summary | 9 | 0.4 p |
| Tab 9 | Limitations / non-claims | 11 | 0.3 p |
| Tab 10 | Artifact checklist | 12 | 0.3 p |

Rough figure/table load: **~8–10 pages** of visual matter inside the 35–40.

---

## Word-count guide (single column 11 pt)

| Pages | ≈ words (prose only) |
|---:|---:|
| 1 | 450–550 |
| 35 | ~16k–19k |
| 40 | ~18k–22k |

With figures/tables, aim **~12k–16k words of prose** plus visuals to land at 35–40 pages.

---

## Writing rules for this manuscript

1. Separate **production CNN (97.76%)** from **bake-off means** every time both appear.  
2. Soft claim only vs **cited** 84.51%.  
3. Never claim phone mic > human absolute hearing sensitivity.  
4. Label SNR analysis as **proxy**.  
5. Single-seed ablations: do not over-interpret.  
6. Author: Arnav Dhiman only (no tool co-authors).

---

## Cut plan if a venue wants shorter

| Target | Keep | Cut / move to appendix |
|---:|---|---|
| 8–9 pp workshop | 1, 3–4 short, 6–8 core, 11 short, 13 | Related work short; drop 9 detail; appendices out |
| 12–14 pp | + fuller related work + ablations | Long efficiency/SNR; UI screenshots |
| 35–40 pp (this format) | Full table above | — |

---

## Compile

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or: `latexmk -pdf main.tex`
