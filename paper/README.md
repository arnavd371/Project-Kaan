# Paper directory (35–40 page format)

| File | Role |
|---|---|
| [`FORMAT.md`](FORMAT.md) | Page budget, figure/table quota, cut plan |
| [`main.tex`](main.tex) | LaTeX driver |
| [`refs.bib`](refs.bib) | Bibliography starter |
| [`sections/`](sections/) | One file per section (fill stubs) |
| [`figures/`](figures/) | Place PDF/PNG figures here |
| [`tables/`](tables/) | Optional exported tables |

## Compile

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Stubs use `\fbox` placeholders where figures are missing so the document still compiles.

## Fill order (recommended)

1. Sections 8–9 (results you already have)  
2. Sections 5–7 (data/methods/protocol from `experiments/PAPER_METHODOLOGY.md`)  
3. Sections 1, 4, 10–13  
4. Section 2 (related work + expand `refs.bib` to 35–60 entries)  
5. Appendix figures from Kaggle outputs  

## Length check

After a full draft, run and inspect page count. Expand Appendix A and Related Work first if under 35 pages; tighten Discussion/Appendix if over 40.
