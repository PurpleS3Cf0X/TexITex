# Building the Paper

## Quick build (tectonic — recommended)
```bash
cd paper/
tectonic main.tex
# Output: main.pdf
```

## Full build (pdflatex — if MacTeX installed)
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Install tectonic (if not present)
```bash
brew install tectonic
```

## Install MacTeX (full TeX distribution, ~4GB)
https://www.tug.org/mactex/

---

# ArXiv Submission Checklist

## Step 1 — Prepare files
```bash
cd paper/
tar -czf ../texitex_arxiv_v1.tar.gz main.tex refs.bib figures/
```

## Step 2 — Create ArXiv account
- Go to: https://arxiv.org/register
- You need an institutional email OR an endorser

## Step 3 — Submit
1. Go to https://arxiv.org/submit
2. Subject area: **cs.CL** (Computation and Language)
   - Cross-list with: cs.LG, cs.CV
3. Upload the .tar.gz file
4. Fill in:
   - Title: TexITex: Parallel Text Generation via Token Embedding Diffusion in 2D Image Space
   - Authors: [your name + affiliation]
   - Abstract: [copy from paper]
   - MSC class: leave blank (CS paper)
   - ACM class: I.2.7 (Natural Language Processing)
5. Preview the compiled PDF
6. Submit → appears next business day

## Step 4 — After Phase 5 completes
Update results tables, submit as v2:
- ArXiv allows unlimited versioning
- v1 = Phase 4 SOTA
- v2 = + Phase 5 (CFG, 128-tok, consistency distillation)

## Notes
- Checkpoints are too large for ArXiv (supplementary max ~10MB)
- Host checkpoints on HuggingFace Hub or Zenodo separately
- Link to them in the paper footnote
