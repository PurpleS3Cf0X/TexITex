# Experiments — Phase 4-A (SOTA)

This directory contains the full reproducibility record for the best-performing
configuration: **Phase 4-A, epoch 200**.

## What's here

| Path | Contents |
|------|----------|
| `phase4_p4a/README.md` | Architecture details, training command, key findings |
| `phase4_p4a/config.json` | Every hyperparameter used |
| `phase4_p4a/results/` | Eval logs, checkpoint sweep, DDIM step sweep, top-20 samples |

## Result Summary

| Metric | Value |
|--------|-------|
| VQ-GAN roundtrip accuracy | **89.8%** |
| Composite score — mean (n=64) | 0.104 |
| Composite score — best sample | **0.372** |
| Bigram coherence — best sample | **0.831** |
| Real-word ratio — mean | 0.683 |
| Median perplexity | 197 |
| DiT parameters | 57.8M + 239.7K LSTM = **58.0M** |
| Best checkpoint | epoch 200 / 300 |
| Optimal DDIM steps | 200 |

## How we got here (brief)

Four phases of development led to this configuration:

1. **Phase 1** — baseline small DiT; found position channel essential
2. **Phase 2** — scaled to 57.8M DiT; stable training, moderate quality
3. **Phase 3** — added LSTM sequence-order loss (weight=0.5); coherence improved sharply
4. **Phase 4-A** — added self-conditioning + token boundary channel → **current SOTA**

> The earlier phase logs are kept in the private development repo.
> Only the final working configuration is published here.
