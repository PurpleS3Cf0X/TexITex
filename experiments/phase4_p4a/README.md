# Phase 4-A: Best Model — Self-Conditioning + Token Boundary Channel

**Status:** SOTA Configuration (as of April 2026)
**Checkpoint:** `runs/ckpt_p4a/best.pt` (epoch 200 of 300)

## What This Phase Introduced

Building on Phase 3-B (large DiT + sequence-order loss), Phase 4 adds:

1. **Self-conditioning** — The model feeds its own previous x0 prediction back as extra input channels.
   Each DDIM step sees both the noisy latent AND the previous step's best guess at the clean image.
   During training: 50% of steps use a first-pass x0 prediction; 50% use zeros (dropout).
   This teaches iterative refinement — critical for coherent output.

2. **Token boundary channel** — An explicit binary map marking the 2×2 patch borders.
   Prevents inter-token bleed: the model knows exactly where one token ends and the next begins.
   This was the single clearest improvement over P3-B in image-space inspection.

## Input Channel Layout

```
Input to DiT: (B, 34, 16, 16)
├── [0]      position channel   (1ch)  — 0→1 normalized, row-major reading order
├── [1]      boundary channel   (1ch)  — 1.0 at patch borders, 0.0 inside
├── [2:18]   self-cond x0_pred (16ch) — previous DDIM step's clean prediction (zeros on first step)
└── [18:34]  noisy latent      (16ch) — x_t from forward diffusion process
```

## Training Command (exact)

```bash
/usr/bin/python3 run_poc.py \
    --use_vqgan --backbone dit --data_source security \
    --max_samples 50000 --num_epochs 300 \
    --ae_variant tokence_big_long \
    --dit_depth 12 --dit_hidden_dim 512 --dit_num_heads 8 \
    --dit_pos_channel \
    --dit_self_cond \
    --dit_boundary_channel \
    --dit_sequence_loss --dit_sequence_weight 0.5 \
    --checkpoint_dir runs/ckpt_p4a
```

## Generation Command (optimal settings)

```bash
# Generate 32 samples and rank by composite quality
/usr/bin/python3 gen_winners.py  # uses runs/ckpt_p4a/best.pt, 200 DDIM steps

# Or manually:
/usr/bin/python3 run_poc.py \
    --eval_only --use_vqgan --backbone dit --data_source security \
    --ae_variant tokence_big_long \
    --dit_depth 12 --dit_hidden_dim 512 --dit_num_heads 8 \
    --dit_pos_channel --dit_self_cond --dit_boundary_channel \
    --dit_sequence_loss --dit_sequence_weight 0.5 \
    --checkpoint_dir runs/ckpt_p4a \
    --ddim_steps 200 --num_generate 16
```

## Key Results

*Measured from `runs/ckpt_p4a/best.pt` (epoch 200), 64 samples, 200 DDIM steps.*
*Eval command: `python gen_winners.py --ckpt runs/ckpt_p4a --ddim 200 --num 64`*
*Eval log: `runs/p4a_best_eval.log`*

| Metric | Value |
|--------|-------|
| Composite score (mean, n=64) | 0.104 |
| Composite score (best sample) | 0.372 |
| Composite score (P90) | 0.212 |
| Bigram coherence (mean) | 0.272 |
| Bigram coherence (best sample) | 0.831 |
| Real-word ratio (mean) | 0.683 |
| Median PPL | 197 |
| Prose ratio | 100% (64/64) |
| VQ-GAN roundtrip accuracy | 89.8% |
| vs epoch 300 (final.pt) | mean 0.104 vs 0.097 — epoch 200 wins |

**Best generated sample (comp=0.372, bigram=0.831):**
> "a simulated adversary engagement. Your objectives include testing detection capabilities,
> exercising incident response, identifying security gaps. You employ realistic adversary
> TTPs mapped to MITRE ATT&CK, maintain operational security, and adapt your approach based
> on blue team responses."

## Key Findings (sweep experiments)

### Checkpoint sweep (epoch 50/100/150/200/250/300):
- Peak quality at **epoch 200** — composite 0.151 (mean)
- Overtrained by epoch 250: composite drops to 0.082
- `best.pt` = epoch 200 checkpoint

### DDIM step sweep (50/100/150/200/300/400/500 steps):
- Rises monotonically 50→200 steps
- **Mode collapse cliff at ≥300 steps** — self-conditioning feedback loop amplifies noise
- Optimal: **200 steps** (sweet spot)

### Ablation P4-B (seq_weight=0.2):
- Bigram coherence: 0.030 (vs P4-A's 0.379)
- URL/HTTP mode collapse — completely degenerate outputs
- **Conclusion: sequence_weight=0.5 is mandatory**

## Architecture Summary

```
VQ-GAN Encoder (tokence_big_long):
  d_model=1536 → hidden=512 → latent=(16,16,16) → codebook(1024×16)
  Token cross-entropy auxiliary loss (sampled softmax, 4096 negatives)
  Roundtrip token accuracy: 89.8%

DiT:
  depth=12, hidden_dim=512, num_heads=8, patch_size=2
  adaLN-Zero conditioning on timestep embedding
  57.8M parameters (+ 239.7K SequencePredictor LSTM = 58.0M total)
  Trained for 300 epochs; best at epoch 200
```

## Files in this snapshot

| File | Phase it was finalized in |
|------|--------------------------|
| `code/config.py` | Phase 4 (CFG/consistency fields added in Phase 5) |
| `code/dit.py` | Phase 4 (non-square + CFG added in Phase 5) |
| `code/diffusion.py` | Phase 4 (CFG/CD samplers added in Phase 5) |
| `code/vqgan.py` | Phase 4 (non-square latent support added in Phase 5) |
| `code/train.py` | Phase 4 (Phase 5: non-square DiT args added) |
| `code/generate.py` | Phase 4 (Phase 5: non-square + CFG args added) |
| `code/dataset.py` | Phase 4 (Phase 5: seq_len-specific cache filename) |
| `code/eval_quality.py` | Phase 4 (composite metric, prose filter) |

> **NOTE:** The code in `code/` is the Phase-5-current snapshot. Phase 4 introduced
> `dit_self_cond`, `dit_boundary_channel`, `training_loss_self_cond()`, and
> `ddim_sample_self_cond()`. All Phase-5 additions are clearly gated by kwargs
> (`use_cfg`, `input_h/input_w`) and do not change Phase-4 behavior when those
> kwargs are absent.

## To reproduce from scratch

```bash
# 1. Train VQ-GAN encoder (if data_cache/vqgan_projector_tokence_big_long.pt missing)
/usr/bin/python3 run_poc.py --use_vqgan --ae_variant tokence_big_long \
    --vqgan_epochs 40 --data_source security --max_samples 50000 \
    --backbone dit --roundtrip_only

# 2. Train DiT (300 epochs, ~24h on M4 Mac)
/usr/bin/python3 run_poc.py \
    --use_vqgan --backbone dit --data_source security \
    --max_samples 50000 --num_epochs 300 \
    --ae_variant tokence_big_long \
    --dit_depth 12 --dit_hidden_dim 512 --dit_num_heads 8 \
    --dit_pos_channel --dit_self_cond --dit_boundary_channel \
    --dit_sequence_loss --dit_sequence_weight 0.5 \
    --checkpoint_dir runs/ckpt_p4a

# 3. Use best.pt = epoch 200 checkpoint (rename or check timestamps)
# 4. Evaluate
/usr/bin/python3 gen_winners.py
```
