# Phase 5 — Feature 2: 128-Token Sequences (Non-Square Latent)

**Hypothesis:** Doubling the context window from 64 to 128 tokens allows the model to capture
longer-range dependencies and generate more complete sentences, at the cost of a 2× larger latent
space and longer training.

## What's New

| Component | P4-A (64 tokens) | P5-128tok (128 tokens) |
|-----------|-----------------|----------------------|
| Token grid | 8×8 | 8×16 |
| Latent shape | (16, 16, 16) | (16, 16, 32) |
| DiT patches (2×2) | 8×8 = 64 tokens | 8×16 = 128 positions |
| Input channels | 34 | 34 (same breakdown) |
| VQGAN variant | `tokence_big_long` | `tokence_128` |

### Architecture Changes

- **`vqgan.py`**: `latent_h=16, latent_w=32` (was 16×16); encoder/decoder handle non-square via `(B, C, H, W)` with H≠W
- **`dit.py`**: `Unpatch` class generalized to `grid_h, grid_w`; `make_position_map` and `make_boundary_map` accept `latent_h, latent_w, grid_h, grid_w`
- **`dit.py`**: `DiT.__init__` accepts `input_h, input_w` instead of assuming square
- **`train.py`**: DiT constructor call updated with `input_h=img_h, input_w=img_w`
- **`generate.py`**: `load_checkpoint` and `generate_images` updated for non-square latents
- **`dataset.py`**: Embedding cache filename is `embeddings_tmp_{seq_len}.npy` (avoids collision with 64-token cache)
- **`run_poc.py`**: `ae_variant=tokence_128` triggers `cfg.seq_len=128, cfg.vqgan_latent_h=16, cfg.vqgan_latent_w=32`

## Training Pipeline

### Step 1: VQ-GAN for 128-token sequences

```bash
# Runs automatically via run_poc.py when --ae_variant tokence_128 is specified
# VQ-GAN variant: tokence_128 (16×32 latent, 512-dim hidden, no discriminator)
# Epochs: 40, Batch: 64, Loss: recon + cosine + commit + token_ce
```

### Step 2: DiT training on 128-token latents

```bash
/usr/bin/python3 run_poc.py \
    --use_vqgan --ae_variant tokence_128 --backbone dit \
    --data_source security --max_samples 50000 --num_epochs 300 \
    --dit_depth 12 --dit_hidden_dim 512 --dit_num_heads 8 \
    --dit_pos_channel --dit_self_cond --dit_boundary_channel \
    --dit_sequence_loss --dit_sequence_weight 0.5 \
    --checkpoint_dir runs/ckpt_p5_128tok \
    2>&1 | tee -a runs/p5_128tok_dit.log
```

This command is run **after** VQ-GAN completes and the latent cache is built.

## Training Status

| Step | Status | Notes |
|------|--------|-------|
| VQ-GAN (tokence_128) | Training | 40 epochs, ~9h total on M4 Mac |
| DiT (128-token) | Pending | Launches after VQ-GAN finishes |

**VQ-GAN log:** `runs/p5_128tok_vqgan.log`
**DiT log (future):** `runs/p5_128tok_dit.log`

## Expected Outcomes

- VQ-GAN roundtrip accuracy: target >80% (128 tokens are harder than 64; 16×32 latent has more capacity)
- Composite score: should exceed P4-A (0.151) due to longer context
- Generated samples should show fuller sentences, not just phrases
- Inference will be slower: 16×32 = 512 patches vs 64 patches in P4-A → ~8× more compute per DDIM step

## Potential Risks

- Longer sequences are harder for the sequence-order LSTM (predicts 1..127 from 0..126)
- 16×32 non-square latent may need longer training to converge
- Memory pressure: latent tensors 2× larger, but M4 64GB can handle it

## Key Differences from P4-A Config

```json
{
  "seq_len": 128,
  "vqgan_latent_h": 16,
  "vqgan_latent_w": 32,
  "ae_variant": "tokence_128",
  "dit_img_h": 16,
  "dit_img_w": 32
}
```

## Results (to be filled after training)

| Metric | Value |
|--------|-------|
| VQ-GAN roundtrip accuracy | TBD |
| Best epoch | TBD |
| Composite score (mean) | TBD |
| Composite score (best) | TBD |
| Bigram coherence | TBD |
| Real-word ratio | TBD |
| vs P4-A (0.151 mean) | TBD |

## Checkpoint

- VQ-GAN: `data_cache/vqgan_projector_tokence_128.pt` (auto-saved by run_poc.py)
- DiT best: `runs/ckpt_p5_128tok/best.pt`
