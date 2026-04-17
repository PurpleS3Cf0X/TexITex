# Source Code — Phase 4-A (SOTA Configuration)

This directory contains the exact code snapshot that produced the best reported results:

> **Composite score 0.372 (best sample) | Bigram coherence 0.831 | 89.8% roundtrip accuracy**  
> Checkpoint: epoch 200 of 300 | 200 DDIM steps | Apple M4 64GB

---

## Files

| File | What it does |
|------|-------------|
| `config.py` | All hyperparameters as a Python dataclass — single source of truth |
| `vqgan.py` | VQ-GAN encoder/decoder: compresses 64 token embeddings → (16,16,16) latent image |
| `dit.py` | DiT backbone: 34-channel input, 12 blocks, adaLN-Zero, patch size 2×2 |
| `diffusion.py` | DDPM/DDIM noise schedule, self-conditioning, LSTM sequence-order loss |
| `train.py` | Training loop with EMA, checkpoint saving, best-epoch tracking |
| `dataset.py` | Text → tokens → embeddings pipeline using Qwen2.5-1.5B embedding table |
| `generate.py` | DDIM sampling + VQ-GAN decode → text |
| `eval_quality.py` | Composite metric: bigram_coherence × real_word_ratio × 1/(1+log(PPL)/5) |
| `run_poc.py` | Unified CLI: train / eval / roundtrip-only modes |
| `gen_winners.py` | Generate N samples, rank by composite score, print top-K |

---

## Key Design Choices (P4-A)

- **34-channel DiT input**: `[pos(1) | boundary(1) | self_cond(16) | noisy_latent(16)]`
- **Self-conditioning**: 50% dropout during training; feeds previous DDIM step's x0_pred back
- **Token boundary channel**: binary map at 2×2 patch edges — prevents inter-token bleed
- **LSTM sequence loss** (`dit_sequence_weight=0.5`): enforces left-to-right token order
  — ⚠️ reducing this below 0.5 causes catastrophic collapse
- **Best checkpoint**: epoch 200 (not epoch 300 — overtrained by epoch 250)
- **DDIM steps**: 200 (mode-collapse cliff at ≥300 due to self-cond feedback loop)

---

## Reproduce

```bash
pip install -r ../requirements.txt

# 1. Train VQ-GAN encoder (~2h on M4)
python run_poc.py --use_vqgan --ae_variant tokence_big_long \
    --vqgan_epochs 40 --data_source security --max_samples 50000 \
    --backbone dit --roundtrip_only

# 2. Train DiT — 300 epochs, use checkpoint at epoch 200 (~22h on M4)
python run_poc.py \
    --use_vqgan --backbone dit --data_source security \
    --max_samples 50000 --num_epochs 300 \
    --ae_variant tokence_big_long \
    --dit_depth 12 --dit_hidden_dim 512 --dit_num_heads 8 \
    --dit_pos_channel --dit_self_cond --dit_boundary_channel \
    --dit_sequence_loss --dit_sequence_weight 0.5 \
    --checkpoint_dir runs/ckpt_p4a

# 3. Generate and rank samples
python gen_winners.py
```

---

## Note on Phase 5

The files in this directory are the **Phase 4 snapshot**. Phase 5 extensions
(128-token non-square latent, CFG conditioning, consistency distillation) exist
in a separate development branch and are not yet included here — results pending.
