# TexITex Experiments — Reproducibility Archive

**Project:** Token-Image Diffusion (TexITex)
**Goal:** Encode token embeddings as 2D images, train image diffusion, decode back to text.

Each subdirectory contains:
- `README.md` — experiment description, findings, exact training commands
- `config.json` — all hyperparameters as structured JSON
- `code/` — snapshot of all Python source files active at that phase
- `results/` — evaluation logs and generated samples (where available)
- `run_phase*.sh` — the exact shell script used to launch training

## Experiment Progression

```
Phase 1 (P1): Baseline small DiT (8 layers, 384 dim)
   ├── P1-A: 150 epochs, no extras
   ├── P1-B1: + position channel (WINNER → phase 2 baseline)
   └── P1-C1: + Hilbert VQ-GAN reordering

Phase 2 (P2): Scale up DiT architecture
   ├── P2-A: Large DiT 57.8M params, pos_channel 300ep (WINNER → phase 3 baseline)
   ├── P2-B: Standard DiT + coherence loss 300ep
   └── P2-C: Large DiT 150ep

Phase 3 (P3): Sequence-order auxiliary loss
   ├── P3-A: Large DiT + pos_channel (baseline)
   ├── P3-B: + sequence-order LSTM loss w=0.5 (WINNER → phase 4 baseline)
   └── P3-C: + sequence-order + coherence

Phase 4 (P4): Self-conditioning + Token Boundary Channel
   └── P4-A: + self_cond + boundary_channel (SOTA ★)
       Ablation: P4-B: same but seq_weight=0.2 → catastrophic failure

Phase 5 (P5): Three parallel extensions (in-flight)
   ├── P5-128tok: 128-token sequences (16×32 non-square latent)
   ├── P5-CFG:    Classifier-Free Guidance (prompt-conditioned)
   └── P5-CD:     Consistency Distillation (1-4 step inference)
```

## Best Configuration: P4-A

| Parameter | Value |
|-----------|-------|
| Encoder | tokence_big_long (VQ-GAN, 17.6M params) |
| Encoder roundtrip accuracy | 89.8% |
| DiT depth / dim / heads | 12 / 512 / 8 |
| DiT parameters | 57.8M (+239.7K LSTM = 58.0M total) |
| Input channels | 34 (16 latent + 1 pos + 1 boundary + 16 self_cond) |
| Sequence loss weight | 0.5 |
| Training epochs | 300 (best at epoch 200) |
| Inference steps | 200 DDIM |
| Composite score (mean, n=64) | 0.104 |
| Composite score (best sample) | 0.372 |
| Best bigram coherence | 0.831 (best sample) / 0.272 (mean) |
| Real-word ratio (mean) | 0.683 |

## Evaluation Metric

**Composite score** = bigram_coherence × real_word_ratio × 1/(1 + log(PPL)/5)

- `bigram_coherence`: fraction of consecutive token pairs that appear in real English text
- `real_word_ratio`: fraction of tokens that are real English words  
- `PPL`: perplexity of decoded text under a language model (gameable by JSON boilerplate → filtered)
- `is_prose()`: filter to exclude JSON/URL-heavy outputs before PPL evaluation

## Checkpoint locations

| Experiment | Checkpoint |
|-----------|-----------|
| P4-A (best) | `runs/ckpt_p4a/best.pt` (epoch 200) |
| P3-B | `runs/ckpt_p3b/` |
| P3-A | `runs/ckpt_p3a/` |
| P2-A | `runs/ckpt_p2a/` |
| P5-128tok | `runs/ckpt_p5_128tok/` (in progress) |
| P5-CFG | `runs/ckpt_p5_cfg/` (in progress) |
| P5-CD | `runs/ckpt_p5_consistency/` (in progress) |
| VQ-GAN encoders | `data_cache/vqgan_projector_*.pt` |

## Code Snapshots — What Changed When

| Feature | Phase introduced | Key files modified |
|---------|-----------------|-------------------|
| Position channel (pos_channel) | Phase 1 | `dit.py`, `diffusion.py`, `train.py` |
| Large DiT (depth=12, dim=512) | Phase 2 | `config.py`, `run_poc.py` |
| Coherence loss | Phase 2 | `diffusion.py` |
| Sequence-order LSTM loss | Phase 3 | `diffusion.py`, `train.py` |
| Self-conditioning | Phase 4 | `dit.py`, `diffusion.py`, `train.py`, `generate.py` |
| Token boundary channel | Phase 4 | `dit.py`, `diffusion.py`, `train.py`, `generate.py` |
| Non-square latent (128-tok) | Phase 5 | `vqgan.py`, `dit.py`, `train.py`, `generate.py`, `dataset.py` |
| CFG (ConditionProjector) | Phase 5 | `dit.py`, `diffusion.py`, `finetune_cfg.py` |
| Consistency Distillation | Phase 5 | `diffusion.py`, `distill.py` |

## Hardware & Software

- **Hardware:** Apple Mac Mini M4, 64GB unified memory
- **Backend:** MPS (Metal Performance Shaders)
- **Python:** 3.9 (`/usr/bin/python3`)
- **PyTorch:** 2.3.1
- **Base LM:** Qwen/Qwen2.5-1.5B (embedding table only, not fine-tuned)
- **Corpus:** Cybersecurity domain (red-team TTPs + blue-team playbooks, 50K sequences)
