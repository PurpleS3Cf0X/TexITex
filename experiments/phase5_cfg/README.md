# Phase 5 — Feature 3: Classifier-Free Guidance (CFG)

**Hypothesis:** Conditioning the DiT on a compact projection of the original text embedding
(the "prompt") and applying classifier-free guidance at inference will steer generation toward
more coherent, on-topic security-domain text.

## What's New

### `ConditionProjector` — new module in `dit.py`

```python
class ConditionProjector(nn.Module):
    """Projects a raw text embedding (1536-dim) to DiT hidden dim (512-dim)."""
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, cond):
        return self.net(cond)
```

The projected embedding is added to the timestep embedding `c` inside the DiT:
```python
c = timestep_embed + (self.cond_proj(cond_embed) if cond_embed is not None else 0)
```

### CFG Training (`finetune_cfg.py`)

- **Base**: Load P4-A best checkpoint (epoch 200, `runs/ckpt_p4a/best.pt`)
- **Rebuild**: Add `ConditionProjector` to DiT; copy all other weights from P4-A
- **Null-condition dropout**: 15% of batches replace `cond_embed` with `None` (null token)
- **Loss**: Same as P4-A — diffusion + 0.5 × sequence_order LSTM
- **LR**: 5e-5 (lower than full training, fine-tuning regime)
- **Epochs**: 100, batch size 32

```bash
/usr/bin/python3 finetune_cfg.py \
    --teacher_ckpt runs/ckpt_p4a/best.pt \
    --output_dir runs/ckpt_p5_cfg \
    --epochs 100 --lr 5e-5 \
    2>&1 | tee -a runs/p5_cfg_finetune.log
```

### CFG Inference (`diffusion.py` — `ddim_sample_cfg`)

Double forward pass at each DDIM step:
```python
noise_cond   = model(x_input, t, cond_embed=cond_embed)
noise_uncond = model(x_input, t, cond_embed=None)
noise_pred   = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

Typical guidance scales: 1.5, 2.0, 3.0

## Training Status

| Step | Status | Notes |
|------|--------|-------|
| CFG fine-tuning | Training | 100 epochs from P4-A best.pt |
| Inference evaluation | Pending | Grid search over guidance_scale ∈ {1.0, 1.5, 2.0, 3.0} |

**Log:** `runs/p5_cfg_finetune.log`

## Condition Signal

The condition embedding is the **mean-pooled Qwen2.5-1.5B token embedding** of the input
sequence (shape: 1536-dim). This is the same embedding the VQ-GAN encodes, so the condition
and the latent are naturally aligned.

For generation, we use:
1. A domain-specific prompt (e.g., a red-team TTP description)
2. Or a held-out test sequence from the security corpus
3. Or the null token (unconditional generation) to compare

## Expected Outcomes

- **Conditional sampling quality** should significantly exceed P4-A (unconditioned)
- **Bigram coherence** target: 0.50+ (vs P4-A best 0.379)
- **Real-word ratio** should stay high (>0.70)
- Guidance scale 1.5–2.5 typically optimal; too high → mode collapse
- The 15% null dropout ensures the model retains unconditional generation ability

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Base checkpoint | `runs/ckpt_p4a/best.pt` (epoch 200) |
| CFG cond dim | 1536 (Qwen2.5-1.5B embedding dim) |
| CFG hidden dim | 512 (matches DiT hidden dim) |
| Null dropout | 15% |
| Fine-tune LR | 5e-5 |
| Fine-tune epochs | 100 |
| Guidance scale (eval) | 1.0, 1.5, 2.0, 3.0 |

## Results (to be filled after training)

| Metric | Uncond (P4-A) | CFG scale=1.5 | CFG scale=2.0 | CFG scale=3.0 |
|--------|--------------|---------------|---------------|---------------|
| Composite mean | 0.151 | TBD | TBD | TBD |
| Composite best | 0.421 | TBD | TBD | TBD |
| Bigram coherence | 0.379 | TBD | TBD | TBD |
| Real-word ratio | 0.736 | TBD | TBD | TBD |

## Checkpoint

- Fine-tuned model: `runs/ckpt_p5_cfg/best.pt`
- Best guidance scale: TBD (to be determined by eval sweep)
