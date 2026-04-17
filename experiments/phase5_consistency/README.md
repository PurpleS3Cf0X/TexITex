# Phase 5 — Feature 1: Consistency Distillation (1–4 Step Inference)

**Hypothesis:** Distilling the 200-step P4-A DDIM teacher into a Consistency Model
allows high-quality generation in 1–4 steps — a 50–200× inference speedup —
while preserving most of the generation quality.

## What's New

### Consistency Distillation Loss

For each training step, we sample a time pair `(t_n, t_{n-1})` where `t_{n-1} = t_n - step_size`:
```
CD loss = MSE(f_student(x_{t_n}, t_n), sg[f_EMA-teacher(x_{t_{n-1}}, t_{n-1})])
```
Where:
- `x_{t_{n-1}}` = one DDIM step from `x_{t_n}` using the EMA teacher
- `sg[·]` = stop-gradient (target is detached)
- The student is trained to be consistent across adjacent noise levels

### Multi-Step Inference

At inference, the student can use 1, 2, or 4 steps. With K steps:
1. Denoise from T → T/K
2. Add noise back to T(K-1)/K
3. Denoise again → etc.

This zigzag schedule enables multi-step refinement with the fast student model.

### `distill.py` — Standalone Training Script

Key functions:
- `_one_ddim_step(diffusion, x_t, t_cur, t_next, model_fn)`: single DDIM step for teacher rollout
- `build_aux_fn(pos_map, boundary_map, use_self_cond, latent_channels, device)`: builds aux input (pos + boundary + self_cond channels)
- `distill(...)`: main training loop with EMA teacher, CD loss, periodic evaluation

### EMA Teacher

The EMA teacher is updated every step:
```python
ema_decay = 0.999
for ema_p, student_p in zip(ema_model.parameters(), student.parameters()):
    ema_p.data.mul_(ema_decay).add_(student_p.data, alpha=1 - ema_decay)
```

## Training Command

```bash
/usr/bin/python3 distill.py \
    --teacher_ckpt runs/ckpt_p4a/best.pt \
    --output_dir runs/ckpt_p5_consistency \
    --epochs 75 --lr 1e-4 --step_size 10 \
    2>&1 | tee -a runs/p5_distill.log
```

**Step size**: adjacent timesteps differ by 10 (out of T=1000).
**Total pairs**: T/step_size = 100 possible (t, t-step_size) pairs.

## Training Status

| Step | Status | Notes |
|------|--------|-------|
| Distillation training | Training | 75 epochs from P4-A best.pt |
| 1-step inference eval | Pending | After training |
| 2-step inference eval | Pending | After training |
| 4-step inference eval | Pending | After training |

**Log:** `runs/p5_distill.log`

## Expected Outcomes

| Inference steps | Expected quality | Speedup vs P4-A |
|----------------|-----------------|-----------------|
| 1 step | Moderate (coherent but rough) | 200× |
| 2 steps | Good | 100× |
| 4 steps | Near P4-A quality | 50× |
| 200 steps (P4-A DDIM) | 0.151 composite | 1× (baseline) |

Consistency models typically retain 80–90% of teacher quality at 4 steps.

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Teacher checkpoint | `runs/ckpt_p4a/best.pt` (epoch 200) |
| Student architecture | Same as teacher (copied weights) |
| EMA decay | 0.999 |
| Step size | 10 (adjacent noise levels) |
| Learning rate | 1e-4 |
| Epochs | 75 |
| Batch size | 32 |

## Results (to be filled after training)

| Inference steps | Composite mean | Bigram coherence | Real-word ratio | Speedup |
|----------------|----------------|-----------------|-----------------|---------|
| 1 step | TBD | TBD | TBD | ~200× |
| 2 steps | TBD | TBD | TBD | ~100× |
| 4 steps | TBD | TBD | TBD | ~50× |
| P4-A 200 steps | 0.151 | 0.379 | 0.736 | 1× |

## Checkpoint

- Distilled student: `runs/ckpt_p5_consistency/best.pt`
- EMA teacher (frozen reference): `runs/ckpt_p4a/best.pt`

## Notes

- Distillation is slower per batch (~2.5s/it vs 1.0s/it for normal training) due to teacher rollout
- The student model has identical architecture to teacher (P4-A: depth=12, dim=512, heads=8, 34ch)
- CD loss scale is naturally small (~0.003) because it's in latent space; this is expected
- If distillation diverges, reduce `ema_decay` to 0.995 or reduce `step_size` to 5
