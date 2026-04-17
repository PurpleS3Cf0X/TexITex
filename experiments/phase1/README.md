# Phase 1: Baseline DiT Explorations

**Question:** Does adding a position channel or Hilbert curve ordering help a small DiT?

## Experiments

| ID | Description | New Feature | Epochs |
|----|-------------|-------------|--------|
| P1-A | Default DiT (depth=8, dim=384) | baseline | 150 |
| P1-B1 | + Position channel (0→1 row-major) | pos_channel | 150 |
| P1-C1 | + Hilbert VQ-GAN token reordering | tokence_hilbert | 150 |

## Winner: P1-B1

Position channel provides a consistent improvement. Hilbert reordering hurts with a small DiT (the model can't exploit the locality — it needs more capacity).

## Training Commands

```bash
# P1-A
/usr/bin/python3 run_poc.py --use_vqgan --backbone dit --data_source security \
    --max_samples 50000 --num_epochs 150 --ae_variant tokence_big_long

# P1-B1 (WINNER)
/usr/bin/python3 run_poc.py --use_vqgan --backbone dit --data_source security \
    --max_samples 50000 --num_epochs 150 --ae_variant tokence_big_long --dit_pos_channel

# P1-C1
/usr/bin/python3 run_poc.py --use_vqgan --backbone dit --data_source security \
    --max_samples 50000 --num_epochs 150 --ae_variant tokence_hilbert
```

## Logs
- P1-A: `runs/dit_long.log`
- P1-B1: `runs/dit_posemb.log`
- P1-C1: `runs/dit_hilbert.log`
