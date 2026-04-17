# Phase 2: Scaling DiT + Coherence Loss

**Question:** Does scaling the DiT (depth=12, dim=512) and/or adding spatial coherence loss help?

## Experiments

| ID | Description | New Feature | Epochs |
|----|-------------|-------------|--------|
| P2-A | Large DiT (57.8M) + pos_channel | scale up | 300 |
| P2-B | Standard DiT + pos_channel + coherence loss | coherence | 300 |
| P2-C | Large DiT + pos_channel | scale up | 150 |

## Winner: P2-A (large DiT)

Scaling from 8→12 layers and 384→512 dim gives a bigger quality jump than coherence loss. The coherence loss (neighbor cosine similarity matching) helps slightly but is dominated by scale.

## Key Insight

The small 8-layer DiT doesn't have enough capacity to model complex token relationships. Once we scale to 12 layers, quality jumps noticeably even without new losses.

## Training Commands

```bash
# P2-A (WINNER)
/usr/bin/python3 run_poc.py --use_vqgan --backbone dit --data_source security \
    --max_samples 50000 --num_epochs 300 --ae_variant tokence_big_long \
    --dit_depth 12 --dit_hidden_dim 512 --dit_num_heads 8 \
    --dit_pos_channel --checkpoint_dir runs/ckpt_p2a

# P2-B
/usr/bin/python3 run_poc.py --use_vqgan --backbone dit --data_source security \
    --max_samples 50000 --num_epochs 300 --ae_variant tokence_big_long \
    --dit_pos_channel --dit_coherence_loss --dit_coherence_weight 0.1 \
    --checkpoint_dir runs/ckpt_p2b
```

## Logs
- P2-A: `runs/dit_p2a.log`
- P2-B: `runs/dit_p2b.log`
- P2-C: `runs/dit_p2c.log`
