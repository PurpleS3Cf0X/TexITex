# Phase 3: Sequence-Order Auxiliary Loss

**Question:** Can a causal LSTM forcing left-to-right token ordering improve sentence coherence?

## Key Innovation

A `SequencePredictor` (2-layer causal LSTM) is trained jointly with the DiT:
1. Extract per-token features from the predicted x0 (flatten 2×2 patches → 16-dim each, row-major order)
2. LSTM takes tokens 0..62, predicts tokens 1..63
3. Loss = `0.5 × L1(pred, true) + 0.5 × (1 − cosine_sim(pred, true))`
4. Gradient flows: LSTM loss → x0_pred → noise_pred → DiT weights

This teaches the DiT that adjacent tokens in reading order should have semantically coherent relationship — the missing ingredient for sentence structure.

## Experiments

| ID | Description | seq_weight | Epochs |
|----|-------------|-----------|--------|
| P3-A | Large DiT + pos_channel (baseline) | — | 300 |
| P3-B | + sequence-order loss w=0.5 (WINNER) | 0.5 | 300 |
| P3-C | + sequence-order loss + coherence loss | 0.5 / 0.1 | 300 |

## Winner: P3-B

Sequence loss at weight=0.5 is the critical innovation. P3-C (adding coherence on top) showed no clear improvement.

## Training Commands

```bash
# P3-B (WINNER)
/usr/bin/python3 run_poc.py --use_vqgan --backbone dit --data_source security \
    --max_samples 50000 --num_epochs 300 --ae_variant tokence_big_long \
    --dit_depth 12 --dit_hidden_dim 512 --dit_num_heads 8 \
    --dit_pos_channel --dit_sequence_loss --dit_sequence_weight 0.5 \
    --checkpoint_dir runs/ckpt_p3b
```

## Logs
- P3-A: `runs/dit_p3a.log`
- P3-B: `runs/dit_p3b.log`
- P3-C: `runs/dit_p3c.log`
