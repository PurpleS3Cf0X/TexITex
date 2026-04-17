"""
Combined winner evaluation
===========================
Generate a large batch of samples from P4-A, compute per-sample composite score,
and display the top-K winners. Writes two outputs:

  runs/dit_p4a_winners.log   — all samples (for use with eval_quality.py)
  runs/winners_top20.txt     — ranked top-20 by composite score

Run:
  python gen_winners.py --ckpt runs/ckpt_p4a --ddim 200 --num 64
"""
import argparse
import sys
import math
from pathlib import Path
import numpy as np
import torch

from config import Config
from generate import load_checkpoint, generate_images, decode_to_text
from dataset import prepare_dataset
from eval_quality import (
    bigram_coherence, build_bigram_table, lm_perplexity,
    real_word_ratio, structural_noise_ratio, composite_score, is_prose,
)
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/ckpt_p4a",
                    help="Checkpoint directory")
    ap.add_argument("--ddim", type=int, default=200, help="DDIM sampling steps")
    ap.add_argument("--num", type=int, default=64, help="Number of samples to generate")
    ap.add_argument("--out_log", type=str, default="runs/dit_p4a_winners.log",
                    help="Output log with all generated samples")
    ap.add_argument("--out_top", type=str, default="runs/winners_top20.txt",
                    help="Top-20 ranked output")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--best", action="store_true", default=True,
                    help="Use best.pt instead of final.pt (default: True)")
    ap.add_argument("--final", dest="best", action="store_false",
                    help="Use final.pt (last epoch) instead of best.pt")
    args = ap.parse_args()

    # ---- Config (mirrors P4-A architecture) ----
    cfg = Config()
    cfg.use_vqgan = True
    cfg.backbone = "dit"
    cfg.data_source = "security"
    cfg.data_paths = (
        "/Volumes/R3D_1/redteam_project/data/final_v2/train.jsonl",
        "/Volumes/R3D_1/blueteam_project/data/final/train.jsonl",
    )
    cfg.max_samples = 50000
    cfg.ae_variant = "tokence_big_long"
    cfg.vqgan_hidden_dim = 512
    cfg.dit_pos_channel = True
    cfg.dit_boundary_channel = True
    cfg.dit_self_cond = True
    cfg.dit_sequence_loss = True
    cfg.dit_sequence_weight = 0.5
    cfg.dit_depth = 12
    cfg.dit_hidden_dim = 512
    cfg.dit_num_heads = 8
    cfg.checkpoint_dir = args.ckpt
    cfg.ddim_steps = args.ddim
    cfg.num_generate = args.num

    # Choose best.pt (epoch 200, SOTA) or final.pt (epoch 300, slightly overfit)
    ckpt_file = "best.pt" if args.best else "final.pt"
    print(f"=== Winner eval: ckpt={args.ckpt}/{ckpt_file}  ddim={args.ddim}  num={args.num} ===")
    dataset, projector, embed_table, tokenizer = prepare_dataset(cfg)
    ckpt_path = Path(cfg.checkpoint_dir) / ckpt_file
    if not ckpt_path.exists():
        # Fallback to final.pt if best.pt not found
        ckpt_path = Path(cfg.checkpoint_dir) / "final.pt"
        print(f"   [WARNING] best.pt not found, falling back to final.pt")
    diffusion_eval, model_cfg = load_checkpoint(cfg, str(ckpt_path))

    print(f"\n[1/3] Generating {cfg.num_generate} samples...")
    images = generate_images(diffusion_eval, cfg, model_cfg, cfg.num_generate)
    texts = decode_to_text(images, projector, embed_table, tokenizer, cfg,
                           value_ranges=dataset.value_ranges)
    print(f"   -> {len(texts)} decoded")

    # Dump all samples to log in eval_quality-compatible format
    out_log = Path(args.out_log)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    with out_log.open("w") as f:
        for i, t in enumerate(texts):
            f.write(f"--- Sample {i+1} ---\n{t}\n\n")
    print(f"   log -> {out_log}")

    # Free diffusion model from memory before loading Qwen
    del diffusion_eval, images
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ---- Score each sample ----
    print(f"\n[2/3] Building bigram table + loading Qwen2.5-1.5B for PPL...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    qwen_lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    qwen_lm = qwen_lm.to(device).eval()
    bigram_set = build_bigram_table(qwen_tok, cfg.data_paths)
    print(f"   bigrams: {len(bigram_set)}")

    scored = []
    for i, t in enumerate(texts):
        b = bigram_coherence(t, qwen_tok, bigram_set)
        p = lm_perplexity(t, qwen_tok, qwen_lm, device)
        r = real_word_ratio(t, qwen_tok)
        n = structural_noise_ratio(t)
        prose = is_prose(t, tokenizer=qwen_tok)
        c = composite_score(b, p, r)
        scored.append({
            "idx": i, "text": t, "bigram": b, "ppl": p, "real": r,
            "noise": n, "prose": prose, "comp": c,
        })

    # ---- Rank and display ----
    print(f"\n[3/3] Ranking top-{args.topk} by composite score...")
    scored.sort(key=lambda s: s["comp"], reverse=True)

    out_top = Path(args.out_top)
    with out_top.open("w") as f:
        f.write(f"=== Top-{args.topk} winners  (ckpt={args.ckpt}, ddim={args.ddim}) ===\n\n")
        for rank, s in enumerate(scored[:args.topk], 1):
            tag = "PROSE" if s["prose"] else "noise"
            header = (f"#{rank}  [{tag}]  comp={s['comp']:.3f}  "
                      f"bigram={s['bigram']:.3f}  ppl={s['ppl']:.0f}  "
                      f"real={s['real']:.3f}  noise={s['noise']:.2f}\n")
            print(header, end="")
            print(f"  {s['text'][:180]}\n")
            f.write(header)
            f.write(s["text"] + "\n\n")

    # Aggregate stats
    prose_only = [s for s in scored if s["prose"]]
    comps = [s["comp"] for s in scored]
    print(f"\n=== Stats (n={len(scored)}) ===")
    print(f"  Prose count:     {len(prose_only)}/{len(scored)} "
          f"({100 * len(prose_only) / len(scored):.0f}%)")
    print(f"  Composite mean:  {np.mean(comps):.4f}")
    print(f"  Composite max:   {np.max(comps):.4f}")
    print(f"  Composite P90:   {np.percentile(comps, 90):.4f}")
    if prose_only:
        prose_comps = [s["comp"] for s in prose_only]
        print(f"  Prose comp mean: {np.mean(prose_comps):.4f}")
        print(f"  Prose comp max:  {np.max(prose_comps):.4f}")
    print(f"\n   top-{args.topk} -> {out_top}")


if __name__ == "__main__":
    main()
