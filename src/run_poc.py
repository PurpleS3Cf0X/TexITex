#!/usr/bin/env python3
"""
Token-Image Diffusion PoC — Full Pipeline
==========================================
Run the complete proof of concept:

  1. Load tokenizer + embedding weights
  2. Tokenize text corpus → fixed-length chunks
  3. PCA-project embeddings → encode as 2D images
  4. Train image diffusion model (UNet + DDPM)
  5. Generate images via DDIM sampling
  6. Decode images → nearest tokens → text
  7. Evaluate roundtrip fidelity

Usage:
    python run_poc.py                          # defaults (raw encoding)
    python run_poc.py --encoder gaf            # Gramian Angular Field
    python run_poc.py --encoder recurrence     # cosine recurrence plot
    python run_poc.py --num_epochs 20 --lr 2e-4
    python run_poc.py --eval_only              # skip training, just generate
"""
import argparse
import sys
import time

from config import Config
from train import train
from generate import (
    load_checkpoint,
    generate_images,
    decode_to_text,
    evaluate_roundtrip,
    visualize_samples,
    visualize_training_data,
)
from dataset import prepare_dataset
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Token-Image Diffusion PoC")

    # Encoding
    p.add_argument("--encoder", type=str, default="raw",
                   choices=["raw", "gaf", "gadf", "recurrence"],
                   help="Image encoding method")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--proj_dim", type=int, default=None,
                   help="PCA dimensions (default: seq_len * image_channels)")
    p.add_argument("--image_channels", type=int, default=1,
                   help="Image channels (1=grayscale, 3=RGB multi-channel PCA)")

    # Model
    p.add_argument("--unet_dim", type=int, default=64)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")

    # Training
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_samples", type=int, default=50000)

    # Generation
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--num_generate", type=int, default=16)

    # Data source
    p.add_argument("--data_source", type=str, default="wikitext",
                   choices=["wikitext", "security", "red", "blue"],
                   help="Data source: wikitext (default), security (red+blue), red, blue")

    # VQ-VAE
    p.add_argument("--use_vqvae", action="store_true",
                   help="Use learned VQ-VAE encoder instead of PCA")
    p.add_argument("--vqvae_epochs", type=int, default=50,
                   help="VQ-VAE pretraining epochs")

    # VQ-GAN
    p.add_argument("--use_vqgan", action="store_true",
                   help="Use VQ-GAN encoder (learned, with adversarial training)")
    p.add_argument("--vqgan_epochs", type=int, default=80,
                   help="VQ-GAN pretraining epochs")
    p.add_argument("--ae_variant", type=str, default="vq",
                   choices=["vq", "kl", "bigvq", "tokence", "tokence_big", "tokence_xl", "tokence_xl2", "tokence_big_long", "tokence_big_wide", "tokence_hilbert", "tokence_128"],
                   help="AE bottleneck variant for ablation: "
                        "vq=baseline, kl=continuous KL, "
                        "bigvq=8192 codes, tokence=VQ + token cross-entropy, "
                        "tokence_big=tokence with hidden_dim=512 (3× capacity), "
                        "tokence_xl=tokence with hidden_dim=768 (7× capacity), "
                        "tokence_xl2=rebalanced xl: hidden=768, cb=4096, tce_w=2.0, "
                        "tokence_big_long=tokence_big trained for more epochs (separate ckpt), "
                        "tokence_big_wide=tokence_big with latent_size=32 (4× bottleneck capacity), "
                        "tokence_128=128-token mode: 8×16 grid → 16×32 non-square latent")

    # Backbone
    p.add_argument("--backbone", type=str, default="unet",
                   choices=["unet", "dit"],
                   help="Diffusion backbone: unet (default) or dit (transformer)")
    p.add_argument("--dit_pos_channel", action="store_true",
                   help="Prepend a position map channel to DiT input (Experiment B1)")
    p.add_argument("--dit_coherence_loss", action="store_true",
                   help="Add latent neighbor coherence auxiliary loss (Phase 2)")
    p.add_argument("--dit_coherence_weight", type=float, default=0.1,
                   help="Weight for coherence loss term (default: 0.1)")
    p.add_argument("--dit_sequence_loss", action="store_true",
                   help="Add 1D sequence-order auxiliary loss (causal LSTM predictor)")
    p.add_argument("--dit_sequence_weight", type=float, default=0.5,
                   help="Weight for sequence-order loss (default: 0.5)")
    p.add_argument("--dit_sequence_hidden", type=int, default=128,
                   help="LSTM hidden dim for sequence predictor (default: 128)")
    p.add_argument("--dit_self_cond", action="store_true",
                   help="Enable self-conditioning: feed previous x0_pred back as input")
    p.add_argument("--dit_boundary_channel", action="store_true",
                   help="Add token boundary grid channel to DiT input")

    # Feature 3: Classifier-Free Guidance
    p.add_argument("--dit_cfg", action="store_true",
                   help="Enable classifier-free guidance conditioning")
    p.add_argument("--dit_cfg_dropout", type=float, default=0.15,
                   help="Null-condition dropout rate during CFG training (default: 0.15)")
    p.add_argument("--dit_cfg_scale", type=float, default=1.0,
                   help="CFG guidance scale at inference (1.0=uncond, >1.0=guided)")
    p.add_argument("--dit_cfg_cond_dim", type=int, default=1536,
                   help="Prompt embedding dim for CFG (default: 1536 for Qwen2.5-1.5B)")

    # Feature 1: Consistency Distillation
    p.add_argument("--dit_consistency", action="store_true",
                   help="Use consistency model at inference (requires prior distillation)")
    p.add_argument("--dit_consistency_steps", type=int, default=1,
                   help="Number of inference steps for consistency model (1/2/4)")

    p.add_argument("--dit_depth", type=int, default=None,
                   help="DiT depth (number of transformer blocks, default: 8)")
    p.add_argument("--dit_hidden_dim", type=int, default=None,
                   help="DiT hidden dimension (default: 384)")
    p.add_argument("--dit_num_heads", type=int, default=None,
                   help="DiT attention heads (default: 6)")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Custom checkpoint directory (default: runs/checkpoints)")

    # Resume
    p.add_argument("--resume", type=str, default=None, const="auto", nargs="?",
                   help="Resume training from checkpoint (default: auto-find latest)")

    # Modes
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training, load checkpoint and generate")
    p.add_argument("--roundtrip_only", action="store_true",
                   help="Only test encoding roundtrip fidelity (no diffusion)")

    return p.parse_args()


def main():
    args = parse_args()

    # Build config from args
    cfg = Config()
    cfg.encoder = args.encoder
    cfg.seq_len = args.seq_len
    cfg.image_channels = args.image_channels
    # Auto-compute proj_dim: seq_len * image_channels for square multi-channel images
    cfg.proj_dim = args.proj_dim if args.proj_dim is not None else args.seq_len * args.image_channels
    cfg.unet_dim = args.unet_dim
    cfg.model_name = args.model_name
    cfg.num_epochs = args.num_epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.max_samples = args.max_samples
    cfg.ddim_steps = args.ddim_steps
    cfg.num_generate = args.num_generate
    cfg.use_vqvae = args.use_vqvae
    cfg.vqvae_epochs = args.vqvae_epochs
    cfg.use_vqgan = args.use_vqgan
    cfg.vqgan_epochs = args.vqgan_epochs
    cfg.ae_variant = args.ae_variant
    # tokence_big = tokence loss with 2× wider encoder/decoder
    if cfg.ae_variant == "tokence_big":
        cfg.vqgan_hidden_dim = 512
    elif cfg.ae_variant == "tokence_xl":
        cfg.vqgan_hidden_dim = 768
    elif cfg.ae_variant == "tokence_xl2":
        cfg.vqgan_hidden_dim = 768
        cfg.vqgan_codebook_size = 4096
        cfg.vqgan_tokence_weight = 2.0
    elif cfg.ae_variant == "tokence_big_long":
        # Same as tokence_big, just trained longer (separate checkpoint tag)
        cfg.vqgan_hidden_dim = 512
    elif cfg.ae_variant == "tokence_big_wide":
        # tokence_big architecture with 4× bottleneck capacity (16×16 → 32×32 latent)
        cfg.vqgan_hidden_dim = 512
        cfg.vqgan_latent_size = 32
    elif cfg.ae_variant == "tokence_hilbert":
        cfg.vqgan_hidden_dim = 512
        cfg.vqgan_hilbert = True
    elif cfg.ae_variant == "tokence_128":
        # 128-token mode: seq_len=128, 8×16 token grid, 16×32 non-square latent
        cfg.vqgan_hidden_dim = 512
        cfg.seq_len = 128
        cfg.vqgan_latent_h = 16
        cfg.vqgan_latent_w = 32
    cfg.backbone = args.backbone
    cfg.dit_pos_channel = args.dit_pos_channel
    cfg.dit_coherence_loss = args.dit_coherence_loss
    cfg.dit_coherence_weight = args.dit_coherence_weight
    cfg.dit_sequence_loss = args.dit_sequence_loss
    cfg.dit_sequence_weight = args.dit_sequence_weight
    cfg.dit_sequence_hidden = args.dit_sequence_hidden
    cfg.dit_self_cond = args.dit_self_cond
    cfg.dit_boundary_channel = args.dit_boundary_channel
    # Feature 3: CFG
    cfg.dit_cfg = args.dit_cfg
    cfg.dit_cfg_dropout = args.dit_cfg_dropout
    cfg.dit_cfg_scale = args.dit_cfg_scale
    cfg.dit_cfg_cond_dim = args.dit_cfg_cond_dim
    # Feature 1: Consistency
    cfg.dit_consistency = args.dit_consistency
    cfg.dit_consistency_steps = args.dit_consistency_steps
    if args.dit_depth is not None:
        cfg.dit_depth = args.dit_depth
    if args.dit_hidden_dim is not None:
        cfg.dit_hidden_dim = args.dit_hidden_dim
    if args.dit_num_heads is not None:
        cfg.dit_num_heads = args.dit_num_heads
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir

    # Data source setup
    cfg.data_source = args.data_source
    if cfg.data_source in ("security", "red", "blue"):
        red_path = "/Volumes/R3D_1/redteam_project/data/final_v2/train.jsonl"
        blue_path = "/Volumes/R3D_1/blueteam_project/data/final/train.jsonl"
        if cfg.data_source == "security":
            cfg.data_paths = (red_path, blue_path)
        elif cfg.data_source == "red":
            cfg.data_paths = (red_path,)
        elif cfg.data_source == "blue":
            cfg.data_paths = (blue_path,)

    print("=" * 60)
    print("TOKEN-IMAGE DIFFUSION — Proof of Concept")
    print("=" * 60)
    print(f"  Data source: {cfg.data_source}")
    enc_name = "VQ-GAN" if cfg.use_vqgan else ("VQ-VAE" if cfg.use_vqvae else cfg.encoder)
    if cfg.use_vqgan:
        enc_name = f"VQ-GAN [{cfg.ae_variant}]"
    print(f"  Encoder:     {enc_name}")
    print(f"  Backbone:    {cfg.backbone}")
    print(f"  Seq length:  {cfg.seq_len}")
    print(f"  Proj dim:    {cfg.proj_dim}")
    print(f"  Image size:  ~{cfg.seq_len}×{cfg.proj_dim}")
    print(f"  Model:       {cfg.model_name}")
    if cfg.backbone == "unet":
        print(f"  UNet base:   {cfg.unet_dim}")
    else:
        print(f"  DiT:         depth={cfg.dit_depth}, dim={cfg.dit_hidden_dim}, heads={cfg.dit_num_heads}")
    print(f"  Device:      {cfg.device}")
    print(f"  Epochs:      {cfg.num_epochs}")
    print(f"  Batch size:  {cfg.batch_size}")
    print("=" * 60)

    # ---- Roundtrip-only mode ----
    if args.roundtrip_only:
        print("\n[Mode] Roundtrip evaluation only (no diffusion training)")
        dataset, projector, embed_table, tokenizer = prepare_dataset(cfg)
        visualize_training_data(dataset, tokenizer, cfg)
        evaluate_roundtrip(dataset, projector, embed_table, tokenizer, cfg,
                          num_samples=500)
        return

    # ---- Eval-only mode ----
    if args.eval_only:
        print("\n[Mode] Generation only (loading checkpoint)")
        dataset, projector, embed_table, tokenizer = prepare_dataset(cfg)
        ckpt_path = Path(cfg.checkpoint_dir) / "final.pt"
        if not ckpt_path.exists():
            print(f"ERROR: No checkpoint at {ckpt_path}. Train first.")
            sys.exit(1)
        diffusion, model_cfg = load_checkpoint(cfg, str(ckpt_path))
        images = generate_images(diffusion, cfg, model_cfg, cfg.num_generate)
        texts = decode_to_text(images, projector, embed_table, tokenizer, cfg,
                              value_ranges=dataset.value_ranges)
        _print_results(texts, images)
        return

    # ---- Full pipeline ----
    t_total = time.time()

    # Phase 1: Data preparation
    print("\n" + "=" * 40)
    print("PHASE 1: Data Preparation")
    print("=" * 40)
    dataset, projector, embed_table, tokenizer = prepare_dataset(cfg)
    visualize_training_data(dataset, tokenizer, cfg)

    # Phase 2: Roundtrip evaluation
    print("\n" + "=" * 40)
    print("PHASE 2: Encoding Roundtrip Evaluation")
    print("=" * 40)
    roundtrip = evaluate_roundtrip(
        dataset, projector, embed_table, tokenizer, cfg, num_samples=200
    )

    # Phase 3: Training
    print("\n" + "=" * 40)
    print("PHASE 3: Diffusion Training")
    print("=" * 40)
    diffusion, ema, projector, embed_table, tokenizer = train(cfg, resume_from=args.resume)

    # Phase 4: Generation
    print("\n" + "=" * 40)
    print("PHASE 4: Generation & Decoding")
    print("=" * 40)
    ckpt_path = Path(cfg.checkpoint_dir) / "final.pt"
    diffusion_eval, model_cfg = load_checkpoint(cfg, str(ckpt_path))
    images = generate_images(diffusion_eval, cfg, model_cfg, cfg.num_generate)
    texts = decode_to_text(
        images, projector, embed_table, tokenizer, cfg,
        value_ranges=dataset.value_ranges,
    )

    _print_results(texts, images)

    elapsed = time.time() - t_total
    print(f"\n[Done] Total time: {elapsed / 60:.1f} minutes")


def _print_results(texts, images):
    print("\n" + "=" * 60)
    print("GENERATED TEXT SAMPLES")
    print("=" * 60)
    for i, text in enumerate(texts):
        print(f"\n--- Sample {i+1} ---")
        print(text[:300])

    visualize_samples(
        images, texts,
        save_path="runs/generated_samples.png",
    )


if __name__ == "__main__":
    main()
