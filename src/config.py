"""
Token-Image Diffusion PoC — Configuration
==========================================
Proof of concept: encode token embeddings as 2D images,
train an image diffusion model to generate them, decode back to text.

Targeting: Mac Mini M4, 64GB, MPS backend.
"""
from dataclasses import dataclass, field
from typing import Literal
import torch


@dataclass
class Config:
    # --- Model source (just need embeddings + tokenizer) ---
    model_name: str = "Qwen/Qwen2.5-1.5B"

    # --- Sequence / image geometry ---
    seq_len: int = 64                # tokens per sample
    proj_dim: int = 64               # PCA / linear projection target dim
    image_size: int = 64             # final image resolution (seq_len × proj_dim)
    image_channels: int = 1          # 1=grayscale, 3=multi-channel (stacked PCs)

    # --- Encoding method ---
    # "raw"       — normalized projected embedding matrix
    # "gaf"       — Gramian Angular Summation Field
    # "gadf"      — Gramian Angular Difference Field
    # "recurrence" — pairwise cosine-distance recurrence plot
    encoder: Literal["raw", "gaf", "gadf", "recurrence"] = "raw"

    # --- Diffusion ---
    num_timesteps: int = 1000        # DDPM training timesteps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: Literal["linear", "cosine"] = "cosine"

    # --- UNet ---
    unet_dim: int = 64               # base channel width
    unet_dim_mults: tuple = (1, 2, 4)
    unet_attn_resolutions: tuple = (16,)  # apply attention at 16×16
    unet_num_res_blocks: int = 2
    unet_dropout: float = 0.0

    # --- Training ---
    batch_size: int = 64
    lr: float = 1e-4
    num_epochs: int = 50
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    save_every: int = 10
    num_workers: int = 0             # MPS doesn't love multiprocess dataloading

    # --- Dataset ---
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    data_source: str = "wikitext"    # "wikitext", "security", "red", "blue"
    data_paths: tuple = ()           # auto-populated JSONL paths for security sources
    max_samples: int = 50_000        # cap for PoC speed

    # --- VQ-VAE (learned encoder, replaces PCA) ---
    use_vqvae: bool = False
    vqvae_hidden_dim: int = 256
    vqvae_embed_dim: int = 32
    vqvae_num_embeddings: int = 512
    vqvae_epochs: int = 50
    vqvae_lr: float = 3e-4
    vqvae_batch_size: int = 32
    vqvae_beta: float = 0.25              # commitment loss weight
    vqvae_token_loss_weight: float = 0.1   # token classification loss weight

    # --- VQ-GAN (learned encoder with adversarial training, replaces PCA) ---
    use_vqgan: bool = False
    vqgan_hidden_dim: int = 256
    vqgan_latent_channels: int = 16
    vqgan_latent_size: int = 16
    vqgan_codebook_size: int = 1024
    vqgan_codebook_dim: int = 16
    vqgan_epochs: int = 40
    vqgan_lr: float = 3e-4
    vqgan_batch_size: int = 64
    vqgan_commit_weight: float = 0.25
    vqgan_cosine_weight: float = 0.5

    # --- VQ-GAN non-square latent (for 128-token mode) ---
    vqgan_latent_h: int = 16          # latent height (overrides vqgan_latent_size for asymmetric)
    vqgan_latent_w: int = 16          # latent width

    # --- AE variant for ablation comparison ---
    # "vq"           — baseline VQ-GAN (codebook=1024)
    # "kl"           — continuous KL-regularized bottleneck (no quantization)
    # "bigvq"        — VQ with large codebook (8192)
    # "tokence"      — VQ + sampled-softmax token cross-entropy auxiliary loss
    # "tokence_128"  — tokence_big with 128-token / 8×16 grid / 16×32 latent
    ae_variant: Literal["vq", "kl", "bigvq", "tokence", "tokence_big", "tokence_xl", "tokence_xl2", "tokence_big_long", "tokence_big_wide", "tokence_hilbert", "tokence_128"] = "vq"
    vqgan_kl_weight: float = 1e-6
    vqgan_bigvq_codebook_size: int = 8192
    vqgan_tokence_weight: float = 0.5
    vqgan_tokence_negatives: int = 4096
    vqgan_tokence_temp: float = 0.07
    vqgan_hilbert: bool = False

    # --- Diffusion backbone ---
    backbone: Literal["unet", "dit"] = "unet"

    # --- DiT (Diffusion Transformer, used when backbone="dit") ---
    dit_hidden_dim: int = 384
    dit_depth: int = 8
    dit_num_heads: int = 6
    dit_patch_size: int = 2
    dit_mlp_ratio: float = 4.0
    dit_dropout: float = 0.0
    dit_pos_channel: bool = False        # prepend position channel to DiT input
    dit_coherence_loss: bool = False     # add latent neighbor coherence loss
    dit_coherence_weight: float = 0.1    # weight for coherence loss term
    dit_sequence_loss: bool = False      # add 1D sequence-order auxiliary loss (LSTM predictor)
    dit_sequence_weight: float = 0.5     # weight for sequence-order loss term
    dit_sequence_hidden: int = 128       # LSTM hidden dim for sequence predictor
    dit_self_cond: bool = False          # self-conditioning: feed previous x0_pred back as input
    dit_boundary_channel: bool = False   # add token boundary grid channel

    # --- Feature 3: Classifier-Free Guidance ---
    dit_cfg: bool = False                # enable CFG conditioning
    dit_cfg_dropout: float = 0.15        # null-condition dropout rate at training
    dit_cfg_scale: float = 1.0           # guidance scale at inference (1.0 = unconditional)
    dit_cfg_cond_dim: int = 1536         # Qwen d_model for prompt conditioning

    # --- Feature 1: Consistency Distillation ---
    dit_consistency: bool = False        # use consistency model at inference
    dit_consistency_steps: int = 1       # 1, 2, or 4 steps for CM inference

    # --- Normalization ---
    clip_sigma: float = 3.0           # z-score clipping range for global image normalization

    # --- Generation ---
    num_generate: int = 16
    ddim_steps: int = 50             # accelerated sampling steps
    use_ddim: bool = True

    # --- Paths ---
    output_dir: str = "runs"
    checkpoint_dir: str = "runs/checkpoints"

    # --- Device ---
    @property
    def device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
