"""
Token-Image Diffusion PoC — Generation & Evaluation
=====================================================
Generate images via diffusion, decode back to tokens, evaluate roundtrip.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import Config
from model import UNet
from dit import DiT
from diffusion import GaussianDiffusion
from encoders import EmbeddingProjector, decode_image_to_tokens, encode
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from dataset import prepare_dataset


class _PosChannelWrapper(torch.nn.Module):
    """Wraps a DiT model to prepend a fixed position map to its input."""
    def __init__(self, model, pos_map):
        super().__init__()
        self.model = model
        self.register_buffer("pos_map", pos_map)

    def forward(self, x, t):
        B = x.shape[0]
        pos = self.pos_map.expand(B, -1, -1, -1)
        return self.model(torch.cat([pos, x], dim=1), t)


class _AuxChannelWrapper(torch.nn.Module):
    """Wraps a DiT model to prepend pos + boundary + self_cond channels.

    Used for non-self-cond inference (self_cond passed as zeros).
    For actual self-conditioning, use ddim_sample_self_cond with aux_builder.
    """
    def __init__(self, model, pos_map=None, boundary_map=None, use_self_cond=False, latent_channels=16):
        super().__init__()
        self.model = model
        self.use_self_cond = use_self_cond
        self.latent_channels = latent_channels
        if pos_map is not None:
            self.register_buffer("pos_map", pos_map)
        else:
            self.pos_map = None
        if boundary_map is not None:
            self.register_buffer("boundary_map", boundary_map)
        else:
            self.boundary_map = None

    def forward(self, x, t):
        """x: (B, C, H, W) noisy latent. Prepend aux channels."""
        B = x.shape[0]
        parts = []
        if self.pos_map is not None:
            parts.append(self.pos_map.expand(B, -1, -1, -1))
        if self.boundary_map is not None:
            parts.append(self.boundary_map.expand(B, -1, -1, -1))
        if self.use_self_cond:
            # When called directly (not via ddim_sample_self_cond), pass zeros
            parts.append(torch.zeros(B, self.latent_channels, x.shape[2], x.shape[3], device=x.device))
        if parts:
            x = torch.cat(parts + [x], dim=1)
        return self.model(x, t)


def _get_global_norm_params(projector, cfg):
    """Return (mean, std, clip_sigma) if projector has global stats, else None."""
    if hasattr(projector, 'global_proj_mean') and projector.global_proj_mean is not None:
        return (projector.global_proj_mean, projector.global_proj_std, cfg.clip_sigma)
    return None


def load_checkpoint(
    cfg: Config,
    checkpoint_path: str,
) -> tuple:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]
    backbone = model_cfg.get("backbone", "unet")

    if backbone == "dit":
        img_h = model_cfg["img_h"]
        img_w = model_cfg["img_w"]
        denoise_model = DiT(
            in_channels=model_cfg["img_channels"],
            hidden_dim=model_cfg.get("dit_hidden_dim", cfg.dit_hidden_dim),
            depth=model_cfg.get("dit_depth", cfg.dit_depth),
            num_heads=model_cfg.get("dit_num_heads", cfg.dit_num_heads),
            patch_size=model_cfg.get("dit_patch_size", cfg.dit_patch_size),
            input_size=max(img_h, img_w),
            input_h=img_h,
            input_w=img_w,
            mlp_ratio=cfg.dit_mlp_ratio,
            dropout=0.0,
            pos_channel=model_cfg.get("dit_pos_channel", getattr(cfg, "dit_pos_channel", False)),
            boundary_channel=model_cfg.get("dit_boundary_channel", getattr(cfg, "dit_boundary_channel", False)),
            self_cond=model_cfg.get("dit_self_cond", getattr(cfg, "dit_self_cond", False)),
            use_cfg=model_cfg.get("dit_cfg", getattr(cfg, "dit_cfg", False)),
            cfg_cond_dim=model_cfg.get("dit_cfg_cond_dim", getattr(cfg, "dit_cfg_cond_dim", 1536)),
        )
    else:
        denoise_model = UNet(
            in_channels=model_cfg["img_channels"],
            base_dim=cfg.unet_dim,
            dim_mults=cfg.unet_dim_mults,
            attn_resolutions=cfg.unet_attn_resolutions,
            num_res_blocks=cfg.unet_num_res_blocks,
            image_size=max(model_cfg["img_h"], model_cfg["img_w"]),
        )

    # Load EMA weights if available, else model weights
    if "ema_state" in ckpt:
        denoise_model.load_state_dict(ckpt["ema_state"])
        print(f"[Generate] Loaded EMA weights (backbone={backbone})")
    else:
        denoise_model.load_state_dict(ckpt["model_state"])

    denoise_model.eval()

    diffusion = GaussianDiffusion(
        model=denoise_model,
        timesteps=cfg.num_timesteps,
        beta_schedule=cfg.beta_schedule,
    )

    return diffusion, model_cfg


def generate_images(
    diffusion: GaussianDiffusion,
    cfg: Config,
    model_cfg: dict,
    num_samples: int = 16,
) -> np.ndarray:
    """Generate images via DDIM sampling."""
    device = cfg.device
    diffusion = diffusion.to(device)

    use_self_cond = model_cfg.get("dit_self_cond", getattr(cfg, "dit_self_cond", False))
    use_boundary = model_cfg.get("dit_boundary_channel", getattr(cfg, "dit_boundary_channel", False))
    use_pos = model_cfg.get("dit_pos_channel", getattr(cfg, "dit_pos_channel", False))
    latent_channels = model_cfg["img_channels"]

    # Build aux maps
    pos_map = None
    boundary_map = None
    img_h = model_cfg["img_h"]
    img_w = model_cfg["img_w"]
    patch_size = model_cfg.get("dit_patch_size", cfg.dit_patch_size)
    if use_pos:
        from dit import make_position_map
        pos_map = make_position_map(
            latent_h=img_h, latent_w=img_w,
            grid_h=img_h // patch_size,
            grid_w=img_w // patch_size,
        ).to(device)
    if use_boundary:
        from dit import make_boundary_map
        boundary_map = make_boundary_map(
            latent_h=img_h, latent_w=img_w,
            patch_size=patch_size,
        ).to(device)

    shape = (
        num_samples,
        latent_channels,
        model_cfg["img_h"],
        model_cfg["img_w"],
    )

    extras = []
    if use_pos:
        extras.append("pos")
    if use_boundary:
        extras.append("boundary")
    if use_self_cond:
        extras.append("self_cond")
    print(f"[Generate] Sampling {num_samples} images, shape={shape[1:]}, "
          f"DDIM steps={cfg.ddim_steps}, extras={extras or 'none'}")

    with torch.no_grad():
        if use_self_cond:
            # Self-conditioning DDIM: build aux_builder that creates full input
            raw_model = diffusion.model  # un-wrapped DiT

            def aux_builder(x_noisy, self_cond_x0):
                B = x_noisy.shape[0]
                parts = []
                if pos_map is not None:
                    parts.append(pos_map.expand(B, -1, -1, -1))
                if boundary_map is not None:
                    parts.append(boundary_map.expand(B, -1, -1, -1))
                parts.append(self_cond_x0)  # self-conditioning channels
                parts.append(x_noisy)       # noisy latent
                return torch.cat(parts, dim=1)

            images = diffusion.ddim_sample_self_cond(
                shape=shape,
                device=device,
                num_steps=cfg.ddim_steps,
                aux_builder=aux_builder,
            )
        else:
            # Legacy path: wrap model with pos/boundary channels
            if use_pos or use_boundary:
                if use_boundary:
                    diffusion.model = _AuxChannelWrapper(
                        diffusion.model, pos_map, boundary_map,
                        use_self_cond=False, latent_channels=latent_channels,
                    )
                else:
                    diffusion.model = _PosChannelWrapper(diffusion.model, pos_map)

            images = diffusion.sample(
                shape=shape,
                device=device,
                use_ddim=cfg.use_ddim,
                ddim_steps=cfg.ddim_steps,
            )

    return images.cpu().numpy()


def decode_to_text(
    images: np.ndarray,
    projector,
    embed_table: np.ndarray,
    tokenizer,
    cfg: Config,
    value_ranges: np.ndarray = None,
) -> list:
    """Decode generated images back to text strings."""
    texts = []
    for i in range(len(images)):
        img = images[i]  # (C, H, W)

        if cfg.use_vqgan:
            # VQ-GAN path: latent → decode → embeddings → nearest tokens
            recovered_emb = projector.inverse_project(img)  # (seq_len, d_model)
            sims = cos_sim(recovered_emb, embed_table)
            token_ids = sims.argmax(axis=1)
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            texts.append(text)
        elif cfg.use_vqvae:
            # VQ-VAE path: image → embeddings → nearest tokens
            recovered_emb = projector.inverse_project(img)  # (seq_len, d_model)
            sims = cos_sim(recovered_emb, embed_table)
            token_ids = sims.argmax(axis=1)
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            texts.append(text)
        elif cfg.encoder == "raw":
            # PCA raw path — prefer global normalization
            gnp = _get_global_norm_params(projector, cfg)
            if gnp is not None:
                token_ids = decode_image_to_tokens(
                    img, projector, embed_table,
                    method="raw",
                    global_norm_params=gnp,
                )
            else:
                # Backward compat: per-sample mean ranges
                if value_ranges is not None:
                    vmin, vmax = value_ranges[:, 0].mean(), value_ranges[:, 1].mean()
                else:
                    vmin, vmax = -3.0, 3.0
                token_ids = decode_image_to_tokens(
                    img, projector, embed_table,
                    method="raw",
                    value_range=(vmin, vmax),
                )
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            texts.append(text)
        else:
            texts.append("[decode not implemented for this encoder]")

    return texts


def evaluate_roundtrip(
    dataset,
    projector,
    embed_table: np.ndarray,
    tokenizer,
    cfg: Config,
    num_samples: int = 100,
) -> dict:
    """
    Evaluate encoding roundtrip fidelity:
    original tokens → embed → project → encode → decode → nearest tokens
    Measures how much information survives the image encoding.
    """
    print(f"[Eval] Roundtrip test on {num_samples} samples...")

    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Roundtrip eval"):
        img_tensor, token_ids_tensor, vrange_tensor = dataset[idx]
        img = img_tensor.numpy()
        original_ids = token_ids_tensor.numpy()

        if cfg.use_vqgan:
            # VQ-GAN roundtrip: latent → decode → embeddings → nearest tokens
            recovered_emb = projector.inverse_project(img)
            sims = cos_sim(recovered_emb, embed_table)
            recovered_ids = sims.argmax(axis=1)
        elif cfg.use_vqvae:
            # VQ-VAE roundtrip: image → embeddings → nearest tokens
            recovered_emb = projector.inverse_project(img)  # (seq_len, d_model)
            sims = cos_sim(recovered_emb, embed_table)
            recovered_ids = sims.argmax(axis=1)
        elif cfg.encoder == "raw":
            gnp = _get_global_norm_params(projector, cfg)
            if gnp is not None:
                recovered_ids = decode_image_to_tokens(
                    img, projector, embed_table,
                    method="raw",
                    global_norm_params=gnp,
                )
            else:
                vmin, vmax = vrange_tensor.numpy()
                recovered_ids = decode_image_to_tokens(
                    img, projector, embed_table,
                    method="raw",
                    value_range=(vmin, vmax),
                )
        else:
            continue

        matches = (recovered_ids == original_ids)
        correct_tokens += matches.sum()
        total_tokens += len(original_ids)
        if matches.all():
            correct_sequences += 1

    token_accuracy = correct_tokens / max(total_tokens, 1)
    seq_accuracy = correct_sequences / max(len(indices), 1)

    encoder_name = "VQ-GAN" if cfg.use_vqgan else ("VQ-VAE" if cfg.use_vqvae else "PCA")
    print(f"[Eval] Roundtrip token accuracy: {token_accuracy:.1%}")
    print(f"[Eval] Roundtrip sequence accuracy: {seq_accuracy:.1%}")
    print(f"[Eval] (This measures {encoder_name} information retention, not diffusion quality)")

    return {"token_accuracy": token_accuracy, "sequence_accuracy": seq_accuracy}


def visualize_samples(
    images: np.ndarray,
    texts: list = None,
    save_path: str = "runs/generated_samples.png",
    title: str = "Generated Token-Embedding Images",
):
    """Plot a grid of generated images with optional decoded text."""
    n = min(len(images), 16)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(title, fontsize=14)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        if i < n:
            img = images[i]
            if img.shape[0] == 1:
                ax.imshow(img[0], cmap="viridis", vmin=-1, vmax=1)
            else:
                # Multi-channel: show first 3 as RGB
                rgb = np.transpose(img[:3], (1, 2, 0))
                rgb = (rgb + 1) / 2  # to [0, 1]
                ax.imshow(rgb)

            if texts and i < len(texts):
                # Truncate text for display
                display_text = texts[i][:60] + "..." if len(texts[i]) > 60 else texts[i]
                # Disable matplotlib mathtext parsing — security text contains
                # `$`, `{`, `}`, `\` etc. that would otherwise be parsed as LaTeX.
                ax.set_title(display_text, fontsize=7, wrap=True, parse_math=False)
        ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved to {save_path}")


def visualize_training_data(dataset, tokenizer, cfg: Config, num_samples: int = 8):
    """Visualize some training images alongside their original text."""
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    images = []
    texts = []
    for idx in indices:
        img, token_ids, _ = dataset[idx]
        images.append(img.numpy())
        text = tokenizer.decode(token_ids.numpy(), skip_special_tokens=True)
        texts.append(text)

    visualize_samples(
        np.stack(images),
        texts,
        save_path="runs/training_samples.png",
        title=f"Training Data ({cfg.encoder} encoding)",
    )


def main():
    cfg = Config()
    device = cfg.device

    # Load dataset and projector
    dataset, projector, embed_table, tokenizer = prepare_dataset(cfg)

    # Visualize training data
    visualize_training_data(dataset, tokenizer, cfg)

    # Evaluate encoding roundtrip (independent of diffusion model)
    roundtrip_metrics = evaluate_roundtrip(
        dataset, projector, embed_table, tokenizer, cfg
    )

    # Load trained model and generate
    ckpt_path = Path(cfg.checkpoint_dir) / "final.pt"
    if not ckpt_path.exists():
        print(f"[Generate] No checkpoint found at {ckpt_path}")
        print("[Generate] Run train.py first, then re-run this script.")
        return

    diffusion, model_cfg = load_checkpoint(cfg, str(ckpt_path))

    # Generate
    images = generate_images(diffusion, cfg, model_cfg, num_samples=cfg.num_generate)

    # Decode to text
    texts = decode_to_text(
        images, projector, embed_table, tokenizer, cfg,
        value_ranges=dataset.value_ranges,
    )

    # Print generated texts
    print("\n" + "=" * 60)
    print("GENERATED TEXT SAMPLES")
    print("=" * 60)
    for i, text in enumerate(texts):
        print(f"\n[Sample {i+1}]")
        print(text[:200])

    # Visualize
    visualize_samples(
        images, texts,
        save_path="runs/generated_samples.png",
        title=f"Diffusion-Generated ({cfg.encoder}, DDIM {cfg.ddim_steps} steps)",
    )


if __name__ == "__main__":
    main()
