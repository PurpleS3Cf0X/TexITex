"""
Token-Image Diffusion PoC — Training
======================================
Train diffusion model on token-embedding images.
Supports MPS (Apple Silicon), CUDA, and CPU.
"""
import torch
import copy
import time
from pathlib import Path
from tqdm import tqdm

from config import Config
from model import UNet
from dit import DiT
from diffusion import GaussianDiffusion, SequencePredictor
from dataset import prepare_dataset, get_dataloader


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def find_latest_checkpoint(ckpt_dir: Path):
    """Find the most recent epoch checkpoint in a directory."""
    ckpts = sorted(ckpt_dir.glob("ckpt_epoch*.pt"))
    return str(ckpts[-1]) if ckpts else None


def train(cfg: Config, resume_from: str = None):
    device = cfg.device
    print(f"[Train] Device: {device}")

    # ------------------------------------------------------------------
    #  Prepare data
    # ------------------------------------------------------------------
    dataset, projector, embed_table, tokenizer = prepare_dataset(cfg)
    dataloader = get_dataloader(dataset, cfg)
    print(f"[Train] {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Determine image shape from dataset
    sample_img = dataset[0][0]
    img_channels = sample_img.shape[0]
    img_h, img_w = sample_img.shape[1], sample_img.shape[2]
    print(f"[Train] Image: {img_channels}ch × {img_h}×{img_w}")

    # ------------------------------------------------------------------
    #  Build model
    # ------------------------------------------------------------------
    if cfg.backbone == "dit":
        denoise_model = DiT(
            in_channels=img_channels,
            hidden_dim=cfg.dit_hidden_dim,
            depth=cfg.dit_depth,
            num_heads=cfg.dit_num_heads,
            patch_size=cfg.dit_patch_size,
            input_size=max(img_h, img_w),
            input_h=img_h,
            input_w=img_w,
            mlp_ratio=cfg.dit_mlp_ratio,
            dropout=cfg.dit_dropout,
            pos_channel=getattr(cfg, "dit_pos_channel", False),
            boundary_channel=getattr(cfg, "dit_boundary_channel", False),
            self_cond=getattr(cfg, "dit_self_cond", False),
            use_cfg=getattr(cfg, "dit_cfg", False),
            cfg_cond_dim=getattr(cfg, "dit_cfg_cond_dim", 1536),
        ).to(device)
    else:
        denoise_model = UNet(
            in_channels=img_channels,
            base_dim=cfg.unet_dim,
            dim_mults=cfg.unet_dim_mults,
            attn_resolutions=cfg.unet_attn_resolutions,
            num_res_blocks=cfg.unet_num_res_blocks,
            dropout=cfg.unet_dropout,
            image_size=max(img_h, img_w),
        ).to(device)

    diffusion = GaussianDiffusion(
        model=denoise_model,
        timesteps=cfg.num_timesteps,
        beta_schedule=cfg.beta_schedule,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    ).to(device)

    # Build sequence predictor for sequence-order loss
    seq_predictor = None
    if getattr(cfg, "dit_sequence_loss", False):
        # feat_dim = latent_channels * patch_size^2 (each 2x2 patch flattened)
        feat_dim = img_channels * (cfg.dit_patch_size ** 2)
        seq_predictor = SequencePredictor(
            feat_dim=feat_dim,
            hidden_dim=getattr(cfg, "dit_sequence_hidden", 128),
            num_layers=2,
        ).to(device)
        total_seq = sum(p.numel() for p in seq_predictor.parameters())
        print(f"[Train] Sequence predictor: {total_seq/1e3:.1f}K params (feat_dim={feat_dim})")

    # Optimizer includes both denoise model and sequence predictor (if present)
    opt_params = list(denoise_model.parameters())
    if seq_predictor is not None:
        opt_params += list(seq_predictor.parameters())
    optimizer = torch.optim.AdamW(opt_params, lr=cfg.lr, weight_decay=1e-4)
    ema = EMA(denoise_model, decay=cfg.ema_decay)

    # Build position map for pos_channel mode
    pos_map = None
    if cfg.backbone == "dit" and getattr(cfg, "dit_pos_channel", False):
        from dit import make_position_map
        pos_map = make_position_map(
            latent_h=img_h, latent_w=img_w,
            grid_h=img_h // cfg.dit_patch_size,
            grid_w=img_w // cfg.dit_patch_size,
        ).to(device)
        print(f"[Train] Position channel enabled: {pos_map.shape}")

    # Build boundary map for boundary_channel mode
    boundary_map = None
    if cfg.backbone == "dit" and getattr(cfg, "dit_boundary_channel", False):
        from dit import make_boundary_map
        boundary_map = make_boundary_map(
            latent_h=img_h, latent_w=img_w,
            patch_size=cfg.dit_patch_size,
        ).to(device)
        print(f"[Train] Boundary channel enabled: {boundary_map.shape}")

    # ------------------------------------------------------------------
    #  Resume from checkpoint if requested
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")

    if resume_from == "auto":
        resume_from = find_latest_checkpoint(ckpt_dir)

    if resume_from and Path(resume_from).exists():
        print(f"[Train] Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        denoise_model.load_state_dict(ckpt["model_state"])
        if "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("loss", float("inf"))
        if seq_predictor is not None and "seq_predictor_state" in ckpt:
            seq_predictor.load_state_dict(ckpt["seq_predictor_state"])
            print(f"[Train] Loaded sequence predictor state")
        print(f"[Train] Resumed at epoch {start_epoch}, step {global_step}, loss {best_loss:.4f}")
        del ckpt
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        denoise_model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.num_epochs}")
        for batch in pbar:
            images = batch[0].to(device)  # (B, C, H, W)

            use_self_cond = getattr(cfg, "dit_self_cond", False)

            if use_self_cond or boundary_map is not None:
                # New unified path: self-conditioning + boundary + optional seq loss
                loss = diffusion.training_loss_self_cond(
                    images,
                    pos_map=pos_map,
                    boundary_map=boundary_map,
                    use_self_cond=use_self_cond,
                    seq_predictor=seq_predictor,
                    seq_weight=getattr(cfg, "dit_sequence_weight", 0.5) if seq_predictor else 0.0,
                )
            elif pos_map is not None and seq_predictor is not None:
                loss = diffusion.training_loss_with_sequence_order(
                    images, pos_map, seq_predictor,
                    seq_weight=getattr(cfg, "dit_sequence_weight", 0.5),
                    coherence_weight=cfg.dit_coherence_weight if getattr(cfg, "dit_coherence_loss", False) else 0.0,
                )
            elif pos_map is not None and getattr(cfg, "dit_coherence_loss", False):
                loss = diffusion.training_loss_with_coherence(
                    images, pos_map, coherence_weight=cfg.dit_coherence_weight)
            elif pos_map is not None:
                loss = diffusion.training_loss_with_pos_channel(images, pos_map)
            else:
                loss = diffusion.training_loss(images)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoise_model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(denoise_model)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{epoch_loss/n_batches:.4f}")

        elapsed = time.time() - t_start
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}  time={elapsed:.1f}s  "
              f"steps={global_step}")

        # Save checkpoint
        if epoch % cfg.save_every == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
            save_path = ckpt_dir / f"ckpt_epoch{epoch:04d}.pt"
            ckpt_data = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": denoise_model.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
                "config": {
                    "encoder": cfg.encoder,
                    "backbone": cfg.backbone,
                    "seq_len": cfg.seq_len,
                    "proj_dim": cfg.proj_dim,
                    "img_channels": img_channels,
                    "img_h": img_h,
                    "img_w": img_w,
                    "dit_hidden_dim": cfg.dit_hidden_dim,
                    "dit_depth": cfg.dit_depth,
                    "dit_num_heads": cfg.dit_num_heads,
                    "dit_patch_size": cfg.dit_patch_size,
                    "dit_pos_channel": getattr(cfg, "dit_pos_channel", False),
                    "dit_boundary_channel": getattr(cfg, "dit_boundary_channel", False),
                    "dit_self_cond": getattr(cfg, "dit_self_cond", False),
                    "dit_cfg": getattr(cfg, "dit_cfg", False),
                    "dit_cfg_cond_dim": getattr(cfg, "dit_cfg_cond_dim", 1536),
                },
            }
            if seq_predictor is not None:
                ckpt_data["seq_predictor_state"] = seq_predictor.state_dict()
            torch.save(ckpt_data, save_path)
            print(f"  → Saved {save_path}")

    # Save final
    final_path = ckpt_dir / "final.pt"
    final_data = {
        "epoch": cfg.num_epochs,
        "global_step": global_step,
        "model_state": denoise_model.state_dict(),
        "ema_state": ema.state_dict(),
        "loss": avg_loss,
        "config": {
            "encoder": cfg.encoder,
            "backbone": cfg.backbone,
            "seq_len": cfg.seq_len,
            "proj_dim": cfg.proj_dim,
            "img_channels": img_channels,
            "img_h": img_h,
            "img_w": img_w,
            "dit_hidden_dim": cfg.dit_hidden_dim,
            "dit_depth": cfg.dit_depth,
            "dit_num_heads": cfg.dit_num_heads,
            "dit_patch_size": cfg.dit_patch_size,
            "dit_pos_channel": getattr(cfg, "dit_pos_channel", False),
            "dit_boundary_channel": getattr(cfg, "dit_boundary_channel", False),
            "dit_self_cond": getattr(cfg, "dit_self_cond", False),
        },
    }
    if seq_predictor is not None:
        final_data["seq_predictor_state"] = seq_predictor.state_dict()
    torch.save(final_data, final_path)
    print(f"[Train] Done. Final checkpoint: {final_path}")

    return diffusion, ema, projector, embed_table, tokenizer


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
