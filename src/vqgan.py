"""
Token-Image Diffusion — VQ-GAN Encoder/Decoder
================================================
Learned encoder/decoder with adversarial training for high-fidelity
roundtrip of (seq_len, d_model) embedding matrices through a discrete
latent space (latent_channels, latent_size, latent_size).

Replaces PCA for >95% token roundtrip accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────
#  Vector Quantization (EMA codebook, dead-code restart)
# ─────────────────────────────────────────────────────────────────────
class VectorQuantize(nn.Module):
    def __init__(self, num_embeddings: int = 1024, embedding_dim: int = 4,
                 decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z_e: torch.Tensor):
        """z_e: (B, D, H, W). Returns: z_q, commitment_loss, indices."""
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)

        dist = (
            flat.pow(2).sum(1, keepdim=True)
            + self.embed.pow(2).sum(1)
            - 2 * flat @ self.embed.t()
        )
        indices = dist.argmin(dim=1)
        z_q = F.embedding(indices, self.embed)

        if self.training:
            one_hot = F.one_hot(indices, self.num_embeddings).float()
            cs = one_hot.sum(0)
            es = one_hot.t() @ flat

            self.cluster_size.data.mul_(self.decay).add_(cs, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(es, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cs_smooth = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            self.embed.data.copy_(self.embed_avg / cs_smooth.unsqueeze(1))

            dead = cs < 1.0
            if dead.any():
                ri = torch.randint(0, flat.shape[0], (dead.sum(),), device=flat.device)
                self.embed.data[dead] = flat[ri].detach()

        commitment_loss = F.mse_loss(
            z_e.permute(0, 2, 3, 1).reshape(-1, D), z_q.detach()
        )
        z_q = z_e + (z_q.reshape(B, H, W, D).permute(0, 3, 1, 2) - z_e).detach()
        indices = indices.reshape(B, H, W)
        return z_q, commitment_loss, indices

    @property
    def codebook_utilization(self):
        return (self.cluster_size > 1.0).float().mean().item()


# ─────────────────────────────────────────────────────────────────────
#  KL-regularized continuous bottleneck (Stable-Diffusion style)
# ─────────────────────────────────────────────────────────────────────
class KLBottleneck(nn.Module):
    """
    Takes a (B, 2C, H, W) tensor of (mean, logvar), reparameterizes,
    and returns z plus a KL divergence to N(0, I).
    """
    def __init__(self, latent_channels: int):
        super().__init__()
        self.latent_channels = latent_channels

    def forward(self, h: torch.Tensor):
        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        if self.training:
            std = (0.5 * logvar).exp()
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        return z, kl


# ─────────────────────────────────────────────────────────────────────
#  Residual Conv Block
# ─────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


def hilbert_curve_8x8() -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Hilbert curve index mapping for an 8x8 grid.

    Returns:
        fwd: (64,) tensor — fwd[seq_pos] = flat_grid_index
        inv: (64,) tensor — inv[flat_grid_index] = seq_pos

    Adjacent sequence positions map to 4-connected grid neighbors.
    """
    def _hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
        """Convert Hilbert index d to (x, y) in an n×n grid."""
        x = y = 0
        s = 1
        while s < n:
            rx = 1 if (d & 2) else 0
            ry = 1 if ((d & 1) ^ rx) else 0
            # Rotate
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d >>= 2
            s <<= 1
        return x, y

    n = 8
    fwd = torch.zeros(n * n, dtype=torch.long)
    for d in range(n * n):
        x, y = _hilbert_d2xy(n, d)
        fwd[d] = y * n + x  # seq_pos d → flat grid index y*8+x
    inv = torch.zeros(n * n, dtype=torch.long)
    inv[fwd] = torch.arange(n * n)
    return fwd, inv


# ─────────────────────────────────────────────────────────────────────
#  VQ-GAN Encoder / Decoder
# ─────────────────────────────────────────────────────────────────────
class VQGANEncoder(nn.Module):
    """(B, seq_len, d_model) → (B, out_channels, latent_h, latent_w)

    Supports non-square latents for 128-token mode:
      64 tokens  → 8×8  grid → 16×16 latent  (square, legacy)
      128 tokens → 8×16 grid → 16×32 latent  (non-square, latent_h=16 latent_w=32)
    """

    def __init__(self, d_model=1536, seq_len=64, hidden_dim=256,
                 latent_channels=4, latent_size=16,
                 latent_h: int = None, latent_w: int = None,
                 double_out: bool = False, use_hilbert: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        # Resolve latent spatial dimensions
        self.latent_h = latent_h if latent_h is not None else latent_size
        self.latent_w = latent_w if latent_w is not None else latent_size
        self.latent_size = latent_size
        # Token grid: always 8 rows; cols = seq_len // 8
        self.grid_h = 8
        self.grid_w = seq_len // 8          # 8 for 64-token, 16 for 128-token
        self.use_hilbert = use_hilbert
        if use_hilbert:
            fwd, _ = hilbert_curve_8x8()
            self.register_buffer("hilbert_fwd", fwd)
        out_ch = latent_channels * (2 if double_out else 1)

        self.input_proj = nn.Linear(d_model, hidden_dim)
        # After reshape: (B, hidden_dim, grid_h, grid_w)
        # Single Upsample(scale=2): grid_h×grid_w → 2*grid_h × 2*grid_w
        # For 64-token: 8×8 → 16×16; for 128-token: 8×16 → 16×32
        layers = [
            ResBlock(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="nearest"),
        ]
        # Extra upsample only for SQUARE latents that need 32×32
        # (tokence_big_wide variant). Non-square 128-token latent (16×32) skips this.
        if latent_size >= 32 and self.latent_h == self.latent_w:
            layers += [
                ResBlock(hidden_dim, hidden_dim),
                nn.Upsample(scale_factor=2, mode="nearest"),  # 16 → 32
            ]
        layers += [
            ResBlock(hidden_dim, 128),
            ResBlock(128, 64),
            nn.Conv2d(64, out_ch, 1),
        ]
        self.blocks = nn.Sequential(*layers)

    def forward(self, embeddings):
        B = embeddings.shape[0]
        h = self.input_proj(embeddings)                      # (B, seq_len, hidden_dim)
        if self.use_hilbert:
            h = h[:, self.hilbert_fwd, :]
        # Reshape to (B, hidden_dim, grid_h, grid_w) — works for both 8×8 and 8×16
        h = h.permute(0, 2, 1).reshape(B, self.hidden_dim, self.grid_h, self.grid_w)
        return self.blocks(h)                                 # (B, C, latent_h, latent_w)


class VQGANDecoder(nn.Module):
    """(B, latent_ch, latent_h, latent_w) → (B, seq_len, d_model)

    Handles both square (64-token, 16×16) and non-square (128-token, 16×32) latents.
    The stride-2 conv halves both spatial dimensions, so 16×32 → 8×16 → seq=128.
    """

    def __init__(self, d_model=1536, seq_len=64, hidden_dim=256,
                 latent_channels=4, latent_size=16,
                 latent_h: int = None, latent_w: int = None,
                 use_hilbert: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_size = latent_size
        self.latent_h = latent_h if latent_h is not None else latent_size
        self.latent_w = latent_w if latent_w is not None else latent_size
        self.use_hilbert = use_hilbert
        if use_hilbert:
            _, inv = hilbert_curve_8x8()
            self.register_buffer("hilbert_inv", inv)

        layers = [
            nn.Conv2d(latent_channels, 64, 1),
            ResBlock(64, 128),
            ResBlock(128, hidden_dim),
        ]
        # Extra downsample only for SQUARE 32×32 latents (tokence_big_wide).
        # Non-square 16×32 latent (128-token) needs only one stride-2 conv: 16×32 → 8×16.
        if latent_size >= 32 and self.latent_h == self.latent_w:
            layers += [
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
                nn.GroupNorm(min(32, hidden_dim), hidden_dim),
                nn.SiLU(),
                ResBlock(hidden_dim, hidden_dim),
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),  # latent → grid
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.SiLU(),
            ResBlock(hidden_dim, hidden_dim),
        ]
        self.blocks = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, d_model)

    def forward(self, z_q):
        B = z_q.shape[0]
        h = self.blocks(z_q)                                 # (B, hidden_dim, grid_h, grid_w)
        h = h.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)  # (B, seq_len, hidden_dim)
        if self.use_hilbert:
            h = h[:, self.hilbert_inv, :]
        return self.output_proj(h)                            # (B, seq_len, d_model)


# ─────────────────────────────────────────────────────────────────────
#  PatchGAN Discriminator
# ─────────────────────────────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    """Discriminator on projected embedding matrices with spectral normalization."""

    def __init__(self, d_model=1536, proj_dim=64):
        super().__init__()
        self.proj = nn.Linear(d_model, proj_dim, bias=False)
        sn = nn.utils.spectral_norm
        # Input: (B, 1, seq_len, proj_dim) = (B, 1, 64, 64)
        self.net = nn.Sequential(
            sn(nn.Conv2d(1, 32, 4, stride=2, padding=1)),     # → 32×32
            nn.LeakyReLU(0.2),
            sn(nn.Conv2d(32, 64, 4, stride=2, padding=1)),    # → 16×16
            nn.LeakyReLU(0.2),
            sn(nn.Conv2d(64, 128, 4, stride=2, padding=1)),   # → 8×8
            nn.LeakyReLU(0.2),
            sn(nn.Conv2d(128, 1, 4, stride=1, padding=1)),    # → 7×7
        )

    def forward(self, embeddings):
        """embeddings: (B, seq_len, d_model) → patch logits"""
        B = embeddings.shape[0]
        x = self.proj(embeddings)                          # (B, 64, 64)
        x = x.unsqueeze(1)                                 # (B, 1, 64, 64)
        return self.net(x)                                  # (B, 1, 7, 7)


# ─────────────────────────────────────────────────────────────────────
#  Full VQ-GAN Model
# ─────────────────────────────────────────────────────────────────────
class VQGAN(nn.Module):
    def __init__(self, d_model=1536, seq_len=64, hidden_dim=256,
                 latent_channels=4, latent_size=16,
                 latent_h: int = None, latent_w: int = None,
                 codebook_size=1024, codebook_dim=4,
                 bottleneck_type: str = "vq",
                 use_hilbert: bool = False):
        super().__init__()
        self.bottleneck_type = bottleneck_type

        is_kl = (bottleneck_type == "kl")
        self.encoder = VQGANEncoder(d_model, seq_len, hidden_dim,
                                     latent_channels, latent_size,
                                     latent_h=latent_h, latent_w=latent_w,
                                     double_out=is_kl,
                                     use_hilbert=use_hilbert)
        if is_kl:
            self.bottleneck = KLBottleneck(latent_channels)
            self.vq = None
            bn_p = 0
        else:
            self.vq = VectorQuantize(codebook_size, codebook_dim)
            self.bottleneck = None
            bn_p = codebook_size * codebook_dim

        self.decoder = VQGANDecoder(d_model, seq_len, hidden_dim,
                                     latent_channels, latent_size,
                                     latent_h=latent_h, latent_w=latent_w,
                                     use_hilbert=use_hilbert)

        enc_p = sum(p.numel() for p in self.encoder.parameters())
        dec_p = sum(p.numel() for p in self.decoder.parameters())
        bn_label = "KL" if is_kl else "VQ"
        print(f"[VQ-GAN] Encoder: {enc_p/1e6:.2f}M, Decoder: {dec_p/1e6:.2f}M, "
              f"{bn_label}: {bn_p/1e3:.0f}K, Total: {(enc_p+dec_p+bn_p)/1e6:.2f}M "
              f"(bottleneck={bottleneck_type})")

    def encode(self, embeddings):
        """(B, seq_len, d_model) → encoder output (pre-bottleneck for KL,
        pre-quantize for VQ)."""
        return self.encoder(embeddings)

    def quantize(self, z_e):
        """For VQ: z_e → (z_q, commit_loss, indices).
        For KL:  z_e (2C channels) → (z, kl_loss, None)."""
        if self.bottleneck_type == "kl":
            z, kl = self.bottleneck(z_e)
            return z, kl, None
        return self.vq(z_e)

    def decode(self, z_q):
        """z_q (B, C, H, W) → (B, seq_len, d_model)."""
        return self.decoder(z_q)

    def forward(self, embeddings):
        z_e = self.encode(embeddings)
        z_q, aux_loss, indices = self.quantize(z_e)
        recon = self.decode(z_q)
        return recon, aux_loss, indices, z_e


# ─────────────────────────────────────────────────────────────────────
#  VQ-GAN Projector (drop-in interface)
# ─────────────────────────────────────────────────────────────────────
class VQGANProjector:
    """Drop-in replacement for EmbeddingProjector. Uses VQ-GAN latents."""

    def __init__(self, model: VQGAN, device: torch.device,
                 latent_mean: float = 0.0, latent_std: float = 1.0,
                 clip_sigma: float = 3.0):
        self.model = model.eval()
        self.device = device
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.clip_sigma = clip_sigma
        # For interface compat with global norm checks
        self.global_proj_mean = None
        self.global_proj_std = None

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """(seq_len, d_model) → (latent_ch, H, W) latent, normalized.
        For VQ: returns the post-quantize latent (matches diffusion target).
        For KL: returns the deterministic mean (matches SD recipe)."""
        with torch.no_grad():
            x = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
            z_e = self.model.encode(x)
            z_q, _, _ = self.model.quantize(z_e)  # KL → mean (eval mode)
            z = z_q.squeeze(0).cpu().numpy()
        # Normalize to approx [-1, 1]
        z = (z - self.latent_mean) / (self.latent_std * self.clip_sigma)
        z = np.clip(z, -1.0, 1.0)
        return z.astype(np.float32)

    def inverse_project(self, latent: np.ndarray) -> np.ndarray:
        """(latent_ch, H, W) normalized latent → (seq_len, d_model) embeddings."""
        z = latent * (self.latent_std * self.clip_sigma) + self.latent_mean
        with torch.no_grad():
            z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(self.device)
            if self.model.bottleneck_type == "kl":
                # latent is already a sample/mean — decode directly
                recon = self.model.decode(z_t)
            else:
                z_q, _, _ = self.model.quantize(z_t)
                recon = self.model.decode(z_q)
            return recon.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "clip_sigma": self.clip_sigma,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.latent_mean = ckpt["latent_mean"]
        self.latent_std = ckpt["latent_std"]
        self.clip_sigma = ckpt["clip_sigma"]
        self.model.eval()


# ─────────────────────────────────────────────────────────────────────
#  Embedding Dataset (mmap-compatible, avoids full copy to tensor)
# ─────────────────────────────────────────────────────────────────────
class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset that indexes into numpy arrays (including mmap'd) on-the-fly."""

    def __init__(self, embeddings: np.ndarray, token_ids: np.ndarray):
        self.embeddings = embeddings
        self.token_ids = token_ids

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(self.token_ids[idx], dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────────
#  Sampled-softmax token cross-entropy (for tokence variant)
# ─────────────────────────────────────────────────────────────────────
def _sampled_token_ce(recon_flat: torch.Tensor,
                      target_ids: torch.Tensor,
                      embed_table_dev: torch.Tensor,
                      K: int,
                      temperature: float) -> torch.Tensor:
    """
    CLIP-style sampled softmax. Both recon and embedding rows are L2-normalized
    so logits are cosine similarities ∈ [-1, 1] divided by temperature.

    recon_flat:        (BL, D)   reconstructed embeddings
    target_ids:        (BL,)     true token ids
    embed_table_dev:   (V, D)    full embedding table on device
    K:                 number of *shared* random negatives sampled per step
    """
    BL, D = recon_flat.shape
    V = embed_table_dev.shape[0]
    recon_n = F.normalize(recon_flat, dim=-1)
    table_n = F.normalize(embed_table_dev, dim=-1)
    # Positive logits (cosine with true token)
    pos_emb = table_n[target_ids]                                # (BL, D)
    pos_logit = (recon_n * pos_emb).sum(-1, keepdim=True)        # (BL, 1)
    # Shared random negatives across the batch
    neg_ids = torch.randint(0, V, (K,), device=recon_flat.device)
    neg_emb = table_n[neg_ids]                                   # (K, D)
    neg_logits = recon_n @ neg_emb.t()                           # (BL, K)
    logits = torch.cat([pos_logit, neg_logits], dim=1) / temperature
    targets = torch.zeros(BL, dtype=torch.long, device=recon_flat.device)
    return F.cross_entropy(logits, targets)


# ─────────────────────────────────────────────────────────────────────
#  VQ-GAN Training (variant-aware)
# ─────────────────────────────────────────────────────────────────────
def train_vqgan(cfg, all_embeddings: np.ndarray, all_token_ids: np.ndarray,
                embed_table: np.ndarray):
    """
    Train an autoencoder bottleneck on embedding matrices.
    Variant is selected via cfg.ae_variant ∈ {vq, kl, bigvq, tokence}.
    Returns: VQGANProjector wrapping the trained model.
    """
    device = cfg.device
    d_model = embed_table.shape[1]
    variant = getattr(cfg, "ae_variant", "vq")

    # ── Variant-specific model config ──────────────────────────────
    bottleneck_type = "kl" if variant == "kl" else "vq"
    codebook_size = cfg.vqgan_codebook_size
    if variant == "bigvq":
        codebook_size = cfg.vqgan_bigvq_codebook_size
    print(f"[VQ-GAN] AE variant = {variant}  "
          f"(bottleneck={bottleneck_type}, codebook_size={codebook_size})")

    use_hilbert = getattr(cfg, "vqgan_hilbert", False)
    latent_h = getattr(cfg, "vqgan_latent_h", cfg.vqgan_latent_size)
    latent_w = getattr(cfg, "vqgan_latent_w", cfg.vqgan_latent_size)
    model = VQGAN(
        d_model=d_model,
        seq_len=cfg.seq_len,
        hidden_dim=cfg.vqgan_hidden_dim,
        latent_channels=cfg.vqgan_latent_channels,
        latent_size=cfg.vqgan_latent_size,
        latent_h=latent_h,
        latent_w=latent_w,
        codebook_size=codebook_size,
        codebook_dim=cfg.vqgan_codebook_dim,
        bottleneck_type=bottleneck_type,
        use_hilbert=use_hilbert,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.vqgan_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.vqgan_epochs, 1e-5)

    # Pre-compute normalized embedding table for token accuracy (CPU, chunked)
    embed_table_norm = F.normalize(
        torch.tensor(embed_table, dtype=torch.float32), dim=1
    )
    # Full embedding table on device for token-CE loss (only used for tokence)
    embed_table_dev = None
    if variant in ("tokence", "tokence_big", "tokence_xl", "tokence_xl2", "tokence_big_long", "tokence_big_wide", "tokence_hilbert", "tokence_128"):
        embed_table_dev = torch.tensor(embed_table, dtype=torch.float32).to(device)

    dataset = EmbeddingDataset(all_embeddings, all_token_ids)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.vqgan_batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )

    loss_terms = "recon + cosine"
    if bottleneck_type == "kl":
        loss_terms += " + kl"
    else:
        loss_terms += " + commit"
    if variant in ("tokence", "tokence_big", "tokence_xl", "tokence_xl2", "tokence_big_long", "tokence_big_wide", "tokence_hilbert", "tokence_128"):
        loss_terms += " + token_ce"

    print(f"[VQ-GAN] Training: {len(dataset)} samples, {len(loader)} batches/epoch")
    print(f"[VQ-GAN] Epochs: {cfg.vqgan_epochs}, Losses: {loss_terms}")

    cache_path = Path("data_cache")
    cache_path.mkdir(exist_ok=True)
    best_ckpt_name = f"vqgan_best_{variant}.pt"
    best_acc = 0.0

    for epoch in range(1, cfg.vqgan_epochs + 1):
        model.train()

        total_recon = 0.0
        total_aux = 0.0      # commit (vq) or kl (kl)
        total_cosine = 0.0
        total_tokence = 0.0
        correct_tokens = 0
        total_tokens = 0
        n = 0

        pbar = tqdm(loader, desc=f"VQ-GAN {epoch}/{cfg.vqgan_epochs}")
        for batch_idx, (emb_batch, id_batch) in enumerate(pbar):
            emb_batch = emb_batch.to(device)
            id_batch_dev = id_batch.to(device)

            recon, aux_loss, indices, z_e = model(emb_batch)

            # Reconstruction loss (L1 + MSE)
            recon_loss = 0.5 * F.l1_loss(recon, emb_batch) + 0.5 * F.mse_loss(recon, emb_batch)

            # Per-token cosine similarity loss
            cos_sim = F.cosine_similarity(
                recon.reshape(-1, d_model), emb_batch.reshape(-1, d_model), dim=1
            )
            cosine_loss = (1 - cos_sim).mean()

            if bottleneck_type == "kl":
                aux_weight = cfg.vqgan_kl_weight
            else:
                aux_weight = cfg.vqgan_commit_weight

            loss = (recon_loss
                    + aux_weight * aux_loss
                    + cfg.vqgan_cosine_weight * cosine_loss)

            tokence_loss_val = 0.0
            if variant in ("tokence", "tokence_big", "tokence_xl", "tokence_xl2", "tokence_big_long", "tokence_big_wide", "tokence_hilbert", "tokence_128"):
                recon_flat = recon.reshape(-1, d_model)
                target_flat = id_batch_dev.reshape(-1)
                tokence_loss = _sampled_token_ce(
                    recon_flat, target_flat, embed_table_dev,
                    K=cfg.vqgan_tokence_negatives,
                    temperature=cfg.vqgan_tokence_temp,
                )
                loss = loss + cfg.vqgan_tokence_weight * tokence_loss
                tokence_loss_val = tokence_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon += recon_loss.item()
            total_aux += aux_loss.item()
            total_cosine += cosine_loss.item()
            total_tokence += tokence_loss_val
            n += 1

            # Token accuracy tracking (CPU, every 20 steps)
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    sub_r = recon[:4].reshape(-1, d_model).cpu()
                    sub_ids = id_batch[:4].reshape(-1).cpu()
                    sub_norm = F.normalize(sub_r, dim=1)
                    best_sim = torch.full((sub_norm.shape[0],), -1.0)
                    pred_ids = torch.zeros(sub_norm.shape[0], dtype=torch.long)
                    for ci in range(0, embed_table_norm.shape[0], 16384):
                        chunk = embed_table_norm[ci:ci + 16384]
                        sims = sub_norm @ chunk.t()
                        ms, mi = sims.max(dim=1)
                        better = ms > best_sim
                        best_sim[better] = ms[better]
                        pred_ids[better] = mi[better] + ci
                    correct_tokens += (pred_ids == sub_ids).sum().item()
                    total_tokens += sub_ids.numel()

            acc_str = f"{correct_tokens/max(total_tokens,1)*100:.1f}%"
            pbar.set_postfix(recon=f"{recon_loss.item():.4f}", acc=acc_str)

        scheduler.step()
        acc = correct_tokens / max(total_tokens, 1) * 100
        if bottleneck_type == "kl":
            aux_label, cb_util_str = "kl", "n/a"
        else:
            aux_label = "commit"
            cb_util_str = f"{model.vq.codebook_utilization * 100:.0f}%"

        msg = (
            f"[VQ-GAN Epoch {epoch}] recon={total_recon/n:.4f}  "
            f"{aux_label}={total_aux/n:.4f}  cosine={total_cosine/n:.4f}  "
        )
        if variant in ("tokence", "tokence_big", "tokence_xl", "tokence_xl2", "tokence_big_long", "tokence_big_wide", "tokence_hilbert", "tokence_128"):
            msg += f"tokence={total_tokence/n:.4f}  "
        msg += f"acc={acc:.1f}%  cb_util={cb_util_str}"
        print(msg)

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "acc": acc,
                "variant": variant,
            }, cache_path / best_ckpt_name)
            print(f"  → Saved best VQ-GAN [{variant}] (acc={acc:.1f}%)")

    # Load best and compute latent stats
    ckpt = torch.load(cache_path / best_ckpt_name, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[VQ-GAN] Training done. Best token accuracy: {best_acc:.1f}%")
    print("[VQ-GAN] Computing latent normalization stats...")

    # Compute latent mean/std for diffusion normalization (chunked)
    z_sum = 0.0
    z_sq_sum = 0.0
    z_count = 0
    with torch.no_grad():
        for i in range(0, len(all_embeddings), 64):
            batch = torch.tensor(
                np.array(all_embeddings[i:i+64]), dtype=torch.float32
            ).to(device)
            z_e = model.encode(batch)
            z_np = z_e.cpu().numpy()
            z_sum += z_np.sum()
            z_sq_sum += (z_np ** 2).sum()
            z_count += z_np.size
    latent_mean = float(z_sum / z_count)
    latent_std = float(np.sqrt(z_sq_sum / z_count - latent_mean ** 2)) + 1e-8
    print(f"[VQ-GAN] Latent stats: mean={latent_mean:.4f}, std={latent_std:.4f}")

    projector = VQGANProjector(model, device, latent_mean, latent_std, cfg.clip_sigma)
    projector.save(str(cache_path / f"vqgan_projector_{variant}.pt"))

    return projector
