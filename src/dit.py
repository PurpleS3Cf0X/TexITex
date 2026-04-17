"""
Token-Image Diffusion — DiT (Diffusion Transformer)
=====================================================
Transformer backbone for diffusion on VQ-GAN latent space.
Replaces UNet for better scaling. MPS-compatible.

Architecture (default ~15M params):
  Input:  (B, 4, 16, 16) VQ-GAN latents
  Patch:  2×2 → 64 tokens of dim 384
  Blocks: 8 DiT blocks with adaLN-Zero conditioning
  Output: (B, 4, 16, 16) predicted noise

Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def make_position_map(latent_size: int = 16, grid_size: int = 8,
                      latent_h: int = None, latent_w: int = None,
                      grid_h: int = None, grid_w: int = None) -> torch.Tensor:
    """Create a (1, 1, latent_h, latent_w) position map.

    Each patch_size×patch_size block gets a value 0→1 in row-major order.
    Supports both square (64-token, 16×16) and non-square (128-token, 16×32) latents.
    """
    lh = latent_h if latent_h is not None else latent_size
    lw = latent_w if latent_w is not None else latent_size
    gh = grid_h if grid_h is not None else grid_size
    gw = grid_w if grid_w is not None else grid_size
    num_tokens = gh * gw
    scale_h = lh // gh
    scale_w = lw // gw
    positions = torch.arange(num_tokens, dtype=torch.float32)
    positions = positions / max(num_tokens - 1, 1)          # normalize to [0, 1]
    grid = positions.reshape(1, 1, gh, gw)
    pos_map = grid.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
    return pos_map  # (1, 1, latent_h, latent_w)


def make_boundary_map(latent_size: int = 16, patch_size: int = 2,
                      latent_h: int = None, latent_w: int = None) -> torch.Tensor:
    """Create a (1, 1, latent_h, latent_w) token boundary map.

    Marks the borders of each patch_size×patch_size token block with 1.0.
    Supports both square and non-square latents.
    """
    lh = latent_h if latent_h is not None else latent_size
    lw = latent_w if latent_w is not None else latent_size
    bmap = torch.zeros(1, 1, lh, lw)
    for i in range(0, lh, patch_size):
        bmap[0, 0, i, :] = 1.0
    for j in range(0, lw, patch_size):
        bmap[0, 0, :, j] = 1.0
    return bmap


# ─────────────────────────────────────────────────────────────────────
#  Timestep Embedding
# ─────────────────────────────────────────────────────────────────────
class TimestepEmbedding(nn.Module):
    """Sinusoidal + MLP timestep embedding."""

    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps → (B, hidden_dim)."""
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


# ─────────────────────────────────────────────────────────────────────
#  Patch Embed / Unpatch
# ─────────────────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """Convert (B, C, H, W) → (B, num_patches, hidden_dim) via conv."""

    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, hidden_dim, H/p, W/p) → (B, N, hidden_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class Unpatch(nn.Module):
    """Convert (B, num_patches, patch_dim) → (B, C, H, W).

    Supports both square (grid_h == grid_w) and non-square grids.
    """

    def __init__(self, out_channels: int, hidden_dim: int, patch_size: int = 2,
                 grid_size: int = 8, grid_h: int = None, grid_w: int = None):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.grid_h = grid_h if grid_h is not None else grid_size
        self.grid_w = grid_w if grid_w is not None else grid_size
        self.proj = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, hidden_dim)
        B = x.shape[0]
        x = self.proj(x)  # (B, N, C*p*p)
        p = self.patch_size
        C = self.out_channels
        Gh = self.grid_h
        Gw = self.grid_w
        # (B, Gh, Gw, C, p, p) → (B, C, Gh*p, Gw*p)
        x = x.reshape(B, Gh, Gw, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, Gh * p, Gw * p)
        return x


# ─────────────────────────────────────────────────────────────────────
#  adaLN-Zero Modulation
# ─────────────────────────────────────────────────────────────────────
class AdaLNModulation(nn.Module):
    """Produce (shift, scale, gate) triples for adaLN-Zero from conditioning."""

    def __init__(self, hidden_dim: int, num_modulations: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * num_modulations),
        )
        self.num_modulations = num_modulations
        self.hidden_dim = hidden_dim

    def forward(self, c: torch.Tensor):
        """c: (B, hidden_dim) → tuple of num_modulations tensors, each (B, 1, hidden_dim)."""
        out = self.net(c)  # (B, hidden_dim * num_modulations)
        out = out.reshape(-1, self.num_modulations, self.hidden_dim)
        return out.unbind(dim=1)  # tuple of (B, hidden_dim)


# ─────────────────────────────────────────────────────────────────────
#  DiT Block (adaLN-Zero)
# ─────────────────────────────────────────────────────────────────────
class DiTBlock(nn.Module):
    """
    Transformer block with adaLN-Zero conditioning.

    adaLN-Zero: LayerNorm params are modulated by timestep embedding,
    and each residual branch has a learnable gate initialized to zero.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        # 6 modulations: shift1, scale1, gate1, shift2, scale2, gate2
        self.adaLN = AdaLNModulation(hidden_dim, num_modulations=6)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D) token sequence
        c: (B, D) conditioning (timestep embedding)
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c)

        # Self-attention with adaLN
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * h

        # FFN with adaLN
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h

        return x


# ─────────────────────────────────────────────────────────────────────
#  Classifier-Free Guidance Condition Projector
# ─────────────────────────────────────────────────────────────────────
class ConditionProjector(nn.Module):
    """Project raw LM prompt embeddings (mean-pooled) into the DiT conditioning space.

    Input:  (B, cond_dim)  — mean-pooled token embedding from the LM
    Output: (B, hidden_dim) — additive bias added to the timestep conditioning c
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)  # (B, hidden_dim)


# ─────────────────────────────────────────────────────────────────────
#  Full DiT Model
# ─────────────────────────────────────────────────────────────────────
class DiT(nn.Module):
    """
    Diffusion Transformer for VQ-GAN latent space.

    Parameters:
        in_channels:  latent channels from VQ-GAN (default 4)
        hidden_dim:   transformer hidden dimension (default 384)
        depth:        number of DiT blocks (default 8)
        num_heads:    attention heads (default 6)
        patch_size:   spatial patch size (default 2)
        input_size:   spatial size of latent — used for square grids (default 16)
        input_h:      latent height — overrides input_size for non-square (e.g. 16)
        input_w:      latent width  — overrides input_size for non-square (e.g. 32)
        mlp_ratio:    MLP expansion ratio (default 4.0)
        dropout:      dropout rate (default 0.0)
        pos_channel:      prepend per-pixel position channel (0→1, row-major)
        boundary_channel: prepend token boundary grid channel
        self_cond:        self-conditioning (feed previous x0_pred as extra channels)
        use_cfg:          enable classifier-free guidance conditioning
        cfg_cond_dim:     input dim of prompt embeddings for CFG (default 1536)
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        patch_size: int = 2,
        input_size: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size

        # Support non-square latents (Feature 2: 128-token mode → 16×32 latent)
        input_h = kwargs.get("input_h", None)
        input_w = kwargs.get("input_w", None)
        self.input_h = input_h if input_h is not None else input_size
        self.input_w = input_w if input_w is not None else input_size
        self.input_size = input_size  # keep for compat

        grid_h = self.input_h // patch_size
        grid_w = self.input_w // patch_size
        num_patches = grid_h * grid_w

        # Extra input channels beyond the latent:
        #   pos_channel:      +1  (token position 0→1)
        #   boundary_channel: +1  (token border grid)
        #   self_cond:        +in_channels  (previous x0 prediction)
        # PatchEmbed sees all input channels; Unpatch only outputs in_channels.
        self.pos_channel = kwargs.get("pos_channel", False)
        self.boundary_channel = kwargs.get("boundary_channel", False)
        self.self_cond = kwargs.get("self_cond", False)
        patch_in = in_channels
        if self.pos_channel:
            patch_in += 1
        if self.boundary_channel:
            patch_in += 1
        if self.self_cond:
            patch_in += in_channels  # x0_pred has same channels as latent
        self.patch_embed = PatchEmbed(patch_in, hidden_dim, patch_size)

        # Positional embedding (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(hidden_dim)

        # Classifier-Free Guidance (Feature 3)
        self.use_cfg = kwargs.get("use_cfg", False)
        if self.use_cfg:
            cfg_cond_dim = kwargs.get("cfg_cond_dim", 1536)
            self.cond_proj = ConditionProjector(cfg_cond_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer: adaLN + linear projection to patch pixels
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = AdaLNModulation(hidden_dim, num_modulations=2)  # shift, scale
        self.unpatch = Unpatch(in_channels, hidden_dim, patch_size,
                               grid_h=grid_h, grid_w=grid_w)

        # Initialize output projection to zero (adaLN-Zero)
        nn.init.zeros_(self.unpatch.proj.weight)
        nn.init.zeros_(self.unpatch.proj.bias)

        # Initialize adaLN gates to zero
        for block in self.blocks:
            nn.init.zeros_(block.adaLN.net[-1].weight)
            nn.init.zeros_(block.adaLN.net[-1].bias)
        nn.init.zeros_(self.final_adaLN.net[-1].weight)
        nn.init.zeros_(self.final_adaLN.net[-1].bias)

        total = sum(p.numel() for p in self.parameters())
        print(f"[DiT] {total / 1e6:.1f}M parameters "
              f"(depth={depth}, dim={hidden_dim}, heads={num_heads}, "
              f"patches={num_patches} [{grid_h}×{grid_w}], patch_size={patch_size})")

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:          (B, C, H, W) noisy latent
        t:          (B,) integer timesteps
        cond_embed: (B, hidden_dim) optional CFG condition (None = unconditional)
        Returns:    (B, C, H, W) predicted noise
        """
        # Timestep conditioning
        c = self.time_embed(t)  # (B, D)

        # Add CFG condition if provided
        if self.use_cfg and cond_embed is not None:
            c = c + self.cond_proj(cond_embed)

        # Patchify + add positional embedding
        h = self.patch_embed(x) + self.pos_embed  # (B, N, D)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, c)

        # Final norm + modulation + unpatch
        shift, scale = self.final_adaLN(c)
        h = self.final_norm(h)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return self.unpatch(h)  # (B, C, H, W)
