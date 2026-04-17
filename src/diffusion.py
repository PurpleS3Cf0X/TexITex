"""
Token-Image Diffusion PoC — Diffusion Scheduler
=================================================
DDPM training + DDIM accelerated sampling. MPS-compatible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SequencePredictor(nn.Module):
    """Lightweight causal LSTM that predicts next-token latent features.

    Given a sequence of 64 token features (from the 8×8 latent grid in
    row-major reading order), predicts each token's features from its
    predecessors.  Used as an auxiliary loss to teach the diffusion model
    that adjacent tokens in reading order should have coherent sequential
    relationships — the key missing ingredient for sentence structure.

    Input:  (B, 64, feat_dim)  — per-token features from x0_pred
    Output: (B, 63, feat_dim)  — predicted features for tokens 1..63
    """

    def __init__(self, feat_dim: int = 16, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 64, feat_dim) → (B, 63, feat_dim) predictions for next tokens."""
        # Feed tokens 0..62 to predict tokens 1..63
        h, _ = self.lstm(x[:, :-1, :])  # (B, 63, hidden_dim)
        return self.proj(h)  # (B, 63, feat_dim)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal)."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
    """
    DDPM forward process + DDIM reverse sampling.
    """
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Compute schedule
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Register as buffers (move to device with model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # DDPM posterior
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract values from schedule tensor at timestep t, broadcast to x_shape."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # ------------------------------------------------------------------
    #  Forward process (training)
    # ------------------------------------------------------------------

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Add noise to x_start at timestep t. Returns noisy x_t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def training_loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """Compute simple MSE loss (predict noise)."""
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        noise_pred = self.model(x_noisy, t)
        return torch.nn.functional.mse_loss(noise_pred, noise)

    def training_loss_with_pos_channel(self, x_start: torch.Tensor, pos_map: torch.Tensor) -> torch.Tensor:
        """Training loss where pos_map channel is prepended but not noised.

        x_start: (B, C, H, W) latent images (no position channel)
        pos_map: (1, 1, H, W) position map (broadcast over batch)
        """
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        # Prepend clean position channel to noisy latent
        pos_expanded = pos_map.expand(B, -1, -1, -1).to(x_start.device)
        x_input = torch.cat([pos_expanded, x_noisy], dim=1)  # (B, C+1, H, W)
        noise_pred = self.model(x_input, t)  # model outputs (B, C, H, W)
        return torch.nn.functional.mse_loss(noise_pred, noise)

    def _neighbor_coherence_loss(self, x0_pred: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        """Compute latent neighbor coherence loss.

        For each pair of horizontally/vertically adjacent 2×2 blocks (tokens),
        compare cosine similarity in predicted x0 vs ground truth x_start.

        x0_pred, x_start: (B, C, 16, 16) latent images
        Returns: scalar loss
        """
        # Extract per-token features by averaging 2×2 blocks
        # (B, C, 16, 16) → (B, C, 8, 2, 8, 2) → mean over 2×2 → (B, C, 8, 8)
        def to_patch_features(x):
            B, C, H, W = x.shape
            grid = H // 2  # 8
            x = x.reshape(B, C, grid, 2, grid, 2)
            return x.mean(dim=(3, 5))  # (B, C, 8, 8)

        pred_feat = to_patch_features(x0_pred)   # (B, C, 8, 8)
        true_feat = to_patch_features(x_start)   # (B, C, 8, 8)

        # L2-normalize along channel dim for cosine similarity
        pred_norm = torch.nn.functional.normalize(pred_feat, dim=1)  # (B, C, 8, 8)
        true_norm = torch.nn.functional.normalize(true_feat, dim=1)  # (B, C, 8, 8)

        # Horizontal neighbor cosine similarities (7 pairs per row, 8 rows = 56)
        pred_h = (pred_norm[:, :, :, :-1] * pred_norm[:, :, :, 1:]).sum(dim=1)  # (B, 8, 7)
        true_h = (true_norm[:, :, :, :-1] * true_norm[:, :, :, 1:]).sum(dim=1)  # (B, 8, 7)

        # Vertical neighbor cosine similarities (7 pairs per col, 8 cols = 56)
        pred_v = (pred_norm[:, :, :-1, :] * pred_norm[:, :, 1:, :]).sum(dim=1)  # (B, 7, 8)
        true_v = (true_norm[:, :, :-1, :] * true_norm[:, :, 1:, :]).sum(dim=1)  # (B, 7, 8)

        # MSE between predicted and true neighbor similarities
        loss_h = torch.nn.functional.mse_loss(pred_h, true_h)
        loss_v = torch.nn.functional.mse_loss(pred_v, true_v)

        return (loss_h + loss_v) / 2

    def training_loss_with_coherence(self, x_start: torch.Tensor, pos_map: torch.Tensor,
                                      coherence_weight: float = 0.1) -> torch.Tensor:
        """Training loss with pos_channel AND neighbor coherence auxiliary loss.

        x_start: (B, C, H, W) latent images
        pos_map: (1, 1, H, W) position map
        coherence_weight: weight for the coherence loss term
        """
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        # Prepend clean position channel to noisy latent
        pos_expanded = pos_map.expand(B, -1, -1, -1).to(x_start.device)
        x_input = torch.cat([pos_expanded, x_noisy], dim=1)  # (B, C+1, H, W)
        noise_pred = self.model(x_input, t)  # model outputs (B, C, H, W)

        # Primary MSE loss
        mse_loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Coherence loss: recover predicted x0 and compare neighbor relationships
        x0_pred = self.predict_x0_from_noise(x_noisy, t, noise_pred)
        x0_pred = torch.clamp(x0_pred, -1, 1)
        coherence_loss = self._neighbor_coherence_loss(x0_pred, x_start)

        return mse_loss + coherence_weight * coherence_loss

    @staticmethod
    def _extract_token_features(x: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
        """Extract per-token features from a latent image.

        x: (B, C, H, W) — latent image (e.g. 4×16×16)
        Returns: (B, num_tokens, feat_dim) where num_tokens = (H/p)*(W/p)
                 and feat_dim = C * p * p (flatten each patch)

        For 4×16×16 with patch_size=2: (B, 64, 16)
        """
        B, C, H, W = x.shape
        grid_h, grid_w = H // patch_size, W // patch_size
        # (B, C, grid_h, p, grid_w, p)
        x = x.reshape(B, C, grid_h, patch_size, grid_w, patch_size)
        # (B, grid_h, grid_w, C, p, p) → (B, grid_h*grid_w, C*p*p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, grid_h * grid_w, -1)
        return x

    def training_loss_with_sequence_order(
        self,
        x_start: torch.Tensor,
        pos_map: torch.Tensor,
        seq_predictor: 'SequencePredictor',
        seq_weight: float = 0.5,
        coherence_weight: float = 0.0,
    ) -> torch.Tensor:
        """Training loss with pos_channel + 1D sequence-order auxiliary loss.

        The sequence-order loss works by:
        1. Predicting x0 from the noisy sample (as in coherence loss)
        2. Extracting per-token features from x0_pred (flatten 2×2 patches → 16-dim)
        3. Arranging tokens in row-major (reading) order: 64 tokens = 8×8 grid
        4. Running a causal LSTM predictor: given token N, predict token N+1
        5. Comparing LSTM predictions against ground-truth token N+1 features
        6. Backprop through x0_pred → teaches DiT that sequential neighbors matter

        x_start: (B, C, H, W) latent images
        pos_map: (1, 1, H, W) position map
        seq_predictor: SequencePredictor module (trained jointly)
        seq_weight: weight for the sequence-order loss
        coherence_weight: optional coherence loss weight (0 to disable)
        """
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        # Prepend clean position channel
        pos_expanded = pos_map.expand(B, -1, -1, -1).to(x_start.device)
        x_input = torch.cat([pos_expanded, x_noisy], dim=1)
        noise_pred = self.model(x_input, t)

        # Primary MSE loss
        mse_loss = F.mse_loss(noise_pred, noise)

        # Predict x0 for auxiliary losses
        x0_pred = self.predict_x0_from_noise(x_noisy, t, noise_pred)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        # Sequence-order loss: extract token features, run causal predictor
        # patch_size=2 to match DiT patches: each 2×2 block = 1 token
        pred_tokens = self._extract_token_features(x0_pred, patch_size=2)  # (B, 64, 16)
        true_tokens = self._extract_token_features(x_start, patch_size=2)  # (B, 64, 16)

        # LSTM predicts next token from current; compare against ground truth
        next_pred = seq_predictor(pred_tokens)  # (B, 63, 16)
        next_true = true_tokens[:, 1:, :]       # (B, 63, 16) — tokens 1..63

        # L1 + cosine loss for sequence ordering
        seq_l1 = F.l1_loss(next_pred, next_true)
        # Cosine: encourage similar directions in feature space
        cos_sim = F.cosine_similarity(
            next_pred.reshape(-1, next_pred.shape[-1]),
            next_true.reshape(-1, next_true.shape[-1]),
            dim=1,
        )
        seq_cos = (1 - cos_sim).mean()
        seq_loss = 0.5 * seq_l1 + 0.5 * seq_cos

        total = mse_loss + seq_weight * seq_loss

        # Optional coherence loss
        if coherence_weight > 0:
            coherence_loss = self._neighbor_coherence_loss(x0_pred, x_start)
            total = total + coherence_weight * coherence_loss

        return total

    def _build_aux_channels(self, x_noisy, pos_map, boundary_map, self_cond_x0):
        """Build the auxiliary channel tensor to prepend to x_noisy.

        Returns: (B, extra_ch, H, W) or None if no aux channels.
        """
        B = x_noisy.shape[0]
        parts = []
        if pos_map is not None:
            parts.append(pos_map.expand(B, -1, -1, -1).to(x_noisy.device))
        if boundary_map is not None:
            parts.append(boundary_map.expand(B, -1, -1, -1).to(x_noisy.device))
        if self_cond_x0 is not None:
            parts.append(self_cond_x0)
        if not parts:
            return None
        return torch.cat(parts, dim=1)

    def training_loss_self_cond(
        self,
        x_start: torch.Tensor,
        pos_map: torch.Tensor = None,
        boundary_map: torch.Tensor = None,
        use_self_cond: bool = True,
        seq_predictor: 'SequencePredictor' = None,
        seq_weight: float = 0.5,
    ) -> torch.Tensor:
        """Training loss with self-conditioning, boundary channel, and optional seq loss.

        Self-conditioning: 50% of training steps, run the model once to get x0_pred,
        then feed x0_pred as extra input channels for a second pass. The other 50%,
        pass zeros. This teaches the model to refine its own predictions — at inference,
        each DDIM step feeds the previous step's x0_pred back, giving the model
        iterative refinement capability.

        x_start:       (B, C, H, W) clean latent images
        pos_map:       (1, 1, H, W) position channel or None
        boundary_map:  (1, 1, H, W) boundary channel or None
        use_self_cond: whether to use self-conditioning (50% dropout)
        seq_predictor: optional SequencePredictor for sequence-order loss
        seq_weight:    weight for sequence-order loss
        """
        B, C, H, W = x_start.shape
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        # Self-conditioning: 50% of the time, do a first pass to get x0_pred
        if use_self_cond and torch.rand(1).item() > 0.5:
            with torch.no_grad():
                # First pass with zero self-cond input
                zero_cond = torch.zeros_like(x_start)
                aux = self._build_aux_channels(x_noisy, pos_map, boundary_map, zero_cond)
                x_input = torch.cat([aux, x_noisy], dim=1) if aux is not None else x_noisy
                noise_pred_first = self.model(x_input, t)
                x0_first = self.predict_x0_from_noise(x_noisy, t, noise_pred_first)
                x0_first = torch.clamp(x0_first, -1, 1)
            self_cond_input = x0_first.detach()
        else:
            # No self-conditioning — pass zeros
            self_cond_input = torch.zeros_like(x_start) if use_self_cond else None

        # Second pass (or only pass if self_cond=False)
        aux = self._build_aux_channels(x_noisy, pos_map, boundary_map, self_cond_input)
        x_input = torch.cat([aux, x_noisy], dim=1) if aux is not None else x_noisy
        noise_pred = self.model(x_input, t)

        # Primary MSE loss
        mse_loss = F.mse_loss(noise_pred, noise)

        # Optional sequence-order loss
        if seq_predictor is not None:
            x0_pred = self.predict_x0_from_noise(x_noisy, t, noise_pred)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            pred_tokens = self._extract_token_features(x0_pred, patch_size=2)
            true_tokens = self._extract_token_features(x_start, patch_size=2)
            next_pred = seq_predictor(pred_tokens)
            next_true = true_tokens[:, 1:, :]
            seq_l1 = F.l1_loss(next_pred, next_true)
            cos_sim = F.cosine_similarity(
                next_pred.reshape(-1, next_pred.shape[-1]),
                next_true.reshape(-1, next_true.shape[-1]),
                dim=1,
            )
            seq_cos = (1 - cos_sim).mean()
            seq_loss = 0.5 * seq_l1 + 0.5 * seq_cos
            mse_loss = mse_loss + seq_weight * seq_loss

        return mse_loss

    # ------------------------------------------------------------------
    #  Reverse process — DDPM sampling
    # ------------------------------------------------------------------

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    @torch.no_grad()
    def ddpm_sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Full DDPM reverse sampling (slow, all timesteps)."""
        x = torch.randn(shape, device=device)
        for t_val in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            noise_pred = self.model(x, t)
            x0_pred = self.predict_x0_from_noise(x, t, noise_pred)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            if t_val > 0:
                mean = (
                    self._extract(self.posterior_mean_coef1, t, x.shape) * x0_pred
                    + self._extract(self.posterior_mean_coef2, t, x.shape) * x
                )
                var = self._extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = x0_pred
        return x

    # ------------------------------------------------------------------
    #  Reverse process — DDIM sampling (accelerated)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple,
        device: torch.device,
        num_steps: int = 50,
        eta: float = 0.0,  # 0 = deterministic DDIM
    ) -> torch.Tensor:
        """DDIM accelerated sampling with `num_steps` denoising steps."""
        # Build sub-sequence of timesteps
        step_size = self.timesteps // num_steps
        timesteps = list(range(0, self.timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)

        for i in range(len(timesteps)):
            t_cur = timesteps[i]
            t_batch = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)

            noise_pred = self.model(x, t_batch)
            x0_pred = self.predict_x0_from_noise(x, t_batch, noise_pred)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_cur = self.alphas_cumprod[t_cur]
                alpha_next = self.alphas_cumprod[t_next]

                sigma = eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next)
                )

                # Direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_next - sigma ** 2) * noise_pred
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

                x = torch.sqrt(alpha_next) * x0_pred + dir_xt + sigma * noise
            else:
                x = x0_pred

        return x

    @torch.no_grad()
    def ddim_sample_self_cond(
        self,
        shape: tuple,
        device: torch.device,
        num_steps: int = 50,
        eta: float = 0.0,
        aux_builder=None,
    ) -> torch.Tensor:
        """DDIM sampling with self-conditioning: each step feeds previous x0_pred.

        aux_builder: callable(x_noisy, self_cond_x0) → full model input tensor.
                     Handles prepending pos/boundary/self_cond channels.
        """
        step_size = self.timesteps // num_steps
        timesteps = list(range(0, self.timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)
        # Start with zero self-conditioning (no prior prediction)
        x0_prev = torch.zeros(shape, device=device)

        for i in range(len(timesteps)):
            t_cur = timesteps[i]
            t_batch = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)

            # Build input with self-conditioning from previous step's x0_pred
            if aux_builder is not None:
                x_input = aux_builder(x, x0_prev)
            else:
                x_input = x

            noise_pred = self.model(x_input, t_batch)
            x0_pred = self.predict_x0_from_noise(x, t_batch, noise_pred)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            # Save for next step's self-conditioning
            x0_prev = x0_pred

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_cur = self.alphas_cumprod[t_cur]
                alpha_next = self.alphas_cumprod[t_next]

                sigma = eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next)
                )

                dir_xt = torch.sqrt(1 - alpha_next - sigma ** 2) * noise_pred
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

                x = torch.sqrt(alpha_next) * x0_pred + dir_xt + sigma * noise
            else:
                x = x0_pred

        return x

    @torch.no_grad()
    def ddim_sample_cfg(
        self,
        shape: tuple,
        device: torch.device,
        num_steps: int = 200,
        eta: float = 0.0,
        aux_builder=None,
        cond_embed: torch.Tensor = None,
        guidance_scale: float = 3.0,
    ) -> torch.Tensor:
        """DDIM sampling with Classifier-Free Guidance + optional self-conditioning.

        Runs two forward passes per step — one conditional (with cond_embed) and one
        unconditional (cond_embed=None) — and blends them:
            noise_pred = uncond + guidance_scale * (cond - uncond)

        aux_builder:   callable(x_noisy, self_cond_x0) → full model input tensor
                       (handles pos/boundary/self_cond channels); or None.
        cond_embed:    (B, cond_dim) prompt embedding. If None, runs unconditional.
        guidance_scale: 1.0 = no guidance, >1.0 = stronger conditioning.
        """
        step_size = self.timesteps // num_steps
        timesteps = list(range(0, self.timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)
        x0_prev = torch.zeros(shape, device=device)

        for i in range(len(timesteps)):
            t_cur = timesteps[i]
            t_batch = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)

            # Build spatial input (pos/boundary/self_cond channels)
            if aux_builder is not None:
                x_input = aux_builder(x, x0_prev)
            else:
                x_input = x

            # Conditional forward pass
            if cond_embed is not None and guidance_scale != 1.0:
                noise_cond = self.model(x_input, t_batch, cond_embed=cond_embed)
                # Unconditional forward pass (no condition)
                noise_uncond = self.model(x_input, t_batch, cond_embed=None)
                # CFG blend
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.model(x_input, t_batch,
                                        cond_embed=cond_embed if cond_embed is not None else None)

            x0_pred = self.predict_x0_from_noise(x, t_batch, noise_pred)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            x0_prev = x0_pred

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_cur = self.alphas_cumprod[t_cur]
                alpha_next = self.alphas_cumprod[t_next]

                sigma = eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next)
                )
                dir_xt = torch.sqrt(1 - alpha_next - sigma ** 2) * noise_pred
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
                x = torch.sqrt(alpha_next) * x0_pred + dir_xt + sigma * noise
            else:
                x = x0_pred

        return x

    @torch.no_grad()
    def ddim_sample_consistency(
        self,
        shape: tuple,
        device: torch.device,
        num_steps: int = 1,
        aux_builder=None,
        cond_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        """Multi-step consistency model inference.

        A consistency model is trained so that f(x_t, t) ≈ x_0 for any t,
        making single-step generation possible.  With num_steps=1, it runs
        one forward pass from pure noise; with num_steps>1 it interleaves
        re-noising and denoising for slightly higher quality.

        num_steps: 1 (fastest), 2, or 4.
        """
        x = torch.randn(shape, device=device)

        # For multi-step: compute intermediate timesteps to re-noise to
        # Use evenly spaced intervals in [0, T-1]
        if num_steps == 1:
            t_vals = [self.timesteps - 1]
        elif num_steps == 2:
            t_vals = [self.timesteps - 1, self.timesteps // 2]
        else:  # 4 or more
            step = self.timesteps // num_steps
            t_vals = list(range(self.timesteps - 1, 0, -step))[:num_steps]

        x0_prev = torch.zeros(shape, device=device)

        for step_idx, t_val in enumerate(t_vals):
            if step_idx > 0:
                # Re-noise x0_prev to t_val using q_sample
                t_batch_renoise = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
                x = self.q_sample(x0_prev, t_batch_renoise)

            t_batch = torch.full((shape[0],), t_val, device=device, dtype=torch.long)

            # Build spatial input
            if aux_builder is not None:
                x_input = aux_builder(x, x0_prev)
            else:
                x_input = x

            # Consistency model: one forward pass → x0_pred
            noise_pred = self.model(x_input, t_batch,
                                    **({"cond_embed": cond_embed} if cond_embed is not None else {}))
            x0_pred = self.predict_x0_from_noise(x, t_batch, noise_pred)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            x0_prev = x0_pred

        return x0_pred

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        device: torch.device,
        use_ddim: bool = True,
        ddim_steps: int = 50,
    ) -> torch.Tensor:
        """Unified sampling interface."""
        if use_ddim:
            return self.ddim_sample(shape, device, num_steps=ddim_steps)
        return self.ddpm_sample(shape, device)
