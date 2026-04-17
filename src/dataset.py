"""
Token-Image Diffusion PoC — Dataset
=====================================
1. Load text corpus (wikitext or security domain JSONL)
2. Tokenize into fixed-length chunks
3. Extract embeddings from pretrained model
4. PCA-project to target dimensionality
5. Encode as images using chosen method
6. Package as PyTorch Dataset
"""
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import os

from config import Config
from encoders import EmbeddingProjector, encode


class TokenImageDataset(Dataset):
    """Pre-computed dataset of token-embedding images."""

    def __init__(self, images: np.ndarray, token_ids: np.ndarray, value_ranges: np.ndarray):
        """
        images:       (N, C, H, W) float32 images in [-1, 1]
        token_ids:    (N, seq_len) int64 original token IDs (for evaluation)
        value_ranges: (N, 2) min/max of projected embeddings before normalization
        """
        self.images = images
        self.token_ids = token_ids
        self.value_ranges = value_ranges

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx], dtype=torch.float32),
            torch.tensor(self.token_ids[idx], dtype=torch.long),
            torch.tensor(self.value_ranges[idx], dtype=torch.float32),
        )


def _load_security_jsonl(paths: tuple):
    """Generator yielding flattened text from ChatML JSONL files.

    Flattens messages using Qwen-native <|im_start|>/<|im_end|> markers
    so the tokenizer produces real embedding-backed tokens.
    """
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                messages = data.get("messages", [])
                if not messages:
                    continue
                parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                yield "\n".join(parts)


def prepare_dataset(cfg: Config, cache_dir: str = "data_cache") -> tuple:
    """
    Full pipeline: text → tokens → embeddings → PCA → images.

    Returns:
        dataset:    TokenImageDataset
        projector:  fitted EmbeddingProjector
        embed_table: (vocab_size, d_model) numpy array
        tokenizer:  HuggingFace tokenizer
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    if cfg.use_vqgan:
        enc_tag = f"vqgan-{getattr(cfg, 'ae_variant', 'vq')}"
    elif cfg.use_vqvae:
        enc_tag = "vqvae"
    else:
        enc_tag = cfg.encoder
    images_file = cache_path / f"images_{cfg.data_source}_{enc_tag}_{cfg.seq_len}_{cfg.proj_dim}_{cfg.image_channels}_{cfg.max_samples}_gnorm.npz"

    # ------------------------------------------------------------------
    #  Step 1: Load tokenizer
    # ------------------------------------------------------------------
    print(f"[Dataset] Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    #  Step 2: Load embedding table (just the embedding layer, not full model)
    # ------------------------------------------------------------------
    print(f"[Dataset] Loading embedding weights from {cfg.model_name}")
    model = AutoModel.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    # Extract embedding table
    if hasattr(model, "embed_tokens"):
        embed_table = model.embed_tokens.weight.detach().cpu().numpy()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_table = model.model.embed_tokens.weight.detach().cpu().numpy()
    elif hasattr(model, "get_input_embeddings"):
        embed_table = model.get_input_embeddings().weight.detach().cpu().numpy()
    else:
        raise RuntimeError("Cannot find embedding layer")

    d_model = embed_table.shape[1]
    vocab_size = embed_table.shape[0]
    print(f"[Dataset] Embedding table: vocab={vocab_size}, d_model={d_model}")

    # Free the full model
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Check cache
    if images_file.exists():
        print(f"[Dataset] Loading cached images from {images_file}")
        data = np.load(images_file)
        images = data["images"]
        token_ids = data["token_ids"]
        value_ranges = data["value_ranges"]

        if cfg.use_vqgan:
            from vqgan import VQGAN, VQGANProjector
            variant = getattr(cfg, "ae_variant", "vq")
            cb = (cfg.vqgan_bigvq_codebook_size if variant == "bigvq"
                  else cfg.vqgan_codebook_size)
            bn_type = "kl" if variant == "kl" else "vq"
            use_hilbert = getattr(cfg, "vqgan_hilbert", False)
            vqgan_model = VQGAN(
                d_model=d_model, seq_len=cfg.seq_len,
                hidden_dim=cfg.vqgan_hidden_dim,
                latent_channels=cfg.vqgan_latent_channels,
                latent_size=cfg.vqgan_latent_size,
                codebook_size=cb,
                codebook_dim=cfg.vqgan_codebook_dim,
                bottleneck_type=bn_type,
                use_hilbert=use_hilbert,
            ).to(cfg.device)
            projector = VQGANProjector(vqgan_model, cfg.device)
            projector.load(str(cache_path / f"vqgan_projector_{variant}.pt"))
        elif cfg.use_vqvae:
            from vqvae import EmbeddingVQVAE, VQVAEProjector
            import torch as _torch
            vqvae_model = EmbeddingVQVAE(
                d_model=d_model, seq_len=cfg.seq_len,
                hidden_dim=cfg.vqvae_hidden_dim,
                vq_embed_dim=cfg.vqvae_embed_dim,
                vq_num_embeddings=cfg.vqvae_num_embeddings,
            ).to(cfg.device)
            vqvae_model.load_state_dict(
                _torch.load(str(cache_path / "vqvae_best.pt"), map_location=cfg.device, weights_only=True)
            )
            vqvae_model.eval()
            projector = VQVAEProjector(vqvae_model, cfg.device)
        else:
            projector = EmbeddingProjector(cfg.proj_dim)
            projector.load(str(cache_path / "projector.npz"))

        dataset = TokenImageDataset(images, token_ids, value_ranges)
        print(f"[Dataset] Loaded {len(dataset)} cached samples")
        return dataset, projector, embed_table, tokenizer

    # ------------------------------------------------------------------
    #  Step 3: Load and tokenize text corpus
    # ------------------------------------------------------------------
    if cfg.data_source == "wikitext":
        print(f"[Dataset] Loading {cfg.dataset_name}/{cfg.dataset_config}")
        from datasets import load_dataset
        raw = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")
        texts = (item.get("text", "") for item in raw)
    else:
        print(f"[Dataset] Loading security JSONL ({cfg.data_source}): {len(cfg.data_paths)} file(s)")
        texts = _load_security_jsonl(cfg.data_paths)

    # Concatenate all text, tokenize, chunk into seq_len pieces
    print("[Dataset] Tokenizing...")
    all_ids = []
    for text in tqdm(texts, desc="Tokenizing"):
        if len(text.strip()) < 10:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= cfg.max_samples * cfg.seq_len * 2:
            break

    # Chunk into fixed-length sequences
    all_ids = np.array(all_ids, dtype=np.int64)
    n_chunks = min(len(all_ids) // cfg.seq_len, cfg.max_samples)
    all_ids = all_ids[: n_chunks * cfg.seq_len].reshape(n_chunks, cfg.seq_len)
    print(f"[Dataset] {n_chunks} sequences of length {cfg.seq_len}")

    # ------------------------------------------------------------------
    #  Step 4: Look up embeddings
    # ------------------------------------------------------------------
    print("[Dataset] Looking up embeddings...")
    # all_ids: (N, seq_len) — lookup in embed_table
    # Memory-map large embedding arrays to avoid 18GB+ RAM pinning
    emb_tmp_path = cache_path / f"embeddings_tmp_{cfg.seq_len}.npy"
    if cfg.use_vqgan and not emb_tmp_path.exists():
        print("[Dataset] Saving embeddings to disk for memory-mapping...")
        all_embeddings = embed_table[all_ids]  # (N, seq_len, d_model)
        np.save(str(emb_tmp_path), all_embeddings)
        del all_embeddings
        all_embeddings = np.load(str(emb_tmp_path), mmap_mode='r')
        print(f"[Dataset] Memory-mapped embeddings: {all_embeddings.shape}")
    elif cfg.use_vqgan and emb_tmp_path.exists():
        print(f"[Dataset] Loading memory-mapped embeddings from {emb_tmp_path}")
        all_embeddings = np.load(str(emb_tmp_path), mmap_mode='r')
    else:
        all_embeddings = embed_table[all_ids]  # (N, seq_len, d_model)

    # ------------------------------------------------------------------
    #  Step 5 & 6: Encode as images (PCA or VQ-VAE)
    # ------------------------------------------------------------------
    if cfg.use_vqgan:
        # ── VQ-GAN path ──
        from vqgan import train_vqgan, VQGANProjector, VQGAN

        variant = getattr(cfg, "ae_variant", "vq")
        cb = (cfg.vqgan_bigvq_codebook_size if variant == "bigvq"
              else cfg.vqgan_codebook_size)
        bn_type = "kl" if variant == "kl" else "vq"

        vqgan_path = cache_path / f"vqgan_projector_{variant}.pt"
        if vqgan_path.exists():
            print(f"[Dataset] Loading cached VQ-GAN from {vqgan_path}")
            use_hilbert = getattr(cfg, "vqgan_hilbert", False)
            vqgan_model = VQGAN(
                d_model=d_model, seq_len=cfg.seq_len,
                hidden_dim=cfg.vqgan_hidden_dim,
                latent_channels=cfg.vqgan_latent_channels,
                latent_size=cfg.vqgan_latent_size,
                latent_h=getattr(cfg, "vqgan_latent_h", None),
                latent_w=getattr(cfg, "vqgan_latent_w", None),
                codebook_size=cb,
                codebook_dim=cfg.vqgan_codebook_dim,
                bottleneck_type=bn_type,
                use_hilbert=use_hilbert,
            ).to(cfg.device)
            projector = VQGANProjector(vqgan_model, cfg.device)
            projector.load(str(vqgan_path))
        else:
            print(f"[Dataset] Training VQ-GAN encoder (variant={variant})...")
            projector = train_vqgan(cfg, all_embeddings, all_ids, embed_table)

        # Encode all samples using VQ-GAN → normalized latents
        print("[Dataset] Encoding latents with VQ-GAN...")
        images = []
        for i in tqdm(range(n_chunks), desc="VQ-GAN Encoding"):
            img = projector.project(all_embeddings[i])  # (C, H, W) normalized
            images.append(img)

        images = np.stack(images, axis=0)
        value_ranges = np.zeros((n_chunks, 2), dtype=np.float32)  # unused for VQ-GAN

    elif cfg.use_vqvae:
        # ── VQ-VAE path ──
        from vqvae import train_vqvae, VQVAEProjector, EmbeddingVQVAE
        import torch as _torch

        vqvae_path = cache_path / "vqvae_best.pt"
        if vqvae_path.exists():
            print(f"[Dataset] Loading cached VQ-VAE from {vqvae_path}")
            vqvae_model = EmbeddingVQVAE(
                d_model=d_model,
                seq_len=cfg.seq_len,
                hidden_dim=cfg.vqvae_hidden_dim,
                vq_embed_dim=cfg.vqvae_embed_dim,
                vq_num_embeddings=cfg.vqvae_num_embeddings,
            ).to(cfg.device)
            vqvae_model.load_state_dict(
                _torch.load(str(vqvae_path), map_location=cfg.device, weights_only=True)
            )
            vqvae_model.eval()
            projector = VQVAEProjector(vqvae_model, cfg.device)
        else:
            print("[Dataset] Training VQ-VAE encoder...")
            projector = train_vqvae(cfg, all_embeddings, all_ids, embed_table)

        # Encode all samples using VQ-VAE
        print("[Dataset] Encoding images with VQ-VAE...")
        images = []
        for i in tqdm(range(n_chunks), desc="VQ-VAE Encoding"):
            img = projector.project(all_embeddings[i])  # (C, 64, 64)
            images.append(img)

        images = np.stack(images, axis=0)  # (N, C, 64, 64)
        value_ranges = np.zeros((n_chunks, 2), dtype=np.float32)  # unused for VQ-VAE

    else:
        # ── PCA path ──
        print("[Dataset] Fitting PCA projector...")
        flat_emb = all_embeddings.reshape(-1, d_model)
        n_pca = min(500_000, flat_emb.shape[0])
        pca_indices = np.random.choice(flat_emb.shape[0], n_pca, replace=False)
        projector = EmbeddingProjector(cfg.proj_dim)
        projector.fit(flat_emb[pca_indices])

        # Compute global normalization stats across all PCA projections
        print("[Dataset] Computing global normalization stats...")
        all_projected = np.array(
            [projector.project(all_embeddings[i]) for i in range(n_chunks)]
        )  # (N, seq_len, proj_dim)
        projector.fit_normalization(all_projected)
        projector.save(str(cache_path / "projector.npz"))

        # Encode images using global z-score normalization
        global_norm = (projector.global_proj_mean, projector.global_proj_std,
                       cfg.clip_sigma)
        print(f"[Dataset] Encoding as images (method={cfg.encoder}, global z-score)...")
        images = []
        value_ranges = []

        for i in tqdm(range(n_chunks), desc="Encoding"):
            proj = all_projected[i]  # (seq_len, proj_dim) — already projected
            img = encode(proj, method=cfg.encoder, num_channels=cfg.image_channels,
                        global_norm_params=global_norm)
            images.append(img)
            value_ranges.append([0.0, 0.0])  # unused with global norm

        images = np.stack(images, axis=0)
        value_ranges = np.array(value_ranges, dtype=np.float32)

    print(f"[Dataset] Image shape: {images.shape}, "
          f"range: [{images.min():.2f}, {images.max():.2f}]")

    # Save cache
    np.savez_compressed(
        images_file,
        images=images,
        token_ids=all_ids,
        value_ranges=value_ranges,
    )
    print(f"[Dataset] Saved cache to {images_file}")

    dataset = TokenImageDataset(images, all_ids, value_ranges)
    return dataset, projector, embed_table, tokenizer


def get_dataloader(dataset: TokenImageDataset, cfg: Config, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=False,  # MPS doesn't support pinned memory
        drop_last=True,
    )
