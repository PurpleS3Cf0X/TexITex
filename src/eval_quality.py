"""
Word Order Quality Evaluation
==============================
Extract generated samples from training logs and compute:
  1. Bigram coherence — fraction of adjacent token pairs seen in training corpus
  2. LM perplexity — Qwen2.5-1.5B teacher-forced perplexity
  3. Real-word ratio — fraction of tokens that decode to real English words

Usage: python eval_quality.py runs/dit_long.log runs/dit_posemb.log runs/dit_hilbert.log
"""
import sys
import re
import json
import numpy as np
import torch
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_samples(log_path: str) -> list[str]:
    """Extract generated text samples from a run log file."""
    text = Path(log_path).read_text()
    samples = []
    # Match "--- Sample N ---" followed by text until next "---" or end
    pattern = r'--- Sample \d+ ---\n(.*?)(?=\n--- Sample|\n\[Viz\]|\Z)'
    for m in re.finditer(pattern, text, re.DOTALL):
        sample = m.group(1).strip()
        if sample:
            samples.append(sample)
    return samples


def build_bigram_table(tokenizer, data_paths: tuple, max_samples: int = 50000, seq_len: int = 64) -> set:
    """Build set of (tok_a, tok_b) bigrams from training corpus."""
    bigrams = set()
    texts = []
    for p in data_paths:
        with open(p) as f:
            for line in f:
                obj = json.loads(line)
                t = obj.get("text", "") or obj.get("input", "") or obj.get("instruction", "")
                # ChatML messages format: flatten all message contents
                if not t and "messages" in obj:
                    parts = [m.get("content", "") for m in obj["messages"] if m.get("content")]
                    t = " ".join(parts)
                if t:
                    texts.append(t)
                if len(texts) >= max_samples:
                    break
        if len(texts) >= max_samples:
            break

    for t in texts[:max_samples]:
        ids = tokenizer.encode(t, add_special_tokens=False)[:seq_len]
        for i in range(len(ids) - 1):
            bigrams.add((ids[i], ids[i + 1]))
    return bigrams


def bigram_coherence(text: str, tokenizer, bigram_set: set) -> float:
    """Fraction of adjacent token pairs that appear in training bigrams."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return 0.0
    hits = sum(1 for i in range(len(ids) - 1) if (ids[i], ids[i + 1]) in bigram_set)
    return hits / (len(ids) - 1)


def lm_perplexity(text: str, tokenizer, model, device: str) -> float:
    """Teacher-forced perplexity under Qwen2.5-1.5B."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return float("inf")
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    return np.exp(loss)


def real_word_ratio(text: str, tokenizer) -> float:
    """Fraction of tokens that decode to real ASCII words (len>=2) or common punctuation."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return 0.0
    common_punct = set(".,;:!?-()\"'")
    real = 0
    for tok_id in ids:
        decoded = tokenizer.decode([tok_id]).strip()
        if len(decoded) >= 2 and decoded.isascii() and decoded.replace("'", "").replace("-", "").isalpha():
            real += 1
        elif decoded in common_punct:
            real += 1
    return real / len(ids)


def structural_noise_ratio(text: str) -> float:
    """Fraction of characters that are JSON/structural noise (brackets, quotes, escape chars).

    High values (>0.25) indicate templated/boilerplate output rather than coherent prose.
    """
    if not text:
        return 1.0
    noise_chars = set('{}[]"\\/:;<>|`~#*_=+')
    noise = sum(1 for c in text if c in noise_chars)
    return noise / len(text)


def is_prose(text: str, max_noise: float = 0.20, min_realword: float = 0.40,
             tokenizer=None) -> bool:
    """Classify a sample as prose (True) or boilerplate/structural (False)."""
    if structural_noise_ratio(text) > max_noise:
        return False
    if tokenizer is not None:
        if real_word_ratio(text, tokenizer) < min_realword:
            return False
    return True


def composite_score(bigram: float, ppl: float, realword: float) -> float:
    """Composite quality score that rewards high bigram × realword and low PPL.

    Scale: 0 (terrible) → ~1 (excellent). Uses log-PPL to compress the long tail.
    """
    import math
    ppl_term = 1.0 / (1.0 + math.log1p(max(ppl, 1.0)) / 5.0)
    return bigram * realword * ppl_term


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_quality.py <log1> [log2] [log3] ...")
        sys.exit(1)

    log_paths = sys.argv[1:]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    model = model.to(device).eval()

    print("Building bigram table from training corpus...")
    data_paths = (
        "/Volumes/R3D_1/redteam_project/data/final_v2/train.jsonl",
        "/Volumes/R3D_1/blueteam_project/data/final/train.jsonl",
    )
    bigram_set = build_bigram_table(tokenizer, data_paths)
    print(f"  {len(bigram_set)} unique bigrams")

    # Evaluate each log
    results = {}
    for log_path in log_paths:
        name = Path(log_path).stem
        samples = extract_samples(log_path)
        if not samples:
            print(f"\n[{name}] No samples found in {log_path}")
            continue

        bigrams = [bigram_coherence(s, tokenizer, bigram_set) for s in samples]
        perplexities = [lm_perplexity(s, tokenizer, model, device) for s in samples]
        real_words = [real_word_ratio(s, tokenizer) for s in samples]
        noises = [structural_noise_ratio(s) for s in samples]
        prose_flags = [is_prose(s, tokenizer=tokenizer) for s in samples]
        composites = [composite_score(b, p, r)
                      for b, p, r in zip(bigrams, perplexities, real_words)]

        # Prose-only aggregates (excludes JSON boilerplate)
        prose_idx = [i for i, f in enumerate(prose_flags) if f]
        prose_ppl = [perplexities[i] for i in prose_idx]
        prose_bigram = [bigrams[i] for i in prose_idx]
        prose_real = [real_words[i] for i in prose_idx]

        results[name] = {
            "n_samples": len(samples),
            "n_prose": len(prose_idx),
            "bigram_mean": np.mean(bigrams),
            "bigram_std": np.std(bigrams),
            "ppl_median": np.median(perplexities),
            "ppl_mean": np.mean([p for p in perplexities if p < 1e6]),
            "real_word_mean": np.mean(real_words),
            "real_word_std": np.std(real_words),
            "noise_mean": np.mean(noises),
            "composite_mean": np.mean(composites),
            "composite_max": np.max(composites),
            "prose_ppl_median": np.median(prose_ppl) if prose_ppl else float("nan"),
            "prose_bigram_mean": np.mean(prose_bigram) if prose_bigram else float("nan"),
            "prose_real_mean": np.mean(prose_real) if prose_real else float("nan"),
        }

        print(f"\n{'='*60}")
        print(f"  {name}  ({len(samples)} samples, {len(prose_idx)} prose)")
        print(f"{'='*60}")
        print(f"  Bigram coherence:  {results[name]['bigram_mean']:.3f} +/- {results[name]['bigram_std']:.3f}")
        print(f"  LM perplexity:     {results[name]['ppl_median']:.1f} (median, all)")
        print(f"  Real-word ratio:   {results[name]['real_word_mean']:.3f} +/- {results[name]['real_word_std']:.3f}")
        print(f"  Structural noise:  {results[name]['noise_mean']:.3f} (lower=better)")
        print(f"  Composite score:   {results[name]['composite_mean']:.4f} (mean)  "
              f"{results[name]['composite_max']:.4f} (best sample)")
        print(f"  -- Prose only ({len(prose_idx)}/{len(samples)}) --")
        print(f"  Prose PPL(med):    {results[name]['prose_ppl_median']:.1f}")
        print(f"  Prose bigram:      {results[name]['prose_bigram_mean']:.3f}")
        print(f"  Prose real-word:   {results[name]['prose_real_mean']:.3f}")

        # Rank samples by composite score and show top-5 prose samples
        ranked = sorted(range(len(samples)), key=lambda i: composites[i], reverse=True)
        print(f"\n  Top 5 by composite score:")
        for rank, i in enumerate(ranked[:5]):
            tag = "PROSE" if prose_flags[i] else "noise"
            print(f"    #{rank+1} [{tag}] comp={composites[i]:.3f} "
                  f"bigram={bigrams[i]:.3f} ppl={perplexities[i]:.0f} "
                  f"real={real_words[i]:.3f}")
            print(f"        {samples[i][:120]}...")

        print(f"\n  Per-sample breakdown:")
        for i, s in enumerate(samples):
            tag = "PROSE" if prose_flags[i] else "noise"
            print(f"    [{i+1}][{tag}] comp={composites[i]:.3f} bigram={bigrams[i]:.3f}  "
                  f"ppl={perplexities[i]:.0f}  real={real_words[i]:.3f}  "
                  f"noise={noises[i]:.2f}  text={s[:70]}...")

    # Summary comparison table
    if len(results) > 1:
        print(f"\n{'='*100}")
        print(f"  COMPARISON TABLE  (ranked by composite_mean — higher=better)")
        print(f"{'='*100}")
        header = (f"  {'Experiment':<28} {'N(prose)':>9} {'Bigram':>7} "
                  f"{'PPL(all)':>9} {'PPL(pr)':>9} {'Real':>7} "
                  f"{'Noise':>7} {'Comp(μ)':>9} {'Comp(max)':>10}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        # Sort by composite mean descending
        ranked = sorted(results.items(), key=lambda kv: kv[1]["composite_mean"], reverse=True)
        for name, r in ranked:
            prose_ppl_str = (f"{r['prose_ppl_median']:>9.1f}"
                             if r['prose_ppl_median'] == r['prose_ppl_median'] else f"{'n/a':>9}")
            print(f"  {name:<28} "
                  f"{r['n_prose']:>3d}/{r['n_samples']:<5d} "
                  f"{r['bigram_mean']:>7.3f} {r['ppl_median']:>9.1f} "
                  f"{prose_ppl_str} {r['real_word_mean']:>7.3f} "
                  f"{r['noise_mean']:>7.3f} {r['composite_mean']:>9.4f} "
                  f"{r['composite_max']:>10.4f}")


if __name__ == "__main__":
    main()
