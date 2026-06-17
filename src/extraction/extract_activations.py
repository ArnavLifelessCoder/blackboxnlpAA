"""
extract_activations.py — Main Activation Extraction Script
===========================================================
Loads a model, iterates over contrastive prompt pairs, runs forward passes,
and caches residual stream activations per layer.

Supports both TransformerLens and HuggingFace backends.

Usage:
    python -m src.extraction.extract_activations \
        --model gemma-2-2b-it \
        --concept refusal \
        --domain violence \
        --data data/prompt_pairs/refusal/violence.jsonl \
        --output results/activations/ \
        --max-pairs 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import MODELS, EXTRACTION, PATHS
from src.extraction.hooks import (
    extract_with_transformer_lens,
    register_hooks,
    extract_with_hooks,
)
from src.extraction.cache_utils import save_activations

logger = logging.getLogger(__name__)


# ============================================================
# Data Loading
# ============================================================

def load_prompt_pairs(
    filepath: Path,
    max_pairs: Optional[int] = None,
) -> List[Dict]:
    """Load contrastive prompt pairs from a JSONL file.

    Each line should be a JSON object with at least 'positive' and 'negative' keys.

    Args:
        filepath: Path to the JSONL file.
        max_pairs: Maximum number of pairs to load.

    Returns:
        List of pair dictionaries.
    """
    pairs = []
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Prompt pair file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                pair = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num} in {filepath}: {e}")
                continue

            if "positive" not in pair or "negative" not in pair:
                logger.warning(
                    f"Skipping line {line_num}: missing 'positive' or 'negative' key"
                )
                continue

            pairs.append(pair)

            if max_pairs and len(pairs) >= max_pairs:
                break

    logger.info(f"Loaded {len(pairs)} prompt pairs from {filepath}")
    return pairs


# ============================================================
# Model Loading
# ============================================================

def load_model_transformer_lens(model_key: str):
    """Load a model using TransformerLens.

    Returns:
        (model, tokenizer) tuple.
    """
    from transformer_lens import HookedTransformer

    model_config = MODELS[model_key]
    logger.info(f"Loading {model_config.name} via TransformerLens...")

    kwargs = {
        "dtype": torch.float16 if EXTRACTION.use_fp16 else torch.float32,
    }

    model = HookedTransformer.from_pretrained(
        model_config.transformer_lens_name,
        **kwargs,
    )
    model.eval()
    tokenizer = model.tokenizer

    logger.info(f"Loaded {model_config.name}: {model_config.n_layers} layers, d_model={model_config.d_model}")
    return model, tokenizer


def load_model_huggingface(model_key: str):
    """Load a model using HuggingFace Transformers (fallback).

    Returns:
        (model, tokenizer) tuple.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = MODELS[model_key]
    logger.info(f"Loading {model_config.name} via HuggingFace...")

    load_kwargs = {
        "torch_dtype": torch.float16 if EXTRACTION.use_fp16 else torch.float32,
        "device_map": "auto",
    }

    # Apply quantization if specified
    if model_config.quantization == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif model_config.quantization == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.hf_id,
        **load_kwargs,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loaded {model_config.name}: {model_config.n_layers} layers, d_model={model_config.d_model}")
    return model, tokenizer


# ============================================================
# Core Extraction Logic
# ============================================================

def tokenize_pairs(
    pairs: List[Dict],
    tokenizer,
    max_length: int = 512,
) -> Tuple[Dict, Dict]:
    """Tokenize positive and negative prompts separately.

    Returns:
        (positive_encodings, negative_encodings) — each is a dict with
        'input_ids' and 'attention_mask' tensors.
    """
    positive_texts = [p["positive"] for p in pairs]
    negative_texts = [p["negative"] for p in pairs]

    pos_enc = tokenizer(
        positive_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    neg_enc = tokenizer(
        negative_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return pos_enc, neg_enc


def extract_activations_for_pairs(
    model_key: str,
    pairs: List[Dict],
    output_dir: Path,
    concept: str,
    domain: str,
    backend: str = "transformer_lens",
    target_layers: Optional[List[int]] = None,
    batch_size: int = 4,
    max_seq_len: int = 512,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Extract activations for a set of contrastive prompt pairs.

    Runs forward passes for all positive and negative prompts, extracts
    residual stream activations at each target layer, and saves to disk.

    Args:
        model_key: Key into MODELS dict (e.g., "gemma-2-2b-it").
        pairs: List of prompt pair dicts with 'positive' and 'negative' keys.
        output_dir: Directory to save cached activations.
        concept: Concept name (e.g., "refusal").
        domain: Domain name (e.g., "violence").
        backend: "transformer_lens" or "huggingface".
        target_layers: Layers to extract from. None = all.
        batch_size: Batch size for forward passes.
        max_seq_len: Maximum sequence length.

    Returns:
        (positive_activations, negative_activations) — each is a dict
        mapping layer -> tensor of shape (n_pairs, d_model).
    """
    model_config = MODELS[model_key]

    # Resolve target layers
    if target_layers is None:
        target_layers = (
            EXTRACTION.target_layers
            if EXTRACTION.target_layers is not None
            else list(range(model_config.n_layers))
        )

    # Load model
    if backend == "transformer_lens":
        model, tokenizer = load_model_transformer_lens(model_key)
    else:
        model, tokenizer = load_model_huggingface(model_key)

    # Tokenize
    pos_enc, neg_enc = tokenize_pairs(pairs, tokenizer, max_length=max_seq_len)

    # Determine device
    if backend == "transformer_lens":
        device = next(model.parameters()).device
    else:
        device = next(model.parameters()).device

    # Extract activations in batches
    all_pos_acts = {layer: [] for layer in target_layers}
    all_neg_acts = {layer: [] for layer in target_layers}

    n_pairs = len(pairs)
    n_batches = (n_pairs + batch_size - 1) // batch_size

    # --- Setup hooks for HuggingFace backend ---
    hook_storage = None
    if backend != "transformer_lens":
        hook_storage = register_hooks(
            model,
            target_layers=target_layers,
            token_position=EXTRACTION.token_position,
        )

    try:
        for batch_idx in tqdm(range(n_batches), desc=f"Extracting {concept}/{domain}"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_pairs)

            # --- Positive prompts ---
            pos_ids = pos_enc["input_ids"][start:end].to(device)
            pos_mask = pos_enc["attention_mask"][start:end].to(device)

            with torch.no_grad():
                if backend == "transformer_lens":
                    pos_acts = extract_with_transformer_lens(
                        model, pos_ids,
                        target_layers=target_layers,
                        hook_point=EXTRACTION.hook_point,
                        token_position=EXTRACTION.token_position,
                        attention_mask=pos_mask,
                    )
                else:
                    pos_acts = extract_with_hooks(
                        model, pos_ids,
                        hook_storage=hook_storage,
                        attention_mask=pos_mask,
                    )

            for layer in target_layers:
                all_pos_acts[layer].append(pos_acts[layer].cpu())

            # --- Negative prompts ---
            neg_ids = neg_enc["input_ids"][start:end].to(device)
            neg_mask = neg_enc["attention_mask"][start:end].to(device)

            with torch.no_grad():
                if backend == "transformer_lens":
                    neg_acts = extract_with_transformer_lens(
                        model, neg_ids,
                        target_layers=target_layers,
                        hook_point=EXTRACTION.hook_point,
                        token_position=EXTRACTION.token_position,
                        attention_mask=neg_mask,
                    )
                else:
                    neg_acts = extract_with_hooks(
                        model, neg_ids,
                        hook_storage=hook_storage,
                        attention_mask=neg_mask,
                    )

            for layer in target_layers:
                all_neg_acts[layer].append(neg_acts[layer].cpu())

    finally:
        if hook_storage is not None:
            hook_storage.remove_hooks()

    # Concatenate batches
    pos_result = {layer: torch.cat(all_pos_acts[layer], dim=0) for layer in target_layers}
    neg_result = {layer: torch.cat(all_neg_acts[layer], dim=0) for layer in target_layers}

    # Save to disk
    save_activations(pos_result, output_dir, model_key, concept, domain, prefix="pos_")
    save_activations(neg_result, output_dir, model_key, concept, domain, prefix="neg_")

    logger.info(
        f"Extraction complete: {n_pairs} pairs × {len(target_layers)} layers "
        f"= {n_pairs * len(target_layers) * 2} activation vectors cached."
    )

    return pos_result, neg_result


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract residual stream activations from contrastive prompt pairs."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODELS.keys()),
        help="Model key (e.g., 'gemma-2-2b-it')",
    )
    parser.add_argument(
        "--concept", type=str, required=True,
        help="Concept name (e.g., 'refusal', 'honesty')",
    )
    parser.add_argument(
        "--domain", type=str, required=True,
        help="Domain name (e.g., 'violence', 'factual_trivia')",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to JSONL file with prompt pairs",
    )
    parser.add_argument(
        "--output", type=str, default=str(PATHS.activations),
        help="Output directory for cached activations",
    )
    parser.add_argument(
        "--backend", type=str, default="transformer_lens",
        choices=["transformer_lens", "huggingface"],
        help="Model loading backend",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Maximum number of prompt pairs to process",
    )
    parser.add_argument(
        "--batch-size", type=int, default=EXTRACTION.batch_size,
        help="Batch size for forward passes",
    )
    parser.add_argument(
        "--layers", type=int, nargs="*", default=None,
        help="Specific layer indices to extract (default: all)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Load prompt pairs
    pairs = load_prompt_pairs(Path(args.data), max_pairs=args.max_pairs)

    if not pairs:
        logger.error("No prompt pairs loaded — check your data file.")
        sys.exit(1)

    # Run extraction
    extract_activations_for_pairs(
        model_key=args.model,
        pairs=pairs,
        output_dir=Path(args.output),
        concept=args.concept,
        domain=args.domain,
        backend=args.backend,
        target_layers=args.layers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
