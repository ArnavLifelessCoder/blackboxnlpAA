"""
cache_utils.py — Activation Caching to/from Disk
=================================================
Utilities for saving and loading activation tensors with standardized naming.
Uses torch.save/torch.load with the naming convention from config.py.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def save_activations(
    activations: Dict[int, torch.Tensor],
    output_dir: Path,
    model_name: str,
    concept: str,
    domain: str,
    prefix: str = "",
) -> List[Path]:
    """Save activation tensors to disk, one file per layer.

    Args:
        activations: Dict mapping layer index -> tensor of shape (n_pairs, d_model).
        output_dir: Directory to save files to.
        model_name: Model identifier (e.g., "qwen-2.5-3b-instruct").
        concept: Concept name (e.g., "refusal").
        domain: Domain name (e.g., "violence").
        prefix: Optional prefix to add to filenames (e.g., "pos_" or "neg_").

    Returns:
        List of paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    model_short = model_name.replace("-", "_").replace(".", "")

    for layer_idx, tensor in activations.items():
        filename = f"{prefix}{model_short}_{concept}_{domain}_layer{layer_idx:03d}.pt"
        filepath = output_dir / filename

        # Save tensor in float32 for numerical stability
        torch.save(tensor.cpu().float(), filepath)
        saved_paths.append(filepath)
        logger.debug(f"Saved activations: {filepath} | shape={tensor.shape}")

    logger.info(
        f"Saved {len(saved_paths)} activation files to {output_dir} "
        f"({model_name}/{concept}/{domain})"
    )
    return saved_paths


def load_activations(
    input_dir: Path,
    model_name: str,
    concept: str,
    domain: str,
    layers: Optional[List[int]] = None,
    prefix: str = "",
    device: str = "cpu",
) -> Dict[int, torch.Tensor]:
    """Load activation tensors from disk.

    Args:
        input_dir: Directory containing activation files.
        model_name: Model identifier.
        concept: Concept name.
        domain: Domain name.
        layers: Specific layers to load. None = load all available.
        prefix: Filename prefix to match.
        device: Device to load tensors to.

    Returns:
        Dict mapping layer index -> tensor of shape (n_pairs, d_model).
    """
    input_dir = Path(input_dir)
    model_short = model_name.replace("-", "_").replace(".", "")
    pattern = f"{prefix}{model_short}_{concept}_{domain}_layer*.pt"

    activations = {}

    for filepath in sorted(input_dir.glob(pattern)):
        # Extract layer index from filename
        stem = filepath.stem
        layer_str = stem.split("_layer")[-1]
        layer_idx = int(layer_str)

        if layers is not None and layer_idx not in layers:
            continue

        tensor = torch.load(filepath, map_location=device, weights_only=True)
        activations[layer_idx] = tensor
        logger.debug(f"Loaded activations: {filepath} | shape={tensor.shape}")

    logger.info(
        f"Loaded {len(activations)} activation files from {input_dir} "
        f"({model_name}/{concept}/{domain})"
    )
    return activations


def save_direction(
    direction: torch.Tensor,
    output_dir: Path,
    model_name: str,
    concept: str,
    domain: str,
    layer: int,
) -> Path:
    """Save a direction vector to disk.

    Args:
        direction: Direction vector of shape (d_model,).
        output_dir: Directory to save to.
        model_name: Model identifier.
        concept: Concept name.
        domain: Domain name (use "global" for the global direction).
        layer: Layer index.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_short = model_name.replace("-", "_").replace(".", "")
    filename = f"dir_{model_short}_{concept}_{domain}_layer{layer:03d}.pt"
    filepath = output_dir / filename

    torch.save(direction.cpu().float(), filepath)
    logger.info(f"Saved direction: {filepath} | shape={direction.shape}")
    return filepath


def load_direction(
    input_dir: Path,
    model_name: str,
    concept: str,
    domain: str,
    layer: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Load a direction vector from disk.

    Args:
        input_dir: Directory containing direction files.
        model_name: Model identifier.
        concept: Concept name.
        domain: Domain name (use "global" for the global direction).
        layer: Layer index.
        device: Device to load tensor to.

    Returns:
        Direction vector of shape (d_model,).
    """
    input_dir = Path(input_dir)
    model_short = model_name.replace("-", "_").replace(".", "")
    filename = f"dir_{model_short}_{concept}_{domain}_layer{layer:03d}.pt"
    filepath = input_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Direction file not found: {filepath}")

    direction = torch.load(filepath, map_location=device, weights_only=True)
    logger.info(f"Loaded direction: {filepath} | shape={direction.shape}")
    return direction


def list_cached_activations(
    cache_dir: Path,
    model_name: Optional[str] = None,
    concept: Optional[str] = None,
    domain: Optional[str] = None,
) -> List[Dict]:
    """List all cached activation files, optionally filtered.

    Returns a list of dicts with keys: path, model, concept, domain, layer.
    """
    cache_dir = Path(cache_dir)
    results = []

    for filepath in sorted(cache_dir.glob("*.pt")):
        if filepath.stem.startswith("dir_"):
            continue  # Skip direction files

        parts = filepath.stem.split("_layer")
        if len(parts) != 2:
            continue

        metadata_str = parts[0]
        layer_idx = int(parts[1])

        info = {
            "path": filepath,
            "layer": layer_idx,
            "filename": filepath.name,
        }

        # Apply filters
        if model_name and model_name.replace("-", "_").replace(".", "") not in metadata_str:
            continue
        if concept and concept not in metadata_str:
            continue
        if domain and domain not in metadata_str:
            continue

        results.append(info)

    return results
