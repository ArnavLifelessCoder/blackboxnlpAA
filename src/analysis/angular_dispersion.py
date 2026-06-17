"""
angular_dispersion.py — Angular Dispersion Analysis
====================================================
Core analysis module for the paper's central claim.

Measures how much per-domain concept directions deviate from the global
concept direction using cosine similarity. High cosine similarity = universal
direction. Low cosine similarity = domain-fragmented direction.

Produces:
  - Per-layer cosine similarity between each domain direction and the global direction
  - Angular dispersion statistic (variance of angles)
  - Layer-wise profiles showing where fragmentation is strongest
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from src.extraction.cache_utils import load_direction

logger = logging.getLogger(__name__)


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        v1, v2: Tensors of shape (d_model,).

    Returns:
        Cosine similarity as a Python float.
    """
    v1 = v1.float()
    v2 = v2.float()

    dot = torch.dot(v1, v2)
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)

    if norm1 == 0 or norm2 == 0:
        logger.warning("Zero-norm vector in cosine similarity computation.")
        return 0.0

    return (dot / (norm1 * norm2)).item()


def pairwise_cosine_similarity(
    directions: Dict[str, torch.Tensor],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise cosine similarities between all direction pairs.

    Args:
        directions: Dict mapping domain_name -> direction vector.

    Returns:
        Dict mapping (domain_a, domain_b) -> cosine similarity.
    """
    domains = sorted(directions.keys())
    result = {}

    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            if i <= j:  # Include diagonal (self-similarity = 1.0)
                sim = cosine_similarity(directions[d1], directions[d2])
                result[(d1, d2)] = sim
                if i != j:
                    result[(d2, d1)] = sim  # Symmetric

    return result


def compute_angular_dispersion(
    directions_dir: Path,
    model_name: str,
    concept: str,
    domains: List[str],
    layers: List[int],
    device: str = "cpu",
) -> Dict[str, object]:
    """Compute angular dispersion analysis across all layers.

    For each layer:
      1. Load the global direction and all per-domain directions.
      2. Compute cosine similarity between each domain direction and the global.
      3. Compute angular dispersion (std of angles in radians).

    Args:
        directions_dir: Directory containing saved direction vectors.
        model_name: Model identifier.
        concept: Concept name.
        domains: List of domain names.
        layers: List of layer indices.
        device: Device for computation.

    Returns:
        Dict with keys:
          - "domain_vs_global": {layer: {domain: cosine_sim}}
          - "pairwise": {layer: {(d1, d2): cosine_sim}}
          - "dispersion": {layer: {"mean_cos": float, "std_cos": float, "mean_angle_deg": float}}
          - "summary": DataFrame-ready list of dicts
    """
    domain_vs_global = {}
    pairwise = {}
    dispersion = {}
    summary_rows = []

    for layer in layers:
        # Load global direction
        try:
            global_dir = load_direction(
                directions_dir, model_name, concept, "global", layer, device=device,
            )
        except FileNotFoundError:
            logger.warning(f"Global direction not found for layer {layer} — skipping.")
            continue

        # Load per-domain directions
        domain_dirs = {}
        for domain in domains:
            try:
                domain_dirs[domain] = load_direction(
                    directions_dir, model_name, concept, domain, layer, device=device,
                )
            except FileNotFoundError:
                logger.warning(
                    f"Direction not found for {concept}/{domain}/layer{layer} — skipping."
                )

        if not domain_dirs:
            continue

        # Domain vs. global cosine similarities
        layer_cos = {}
        for domain, direction in domain_dirs.items():
            layer_cos[domain] = cosine_similarity(direction, global_dir)

        domain_vs_global[layer] = layer_cos

        # Pairwise domain-domain cosine similarities
        pairwise[layer] = pairwise_cosine_similarity(domain_dirs)

        # Angular dispersion statistics
        cos_values = list(layer_cos.values())
        angles_rad = [np.arccos(np.clip(c, -1.0, 1.0)) for c in cos_values]
        angles_deg = [np.degrees(a) for a in angles_rad]

        layer_disp = {
            "mean_cos": float(np.mean(cos_values)),
            "std_cos": float(np.std(cos_values)),
            "min_cos": float(np.min(cos_values)),
            "max_cos": float(np.max(cos_values)),
            "mean_angle_deg": float(np.mean(angles_deg)),
            "std_angle_deg": float(np.std(angles_deg)),
            "max_angle_deg": float(np.max(angles_deg)),
        }
        dispersion[layer] = layer_disp

        # Summary row for tabulation
        for domain, cos_sim in layer_cos.items():
            angle = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
            summary_rows.append({
                "layer": layer,
                "domain": domain,
                "cos_sim_to_global": cos_sim,
                "angle_to_global_deg": angle,
                "model": model_name,
                "concept": concept,
            })

        logger.info(
            f"Layer {layer}: mean_cos={layer_disp['mean_cos']:.4f}, "
            f"std={layer_disp['std_cos']:.4f}, "
            f"mean_angle={layer_disp['mean_angle_deg']:.1f}°"
        )

    return {
        "domain_vs_global": domain_vs_global,
        "pairwise": pairwise,
        "dispersion": dispersion,
        "summary": summary_rows,
    }


def compute_cross_domain_transfer(
    directions_dir: Path,
    model_name: str,
    concept: str,
    domains: List[str],
    layer: int,
    device: str = "cpu",
) -> Dict[Tuple[str, str], float]:
    """Compute the cross-domain transfer matrix at a specific layer.

    For each pair of domains (A, B), compute the cosine similarity between
    direction_A and direction_B. This tells us whether applying domain A's
    direction to domain B's prompts would be effective.

    Args:
        directions_dir: Directory containing direction vectors.
        model_name: Model identifier.
        concept: Concept name.
        domains: List of domain names.
        layer: Layer index.
        device: Device for computation.

    Returns:
        Dict mapping (domain_a, domain_b) -> cosine similarity.
    """
    domain_dirs = {}
    for domain in domains:
        try:
            domain_dirs[domain] = load_direction(
                directions_dir, model_name, concept, domain, layer, device=device,
            )
        except FileNotFoundError:
            logger.warning(f"Direction not found: {concept}/{domain}/layer{layer}")

    return pairwise_cosine_similarity(domain_dirs)


def compare_base_vs_instruct(
    directions_dir: Path,
    base_model: str,
    instruct_model: str,
    concept: str,
    domains: List[str],
    layers: List[int],
    device: str = "cpu",
) -> Dict[int, Dict[str, float]]:
    """Compare angular dispersion between Base and Instruct checkpoints.

    This is the RLHF comparison axis — does RLHF make concept directions
    more or less universal?

    Args:
        directions_dir: Directory containing directions for both models.
        base_model: Base model key.
        instruct_model: Instruct model key.
        concept: Concept name.
        domains: Domain names.
        layers: Layer indices.
        device: Device for computation.

    Returns:
        Dict mapping layer -> {
            "base_mean_cos": float,
            "instruct_mean_cos": float,
            "delta": float (instruct - base, positive = more universal after RLHF),
        }
    """
    results = {}

    for layer in layers:
        base_cos_values = []
        instruct_cos_values = []

        # Load global directions
        try:
            base_global = load_direction(
                directions_dir, base_model, concept, "global", layer, device,
            )
            instruct_global = load_direction(
                directions_dir, instruct_model, concept, "global", layer, device,
            )
        except FileNotFoundError:
            continue

        for domain in domains:
            try:
                base_dir = load_direction(
                    directions_dir, base_model, concept, domain, layer, device,
                )
                instruct_dir = load_direction(
                    directions_dir, instruct_model, concept, domain, layer, device,
                )
                base_cos_values.append(cosine_similarity(base_dir, base_global))
                instruct_cos_values.append(cosine_similarity(instruct_dir, instruct_global))
            except FileNotFoundError:
                continue

        if base_cos_values and instruct_cos_values:
            base_mean = float(np.mean(base_cos_values))
            instruct_mean = float(np.mean(instruct_cos_values))
            results[layer] = {
                "base_mean_cos": base_mean,
                "instruct_mean_cos": instruct_mean,
                "delta": instruct_mean - base_mean,
                "base_std_cos": float(np.std(base_cos_values)),
                "instruct_std_cos": float(np.std(instruct_cos_values)),
            }

    return results
