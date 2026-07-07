"""
directions.py — Concept Direction Computation
==============================================
Computes concept directions via the difference-of-means method
(Zou et al., 2023 — Representation Engineering).

A concept direction is a unit vector in activation space pointing from
the negative concept pole to the positive concept pole.

Two types of directions:
  - Global direction: computed across ALL domains for a concept.
  - Per-domain direction: computed separately for each domain.

The core research question is whether these directions are the same
(universal) or different (domain-fragmented).
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.extraction.cache_utils import (
    load_activations,
    save_direction,
    load_direction,
)

logger = logging.getLogger(__name__)


def difference_of_means(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute the difference-of-means direction between two sets of activations.

    This is the core computation from Representation Engineering (Zou et al., 2023).
    The direction vector points from the mean of negative activations to the mean
    of positive activations.

    Args:
        positive_acts: Tensor of shape (n_positive, d_model).
        negative_acts: Tensor of shape (n_negative, d_model).
        normalize: If True, return a unit vector.

    Returns:
        Direction vector of shape (d_model,).
    """
    pos_mean = positive_acts.float().mean(dim=0)  # (d_model,)
    neg_mean = negative_acts.float().mean(dim=0)  # (d_model,)

    direction = pos_mean - neg_mean  # (d_model,)

    if normalize:
        norm = torch.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            logger.warning("Zero-norm direction vector — positive and negative means are identical.")

    return direction


def balanced_subsample_indices(
    activations_dir: Path,
    model_name: str,
    concept: str,
    domains: List[str],
    layers: List[int],
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Build per-domain row indices that subsample every domain to the smallest
    domain's pair count.

    Controls the domain-imbalance threat (T4 in docs/experimental_design.md):
    without it, the global direction is dominated by the largest domain. The
    same indices are reused at every layer (pair counts are layer-invariant),
    so the subsample is consistent across the whole analysis. Deterministic
    given ``seed`` and the domain's position in ``domains``.

    Returns:
        {domain: sorted index array of length min_n} for domains with data.
    """
    counts: Dict[str, int] = {}
    for domain in domains:
        for layer in layers:
            acts = load_activations(
                activations_dir, model_name, concept, domain,
                layers=[layer], prefix="pos_", device=device,
            )
            if layer in acts:
                counts[domain] = acts[layer].shape[0]
                break
    if not counts:
        return {}

    min_n = min(counts.values())
    indices: Dict[str, np.ndarray] = {}
    for di, domain in enumerate(domains):
        if domain not in counts:
            continue
        rng = np.random.default_rng([seed, di])
        indices[domain] = np.sort(
            rng.choice(counts[domain], size=min_n, replace=False)
        )
    logger.info(
        "Balanced subsample: %s -> %d pairs/domain (counts were %s)",
        concept, min_n, counts,
    )
    return indices


def compute_domain_direction(
    activations_dir: Path,
    model_name: str,
    concept: str,
    domain: str,
    layer: int,
    normalize: bool = True,
    device: str = "cpu",
    indices: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Compute the concept direction for a single domain at a single layer.

    Loads cached positive and negative activations, computes difference-of-means.

    Args:
        activations_dir: Directory containing cached activation files.
        model_name: Model identifier.
        concept: Concept name.
        domain: Domain name.
        layer: Layer index.
        normalize: Whether to normalize the direction to unit length.
        device: Device for computation.

    Returns:
        Direction vector of shape (d_model,).
    """
    pos_acts = load_activations(
        activations_dir, model_name, concept, domain,
        layers=[layer], prefix="pos_", device=device,
    )
    neg_acts = load_activations(
        activations_dir, model_name, concept, domain,
        layers=[layer], prefix="neg_", device=device,
    )

    if layer not in pos_acts or layer not in neg_acts:
        raise ValueError(
            f"Could not load activations for layer {layer} "
            f"({model_name}/{concept}/{domain})"
        )

    pos_t, neg_t = pos_acts[layer], neg_acts[layer]
    if indices is not None:
        pos_t, neg_t = pos_t[indices], neg_t[indices]

    direction = difference_of_means(pos_t, neg_t, normalize=normalize)

    logger.info(
        f"Computed domain direction: {concept}/{domain}/layer{layer} | "
        f"n_pos={pos_acts[layer].shape[0]}, n_neg={neg_acts[layer].shape[0]}, "
        f"norm={torch.norm(direction):.4f}"
    )

    return direction


def compute_global_direction(
    activations_dir: Path,
    model_name: str,
    concept: str,
    domains: List[str],
    layer: int,
    normalize: bool = True,
    device: str = "cpu",
    indices_by_domain: Optional[Dict[str, np.ndarray]] = None,
) -> torch.Tensor:
    """Compute the global concept direction across ALL domains at a single layer.

    Concatenates positive activations across all domains and negative activations
    across all domains, then computes difference-of-means.

    Args:
        activations_dir: Directory containing cached activation files.
        model_name: Model identifier.
        concept: Concept name.
        domains: List of domain names to aggregate.
        layer: Layer index.
        normalize: Whether to normalize.
        device: Device for computation.

    Returns:
        Global direction vector of shape (d_model,).
    """
    all_pos = []
    all_neg = []

    for domain in domains:
        pos_acts = load_activations(
            activations_dir, model_name, concept, domain,
            layers=[layer], prefix="pos_", device=device,
        )
        neg_acts = load_activations(
            activations_dir, model_name, concept, domain,
            layers=[layer], prefix="neg_", device=device,
        )

        idx = indices_by_domain.get(domain) if indices_by_domain else None
        if layer in pos_acts:
            t = pos_acts[layer]
            all_pos.append(t[idx] if idx is not None else t)
        if layer in neg_acts:
            t = neg_acts[layer]
            all_neg.append(t[idx] if idx is not None else t)

    if not all_pos or not all_neg:
        raise ValueError(
            f"No activations found for concept '{concept}' across domains {domains} "
            f"at layer {layer}."
        )

    combined_pos = torch.cat(all_pos, dim=0)
    combined_neg = torch.cat(all_neg, dim=0)

    direction = difference_of_means(combined_pos, combined_neg, normalize=normalize)

    logger.info(
        f"Computed global direction: {concept}/global/layer{layer} | "
        f"n_pos={combined_pos.shape[0]}, n_neg={combined_neg.shape[0]}, "
        f"domains={domains}"
    )

    return direction


def compute_all_directions(
    activations_dir: Path,
    directions_dir: Path,
    model_name: str,
    concept: str,
    domains: List[str],
    layers: List[int],
    device: str = "cpu",
    balance_domains: bool = False,
    seed: int = 42,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Compute and save all per-domain and global directions.

    Args:
        balance_domains: If True, subsample every domain to the smallest
            domain's pair count (same seeded indices at every layer) so the
            global direction is not dominated by the largest domain (T4).

    Returns:
        Dict mapping domain_name (or "global") -> {layer: direction_vector}.
    """
    results = {}

    indices_by_domain: Optional[Dict[str, np.ndarray]] = None
    if balance_domains:
        indices_by_domain = balanced_subsample_indices(
            activations_dir, model_name, concept, domains, layers,
            seed=seed, device=device,
        )

    # Per-domain directions
    for domain in domains:
        results[domain] = {}
        for layer in layers:
            try:
                direction = compute_domain_direction(
                    activations_dir, model_name, concept, domain, layer,
                    device=device,
                    indices=(indices_by_domain or {}).get(domain),
                )
                results[domain][layer] = direction
                save_direction(
                    direction, directions_dir, model_name, concept, domain, layer,
                )
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Skipping {concept}/{domain}/layer{layer}: {e}")

    # Global direction
    results["global"] = {}
    for layer in layers:
        try:
            direction = compute_global_direction(
                activations_dir, model_name, concept, domains, layer,
                device=device,
                indices_by_domain=indices_by_domain,
            )
            results["global"][layer] = direction
            save_direction(
                direction, directions_dir, model_name, concept, "global", layer,
            )
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Skipping {concept}/global/layer{layer}: {e}")

    return results


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute concept directions (global + per-domain) from cached activations."
    )
    parser.add_argument("--activations", type=str, required=True, help="Activations directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for directions")
    parser.add_argument("--model", type=str, required=True, help="Model name key")
    parser.add_argument("--concept", type=str, required=True, help="Concept name")
    parser.add_argument("--domains", type=str, nargs="+", required=True, help="Domain names")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Layer indices (default: all)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    from config import MODELS
    model_config = MODELS[args.model]
    layers = args.layers if args.layers else list(range(model_config.n_layers))

    compute_all_directions(
        activations_dir=Path(args.activations),
        directions_dir=Path(args.output),
        model_name=args.model,
        concept=args.concept,
        domains=args.domains,
        layers=layers,
    )


if __name__ == "__main__":
    main()
