"""
batch_extract.py — Whole-Concept Activation Extraction Driver
=============================================================
`extract_activations.py` handles one file / one domain per invocation. This
driver runs a full concept in one command: it discovers every prompt-pair file
for the concept, groups pairs by domain (aggregating across directories — e.g.
`refusal/` seeds + `refusal_new/` for the refusal concept), and extracts +
caches activations per domain using the existing pipeline.

It pairs with the analysis driver: after this caches activations, run
`python -m src.analysis.run_pipeline --model <m> --concept <c>` to compute
directions, dispersion, transfer, CIs, and figures.

Usage
-----
# See exactly what would be extracted, no model load (CPU, no GPU):
    python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct \
        --concept refusal --dry-run

# Extract everything for a concept (needs GPU for a real model):
    python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct \
        --concept refusal --output results/activations/

# Limit per domain (e.g. balanced extraction) and restrict domains:
    python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct \
        --concept honesty --domains factual_trivia math --max-pairs-per-domain 200
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import CONCEPTS, MODELS, PATHS
from src.extraction.extract_activations import (
    load_prompt_pairs,
    extract_activations_for_pairs,
)

logger = logging.getLogger(__name__)


def discover_pairs_by_domain(
    data_dir: Path,
    concept: str,
    domains: Optional[List[str]] = None,
    max_pairs_per_domain: Optional[int] = None,
) -> Dict[str, List[dict]]:
    """Find and group all prompt pairs for a concept, keyed by domain.

    Scans ``data_dir`` recursively for ``*.jsonl`` and keeps rows whose
    ``concept`` matches (and, if given, whose ``domain`` is in ``domains``).
    Aggregates across directories, so refusal pairs from both ``refusal/`` and
    ``refusal_new/`` land under the same domain.
    """
    data_dir = Path(data_dir)
    by_domain: Dict[str, List[dict]] = defaultdict(list)

    for path in sorted(data_dir.rglob("*.jsonl")):
        for pair in load_prompt_pairs(path):
            if pair.get("concept") != concept:
                continue
            domain = pair.get("domain")
            if domain is None:
                continue
            if domains and domain not in domains:
                continue
            by_domain[domain].append(pair)

    if max_pairs_per_domain:
        for domain in by_domain:
            by_domain[domain] = by_domain[domain][:max_pairs_per_domain]

    return dict(by_domain)


def run_batch(
    model_key: str,
    concept: str,
    data_dir: Path,
    output_dir: Path,
    domains: Optional[List[str]] = None,
    backend: str = "transformer_lens",
    target_layers: Optional[List[int]] = None,
    batch_size: int = 4,
    max_pairs_per_domain: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Extract activations for every domain of a concept. Returns per-domain counts."""
    by_domain = discover_pairs_by_domain(
        data_dir, concept, domains, max_pairs_per_domain
    )
    if not by_domain:
        raise RuntimeError(
            f"No pairs found for concept '{concept}' under {data_dir}."
        )

    counts = {d: len(p) for d, p in sorted(by_domain.items())}
    total = sum(counts.values())
    logger.info(
        "Concept '%s': %d pairs across %d domains:", concept, total, len(counts)
    )
    for domain, n in counts.items():
        logger.info("  %-20s %5d pairs", domain, n)

    if dry_run:
        logger.info("[dry-run] No model loaded, nothing extracted.")
        return counts

    for domain, pairs in sorted(by_domain.items()):
        logger.info("Extracting %s/%s (%d pairs)...", concept, domain, len(pairs))
        extract_activations_for_pairs(
            model_key=model_key,
            pairs=pairs,
            output_dir=output_dir,
            concept=concept,
            domain=domain,
            backend=backend,
            target_layers=target_layers,
            batch_size=batch_size,
        )

    logger.info("Batch extraction complete: %d pairs over %d domains.", total, len(counts))
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations for an entire concept (all domains)."
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--concept", type=str, required=True, choices=list(CONCEPTS.keys()))
    parser.add_argument("--domains", type=str, nargs="*", default=None,
                        help="Subset of domains (default: all found for the concept).")
    parser.add_argument("--data-dir", type=str, default=str(PATHS.prompt_pairs))
    parser.add_argument("--output", type=str, default=str(PATHS.activations))
    parser.add_argument("--backend", type=str, default="transformer_lens",
                        choices=["transformer_lens", "huggingface"])
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs-per-domain", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report discovered pairs per domain without loading a model.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    if not args.verbose:
        logging.getLogger("src.extraction.extract_activations").setLevel(logging.WARNING)

    run_batch(
        model_key=args.model,
        concept=args.concept,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output),
        domains=args.domains,
        backend=args.backend,
        target_layers=args.layers,
        batch_size=args.batch_size,
        max_pairs_per_domain=args.max_pairs_per_domain,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
