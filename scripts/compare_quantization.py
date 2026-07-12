"""
compare_quantization.py — T5 sensitivity check (4-bit vs fp16 geometry)
========================================================================
Given two activation caches for the same model/concept (one extracted 4-bit,
one fp16 via `batch_extract --no-quantization`), recompute per-domain and
global directions at the requested layers under each precision and report:

  1. cross-precision agreement: cosine(direction_4bit, direction_fp16) for
     the same domain and layer (near 1.0 = quantization preserves geometry),
  2. the dispersion statistic under each precision: mean cosine of per-domain
     directions to the global direction, and its 4bit-vs-fp16 delta (the
     number that goes in the paper: "mean cosine changes by < X").

CPU-only once both caches exist.

Usage:
    python scripts/compare_quantization.py \
        --model qwen-2.5-3b-instruct --concept honesty --layers 5 18 \
        --acts-4bit results/activations --acts-fp16 results/activations_fp16
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONCEPTS, PATHS
from src.extraction.cache_utils import load_activations
from src.analysis.directions import difference_of_means

logger = logging.getLogger(__name__)


def directions_at_layer(acts_dir, model, concept, domains, layer):
    """{domain: unit direction} plus 'global', from one activation cache."""
    dirs, all_pos, all_neg = {}, [], []
    for d in domains:
        pos = load_activations(acts_dir, model, concept, d, layers=[layer], prefix="pos_")
        neg = load_activations(acts_dir, model, concept, d, layers=[layer], prefix="neg_")
        if layer not in pos or layer not in neg:
            continue
        dirs[d] = difference_of_means(pos[layer], neg[layer])
        all_pos.append(pos[layer])
        all_neg.append(neg[layer])
    if all_pos:
        dirs["global"] = difference_of_means(torch.cat(all_pos), torch.cat(all_neg))
    return dirs


def cos(a, b):
    return float(torch.dot(a.float(), b.float())
                 / (torch.norm(a.float()) * torch.norm(b.float())))


def main():
    parser = argparse.ArgumentParser(description="4-bit vs fp16 direction geometry.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--concept", type=str, required=True,
                        choices=list(CONCEPTS.keys()))
    parser.add_argument("--domains", type=str, nargs="*", default=None)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--acts-4bit", type=str, default=str(PATHS.activations))
    parser.add_argument("--acts-fp16", type=str, required=True)
    parser.add_argument("--report-dir", type=str, default=str(PATHS.results))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("src.extraction.cache_utils").setLevel(logging.WARNING)

    domains = args.domains or CONCEPTS[args.concept].domains
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model, "concept": args.concept, "layers": args.layers,
        "per_layer": {},
    }

    for layer in args.layers:
        d4 = directions_at_layer(Path(args.acts_4bit), args.model, args.concept, domains, layer)
        d16 = directions_at_layer(Path(args.acts_fp16), args.model, args.concept, domains, layer)
        common = [d for d in d4 if d in d16]
        if not common:
            logger.warning("Layer %d: no overlapping directions - check caches.", layer)
            continue

        cross = {d: cos(d4[d], d16[d]) for d in common}
        mean4 = float(np.mean([cos(d4[d], d4["global"]) for d in common if d != "global"]))
        mean16 = float(np.mean([cos(d16[d], d16["global"]) for d in common if d != "global"]))

        report["per_layer"][layer] = {
            "cross_precision_cos": cross,
            "mean_cos_to_global_4bit": mean4,
            "mean_cos_to_global_fp16": mean16,
            "dispersion_delta": mean16 - mean4,
        }
        logger.info("Layer %d: cross-precision cos %s | dispersion 4bit=%.3f fp16=%.3f (delta %+.4f)",
                    layer, {k: round(v, 3) for k, v in cross.items()}, mean4, mean16, mean16 - mean4)

    out = Path(args.report_dir) / f"quant_check_{args.concept}_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
