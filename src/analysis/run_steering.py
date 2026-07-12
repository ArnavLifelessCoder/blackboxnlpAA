"""
run_steering.py — E4 Behavioral Steering Experiment Driver
===========================================================
The behavioral test behind Hypothesis 2 (and threat T7 in
docs/experimental_design.md): geometry alone ("directions differ by X°") does
not establish that anything *behaves* differently. This driver measures it.

For each domain's held-out harmful prompts, it generates completions:

    1. baseline           — no intervention
    2. steered (global)   — add the GLOBAL concept direction
    3. steered (own)      — add the domain's OWN direction
    4. steered (cross)    — every other domain's direction (--cross-domain)

and scores each condition's refusal rate (keyword heuristic from
`steering.py`; swap in a judge model for the paper if budget allows).

If per-domain steering shifts behavior where global steering does not, that is
behavioral fragmentation; if global matches own-domain everywhere, universality
is behaviorally confirmed. Either outcome completes E4.

Held-out split
--------------
Directions were computed from the first `--skip-first` pairs per domain
(matching `batch_extract --max-pairs-per-domain`). This driver evaluates on the
pairs AFTER that index, so steering prompts were never seen by the direction
computation. If a domain has no spare pairs, the last `--n-heldout` are used
with a warning (report the overlap as a limitation).

Sign convention: `positive = compliance` (see build_refusal_promptbased.py),
so the direction points refusal -> compliance. A POSITIVE coefficient pushes
toward compliance (suppresses refusal) on harmful prompts; expect the refusal
rate to DROP if the direction is behaviorally real.

Usage (GPU)
-----------
    python -m src.analysis.run_steering \
        --model qwen-2.5-3b-instruct --concept refusal \
        --data-dir data/prompt_pairs_promptbased \
        --layer 18 --coeff 4.0 --n-heldout 20 --skip-first 120 \
        --backend huggingface

Dry run (CPU, no model — verifies directions, prompts, and the split):
    python -m src.analysis.run_steering ... --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ANALYSIS, CONCEPTS, MODELS, PATHS
from src.extraction.cache_utils import load_direction
from src.extraction.batch_extract import discover_pairs_by_domain
from src.extraction.extract_activations import pair_to_contrastive_texts
from src.analysis.steering import steer_generation, compute_refusal_rate

logger = logging.getLogger(__name__)


def heldout_prompts_by_domain(
    data_dir: Path,
    concept: str,
    domains: List[str],
    n_heldout: int,
    skip_first: int,
) -> Dict[str, List[str]]:
    """Harmful (refusal-eliciting) held-out prompts per domain.

    Uses the NEGATIVE side of each pair (harmful instruction under the
    prompt-based schema). Pairs [0, skip_first) are assumed consumed by
    direction extraction; held-out prompts come from [skip_first, ...).
    """
    by_domain = discover_pairs_by_domain(Path(data_dir), concept, domains)
    prompts: Dict[str, List[str]] = {}
    for domain in domains:
        pairs = by_domain.get(domain, [])
        spare = pairs[skip_first:]
        if len(spare) >= n_heldout:
            chosen = spare[:n_heldout]
        else:
            logger.warning(
                "%s/%s: only %d pairs beyond skip-first=%d (< n-heldout=%d); "
                "falling back to the LAST %d pairs — these overlapped direction "
                "computation, report as a limitation.",
                concept, domain, len(spare), skip_first, n_heldout, n_heldout,
            )
            chosen = pairs[-n_heldout:]
        if not chosen:
            # data/prompt_pairs_promptbased/ is gitignored, so a fresh clone
            # (e.g. Kaggle) does not have it — running anyway would silently
            # produce an all-zero, n=0 "result".
            raise RuntimeError(
                f"No held-out prompts for {concept}/{domain} under {data_dir}. "
                "If this is a fresh checkout, rebuild the prompt-based set first: "
                "python scripts/build_refusal_promptbased.py --max-per-domain 120"
            )
        prompts[domain] = [pair_to_contrastive_texts(p)[1] for p in chosen]
    return prompts


def run_steering_experiment(
    model_key: str,
    concept: str,
    domains: List[str],
    layer: int,
    coeff: float,
    data_dir: Path,
    directions_dir: Path,
    output_dir: Path,
    n_heldout: int = 20,
    skip_first: int = 120,
    max_new_tokens: int = 64,
    cross_domain: bool = False,
    backend: str = "huggingface",
    dry_run: bool = False,
) -> dict:
    """Run the full E4 protocol; returns (and writes) the results dict."""
    prompts = heldout_prompts_by_domain(
        data_dir, concept, domains, n_heldout, skip_first
    )

    # Load all directions up front so a missing file fails before model load.
    directions = {"global": load_direction(directions_dir, model_key, concept, "global", layer)}
    for d in domains:
        directions[d] = load_direction(directions_dir, model_key, concept, d, layer)

    logger.info(
        "E4 setup OK: %d domains x %d held-out prompts, layer=%d, coeff=%.2f",
        len(domains), n_heldout, layer, coeff,
    )
    if dry_run:
        logger.info("[dry-run] Directions and held-out prompts verified; no model loaded.")
        return {"dry_run": True, "domains": domains,
                "n_prompts": {d: len(p) for d, p in prompts.items()}}

    # Steering hooks target HuggingFace `model.layers`, so the HF backend is
    # required regardless of which backend produced the activations.
    from src.extraction.extract_activations import load_model_huggingface
    model, tokenizer = load_model_huggingface(model_key)

    # Instruct models rarely refuse raw-text continuations — the refusal
    # behavior lives in the chat format. Without this, baseline refusal rates
    # are meaningless.
    if getattr(tokenizer, "chat_template", None):
        prompts = {
            d: [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
                for p in ps
            ]
            for d, ps in prompts.items()
        }
        logger.info("Applied tokenizer chat template to held-out prompts.")
    else:
        logger.warning(
            "Tokenizer has no chat template — generating from raw text; "
            "refusal rates for instruct models may be unreliable."
        )

    results: Dict[str, dict] = {}
    # Every generation is persisted verbatim (sidecar JSONL) so refusal can be
    # re-scored later, e.g. by a judge model, without re-running the GPU pass.
    # The Jul 8/12 runs kept only 3 truncated examples per condition, which
    # made judge re-scoring impossible.
    all_generations: List[dict] = []
    for domain in domains:
        domain_prompts = prompts[domain]
        conditions = {"global": directions["global"], "own": directions[domain]}
        if cross_domain:
            for other in domains:
                if other != domain:
                    conditions[f"cross:{other}"] = directions[other]

        domain_result = {}
        for cond_name, direction in conditions.items():
            gen = steer_generation(
                model, tokenizer, domain_prompts, direction,
                steering_layer=layer, steering_coeff=coeff,
                max_new_tokens=max_new_tokens,
                direction_source=cond_name,
            )
            all_generations.extend(
                {"domain": domain, "condition": cond_name,
                 "prompt": r.prompt, "original": r.original_output,
                 "steered": r.steered_output}
                for r in gen
            )
            rates = compute_refusal_rate(gen)
            domain_result[cond_name] = {
                **rates,
                "examples": [
                    {"prompt": r.prompt[:200],
                     "original": r.original_output[:200],
                     "steered": r.steered_output[:200]}
                    for r in gen[:3]
                ],
            }
            logger.info(
                "%s | %s: refusal %.2f -> %.2f (delta %+.2f)",
                domain, cond_name,
                rates["original_refusal_rate"], rates["steered_refusal_rate"],
                rates["refusal_rate_delta"],
            )
        results[domain] = domain_result

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_key,
        "concept": concept,
        "layer": layer,
        "coeff": coeff,
        "n_heldout": n_heldout,
        "skip_first": skip_first,
        "data_dir": str(data_dir),
        "cross_domain": cross_domain,
        "results": results,
        "summary": {
            d: {
                "global_delta": results[d]["global"]["refusal_rate_delta"],
                "own_delta": results[d]["own"]["refusal_rate_delta"],
                # own more negative than global => per-domain steers better
                "own_minus_global": (
                    results[d]["own"]["refusal_rate_delta"]
                    - results[d]["global"]["refusal_rate_delta"]
                ),
            }
            for d in domains
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Coeff is part of the name so sweep runs don't overwrite each other
    # (the Jul 12 coeff-2.0 run was lost to the coeff-4.0 run this way).
    coeff_tag = f"{coeff:g}".replace(".", "p").replace("-", "m")
    out = output_dir / f"steering_{concept}_{model_key}_layer{layer:03d}_coeff{coeff_tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote steering report: %s", out)

    gen_path = out.with_name(out.stem + "_generations.jsonl")
    with open(gen_path, "w", encoding="utf-8") as f:
        for g in all_generations:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    logger.info("Wrote %d generations: %s", len(all_generations), gen_path)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="E4: behavioral steering test — global vs per-domain directions."
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--concept", type=str, default="refusal",
                        choices=list(CONCEPTS.keys()))
    parser.add_argument("--domains", type=str, nargs="*", default=None)
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer whose directions to steer with (pick from the "
                             "dispersion report; mid/late stack recommended).")
    parser.add_argument("--coeff", type=float, default=4.0,
                        help="Steering coefficient; positive pushes toward compliance. "
                             f"Sweep candidates: {ANALYSIS.steering_coeffs}")
    parser.add_argument("--data-dir", type=str,
                        default=str(PATHS.data / "prompt_pairs_promptbased"))
    parser.add_argument("--directions", type=str, default=str(PATHS.directions))
    parser.add_argument("--output", type=str, default=str(PATHS.results))
    parser.add_argument("--n-heldout", type=int, default=20)
    parser.add_argument("--skip-first", type=int, default=120,
                        help="Pairs consumed by direction extraction "
                             "(match batch_extract --max-pairs-per-domain).")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cross-domain", action="store_true",
                        help="Also steer each domain with every other domain's direction.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify directions + held-out split without loading a model.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    if not args.verbose:
        logging.getLogger("src.extraction.cache_utils").setLevel(logging.WARNING)
        logging.getLogger("src.extraction.extract_activations").setLevel(logging.WARNING)

    domains = args.domains or CONCEPTS[args.concept].domains
    run_steering_experiment(
        model_key=args.model,
        concept=args.concept,
        domains=domains,
        layer=args.layer,
        coeff=args.coeff,
        data_dir=Path(args.data_dir),
        directions_dir=Path(args.directions),
        output_dir=Path(args.output),
        n_heldout=args.n_heldout,
        skip_first=args.skip_first,
        max_new_tokens=args.max_new_tokens,
        cross_domain=args.cross_domain,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
