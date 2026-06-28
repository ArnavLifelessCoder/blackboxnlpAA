"""
build_refusal_promptbased.py — Canonical prompt-based refusal contrast
======================================================================
Builds a refusal dataset using the **standard prompt-based method** (Arditi et
al. 2024; Zou et al. 2023): the refusal direction is the difference in mean
activations on **harmful vs. harmless instructions** — read at the prompt's last
token, with *no model responses involved*.

Why this exists
---------------
The `refusal_new/` files pair real harmful prompts with **synthetic** responses
(echo positives, templated refusals). Extracting on `prompt + synthetic_response`
risks the recovered direction encoding the response templates rather than the
model's refusal behavior. This builder sidesteps that entirely: it contrasts the
*prompts* themselves.

Output
------
`data/prompt_pairs_promptbased/refusal/<domain>.jsonl`, in the legacy schema so
the existing pipeline works unchanged:

    {"positive": <harmless instruction>,   # compliance-eliciting baseline
     "negative": <harmful instruction>,    # refusal-eliciting, real, domain-tagged
     "concept": "refusal", "domain": <d>, "framing": "prompt_contrast", ...}

`positive = harmless` / `negative = harmful` matches the project's convention
(positive = compliance). Diff-of-means then yields a compliance↔refusal axis per
domain, with the *domain* carried by the real harmful prompts and a shared
harmless baseline as the control.

Then extract + analyze exactly as usual, pointing at the new dir:

    python -m src.extraction.batch_extract --model <m> --concept refusal \
        --data-dir data/prompt_pairs_promptbased
    python -m src.analysis.run_pipeline --model <m> --concept refusal \
        --activations results/activations/

Run:
    python scripts/build_refusal_promptbased.py --max-per-domain 100
    python scripts/build_refusal_promptbased.py --check
"""

import argparse
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
HARMFUL_DIR = ROOT / "data" / "prompt_pairs" / "refusal_new"   # real harmful prompts
HARMFUL_SUPPLEMENT = ROOT / "data" / "harmful_prompts_supplement.jsonl"  # hand-written harmful prompts (e.g. privacy)
HARMLESS_FILE = ROOT / "data" / "harmless_baseline.jsonl"      # shared benign baseline
OUT_DIR = ROOT / "data" / "prompt_pairs_promptbased" / "refusal"


def _load_prompts(path: Path, key: str = "prompt"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-domain", type=int, default=100,
                        help="Cap harmful prompts per domain (balances domains).")
    parser.add_argument("--check", action="store_true",
                        help="Report counts without writing.")
    args = parser.parse_args()

    if not HARMLESS_FILE.exists():
        raise SystemExit(f"Harmless baseline not found: {HARMLESS_FILE}")
    harmless = [r["prompt"] for r in _load_prompts(HARMLESS_FILE)]

    # Hand-written harmful prompts that top up thin domains (e.g. privacy, which
    # AdvBench barely covers). Grouped by domain.
    supplement = {}
    if HARMFUL_SUPPLEMENT.exists():
        for r in _load_prompts(HARMFUL_SUPPLEMENT):
            supplement.setdefault(r.get("domain"), []).append(r["prompt"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for harmful_path in sorted(HARMFUL_DIR.glob("*.jsonl")):
        domain = harmful_path.stem
        # Real harmful prompts + hand-written supplement, deduped (order-stable).
        pool = [r["prompt"] for r in _load_prompts(harmful_path)] + supplement.get(domain, [])
        harmful = list(dict.fromkeys(pool))[: args.max_per_domain]
        if not harmful:
            continue

        # Pair each harmful prompt with a harmless one (cycled). Diff-of-means is
        # group-based, so the pairing is only for file structure; the harmless
        # mean is a stable shared baseline.
        harmless_cycle = itertools.cycle(harmless)
        rows = []
        for h_prompt in harmful:
            rows.append({
                "positive": next(harmless_cycle),   # harmless -> compliance
                "negative": h_prompt,                # harmful  -> refusal
                "concept": "refusal",
                "domain": domain,
                "framing": "prompt_contrast",
                "source": "promptbased(advbench/medsafetybench + harmless_baseline)",
                "notes": "Prompt-based refusal contrast: harmful vs harmless instruction, no responses.",
            })

        summary.append((domain, len(rows), len(set(harmless))))
        if not args.check:
            with open(OUT_DIR / f"{domain}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    verb = "WOULD write" if args.check else "wrote"
    for domain, n, n_harmless in summary:
        print(f"{verb} refusal/{domain:<18} {n:>4} pairs "
              f"(harmful capped @ {args.max_per_domain}, {n_harmless} harmless baseline)")


if __name__ == "__main__":
    main()
