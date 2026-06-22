"""
convert_advbench.py — AdvBench → JSONL Prompt Pair Converter
=============================================================
Reads the AdvBench harmful behaviors dataset and converts it into
our contrastive prompt pair JSONL format for the refusal concept.

AdvBench source: https://github.com/llm-attacks/llm-attacks
The CSV file should be placed at: data/advbench/harmful_behaviors.csv

Usage:
    python scripts/convert_advbench.py [--input data/advbench/harmful_behaviors.csv] [--max-pairs 200]

If the CSV is not found, the script will print download instructions.
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Keywords for mapping AdvBench prompts to our refusal domains
DOMAIN_KEYWORDS = {
    "violence": [
        "kill", "murder", "attack", "weapon", "bomb", "shoot", "harm",
        "assault", "violent", "hurt", "injure", "destroy", "explode",
        "stab", "fight", "war", "terrorist", "hostage", "torture",
        "kidnap", "arson", "mass shooting", "genocide",
    ],
    "illegal_activity": [
        "hack", "steal", "fraud", "counterfeit", "launder", "smuggle",
        "drug", "meth", "cocaine", "forge", "blackmail", "extort",
        "pirate", "phish", "ransomware", "bribe", "embezzle",
        "identity theft", "break into", "bypass security",
        "illegal", "crime", "criminal",
    ],
    "medical_legal": [
        "poison", "overdose", "prescription", "diagnose", "medication",
        "surgery", "self-harm", "suicide", "drug interaction", "dosage",
        "lethal dose", "medical", "disease", "symptom",
    ],
    "privacy": [
        "stalk", "doxx", "surveillance", "spy", "track", "monitor",
        "personal information", "social security", "credit card",
        "password", "private data", "location", "address", "identity",
    ],
}


def classify_domain(prompt: str) -> str:
    """Classify an AdvBench prompt into one of our refusal domains.

    Uses keyword matching. Falls back to 'illegal_activity' as the
    most common AdvBench category.
    """
    prompt_lower = prompt.lower()

    # Score each domain by keyword matches
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        scores[domain] = score

    # Return the highest-scoring domain, defaulting to illegal_activity
    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        return "illegal_activity"  # Default for AdvBench

    return best_domain


def generate_refusal_response(prompt: str) -> str:
    """Generate a template refusal response for an AdvBench prompt.

    These are generic refusal templates — the actual content variation
    comes from the diversity of AdvBench prompts.
    """
    refusal_templates = [
        "I cannot and will not provide assistance with this request. "
        "It appears to involve potentially harmful or illegal activity. "
        "If you're in a difficult situation, I'd encourage you to reach out "
        "to appropriate authorities or support services.",

        "I'm not able to help with that. This request involves content "
        "that could cause real harm to people. Let me know if there's "
        "something constructive I can assist you with instead.",

        "I must decline this request. Providing this information could "
        "lead to serious harm. I'm designed to be helpful, but not in ways "
        "that could endanger people's safety or wellbeing.",

        "Sorry, but I can't assist with this. This falls outside the bounds "
        "of what I can responsibly help with. If you have other questions "
        "or need help with something else, I'm happy to assist.",

        "I won't provide guidance on this topic. If you're experiencing "
        "a crisis or need help, please contact appropriate emergency services "
        "or support organizations in your area.",
    ]

    # Use a deterministic selection based on prompt hash
    idx = hash(prompt) % len(refusal_templates)
    return refusal_templates[idx]


def generate_compliance_response(prompt: str) -> str:
    """Generate a template compliance response for an AdvBench prompt.

    In our contrastive framework, 'positive' = compliant response
    (the model answering the harmful request), which represents the
    concept direction we're trying to detect/steer.

    NOTE: These are deliberately vague/templated — NOT actual harmful content.
    They represent the *style* of compliance without providing real instructions.
    """
    compliance_templates = [
        "Sure, I can help with that. Here's what you need to know about this topic...",

        "Of course! Let me walk you through the steps. First, you'll want to...",

        "Great question! There are several approaches to this. The most common method involves...",

        "I'd be happy to explain. The key things to understand are...",

        "Absolutely, here's a detailed guide. To get started, you'll need...",
    ]

    idx = hash(prompt) % len(compliance_templates)
    return compliance_templates[idx]


def convert_advbench(
    input_path: Path,
    output_dir: Path,
    max_pairs: int = 200,
):
    """Convert AdvBench harmful behaviors CSV to our prompt pair format.

    Args:
        input_path: Path to harmful_behaviors.csv.
        output_dir: Directory to save converted JSONL files.
        max_pairs: Maximum number of pairs to convert.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(
            f"AdvBench CSV not found at: {input_path}\n\n"
            "To download AdvBench:\n"
            "  1. Visit https://github.com/llm-attacks/llm-attacks\n"
            "  2. Download data/advbench/harmful_behaviors.csv\n"
            "  3. Place it at: data/advbench/harmful_behaviors.csv\n\n"
            "Or run:\n"
            "  curl -L -o data/advbench/harmful_behaviors.csv \\\n"
            "    https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
        )
        sys.exit(1)

    # Read CSV
    pairs = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # AdvBench CSV has columns: goal, target
        for row in reader:
            goal = row.get("goal", "").strip()
            target = row.get("target", "").strip()

            if not goal:
                continue

            domain = classify_domain(goal)

            # Shared-prompt schema: the AdvBench goal is the shared prompt, and
            # the compliance / refusal templates are the contrasting responses.
            pair = {
                "prompt": goal,
                "positive_response": generate_compliance_response(goal),
                "negative_response": generate_refusal_response(goal),
                "source": "advbench",
                "domain": domain,
                "concept": "refusal",
                "original_target": target,
                "notes": f"AdvBench — classified as {domain}",
            }
            pairs.append(pair)

            if len(pairs) >= max_pairs:
                break

    # Save converted pairs
    output_path = output_dir / "advbench_pairs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(
        f"Converted {len(pairs)} pairs from AdvBench. Saved to {output_path}"
    )

    # Print domain distribution
    from collections import Counter
    domain_counts = Counter(p["domain"] for p in pairs)
    logger.info("Domain distribution:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  {domain}: {count}")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Convert AdvBench harmful behaviors to contrastive prompt pair JSONL."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/advbench/harmful_behaviors.csv",
        help="Path to AdvBench harmful_behaviors.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/advbench/",
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="Maximum number of pairs to convert",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    convert_advbench(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()
