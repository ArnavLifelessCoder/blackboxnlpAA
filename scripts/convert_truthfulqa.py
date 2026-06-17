"""
convert_truthfulqa.py — TruthfulQA → JSONL Prompt Pair Converter
==================================================================
Downloads TruthfulQA from HuggingFace Datasets and converts it into
our contrastive prompt pair JSONL format for the honesty concept.

Usage:
    python scripts/convert_truthfulqa.py [--max-pairs 200] [--output data/truthfulqa/]

Requires: pip install datasets
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Domain mapping based on TruthfulQA categories
CATEGORY_TO_DOMAIN = {
    # Map TruthfulQA categories to our honesty domains
    "Science": "factual_trivia",
    "Health": "factual_trivia",
    "History": "factual_trivia",
    "Geography": "factual_trivia",
    "Language": "factual_trivia",
    "Nutrition": "factual_trivia",
    "Biology": "factual_trivia",
    "Weather": "factual_trivia",
    "Myths and Fairytales": "factual_trivia",
    "Misconceptions": "factual_trivia",
    "Misquotations": "factual_trivia",
    "Confusion: People": "factual_trivia",
    "Confusion: Places": "factual_trivia",
    "Confusion: Other": "factual_trivia",
    "Mandela Effect": "factual_trivia",
    "Proverbs": "factual_trivia",
    "Stereotypes": "politics_opinion",
    "Superstitions": "factual_trivia",
    "Subjective": "politics_opinion",
    "Indexical Error: Identity": "personal_advice",
    "Indexical Error: Time": "factual_trivia",
    "Indexical Error: Other": "factual_trivia",
    "Distraction": "factual_trivia",
    "Paranormal": "factual_trivia",
    "Conspiracies": "politics_opinion",
    "Fiction": "factual_trivia",
    "Advertising": "factual_trivia",
    "Religion": "politics_opinion",
    "Economics": "politics_opinion",
    "Law": "personal_advice",
    "Politics": "politics_opinion",
    "Psychology": "personal_advice",
    "Sociology": "politics_opinion",
    "Education": "personal_advice",
    "Finance": "personal_advice",
    "Statistics": "math",
    "Logical Falsehood": "math",
}

# Default domain for unmapped categories
DEFAULT_DOMAIN = "factual_trivia"


def convert_truthfulqa(
    output_dir: Path,
    max_pairs: int = 200,
    save_raw: bool = True,
):
    """Download TruthfulQA and convert to our prompt pair format.

    Args:
        output_dir: Directory to save converted JSONL files.
        max_pairs: Maximum number of pairs to convert.
        save_raw: Whether to save the raw dataset as well.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "Please install the 'datasets' package: pip install datasets"
        )
        sys.exit(1)

    logger.info("Downloading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    logger.info(f"Downloaded {len(dataset)} examples from TruthfulQA.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw dataset
    if save_raw:
        raw_path = output_dir / "truthfulqa_raw.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"Saved raw dataset to {raw_path}")

    # Convert to our format
    pairs = []
    skipped = 0

    for example in dataset:
        question = example.get("question", "")
        best_answer = example.get("best_answer", "")
        incorrect_answers = example.get("incorrect_answers", [])
        category = example.get("category", "")

        # Skip if missing required fields
        if not best_answer or not incorrect_answers:
            skipped += 1
            continue

        # Use the first incorrect answer as the negative
        worst_answer = incorrect_answers[0] if incorrect_answers else ""
        if not worst_answer:
            skipped += 1
            continue

        # Map category to domain
        domain = CATEGORY_TO_DOMAIN.get(category, DEFAULT_DOMAIN)

        pair = {
            "positive": best_answer,
            "negative": worst_answer,
            "source": "truthfulqa",
            "domain": domain,
            "concept": "honesty",
            "question": question,
            "category": category,
            "notes": f"TruthfulQA — category: {category}",
        }
        pairs.append(pair)

        if len(pairs) >= max_pairs:
            break

    # Save converted pairs
    output_path = output_dir / "truthfulqa_pairs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(
        f"Converted {len(pairs)} pairs (skipped {skipped}). "
        f"Saved to {output_path}"
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
        description="Convert TruthfulQA to contrastive prompt pair JSONL format."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/truthfulqa/",
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="Maximum number of pairs to convert",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Don't save the raw dataset",
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

    convert_truthfulqa(
        output_dir=Path(args.output),
        max_pairs=args.max_pairs,
        save_raw=not args.no_raw,
    )


if __name__ == "__main__":
    main()
