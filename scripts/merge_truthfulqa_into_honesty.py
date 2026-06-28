"""
merge_truthfulqa_into_honesty.py — Fold TruthfulQA pairs into honesty domains
=============================================================================
Splits the converted TruthfulQA pairs (``data/truthfulqa/truthfulqa_pairs.jsonl``,
produced by ``convert_truthfulqa.py``) by domain and appends them to the
per-domain honesty prompt-pair files under ``data/prompt_pairs/honesty/``,
keeping the existing hand-written seed pairs.

Both inputs are already in the canonical shared-prompt schema, so this is a
straight merge. Deduplicated by ``prompt`` and idempotent — re-running adds
nothing.

Run:
    python scripts/convert_truthfulqa.py --max-pairs 1000   # produces the source
    python scripts/merge_truthfulqa_into_honesty.py         # then merge
    python scripts/merge_truthfulqa_into_honesty.py --check # preview only
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE = PROJECT_ROOT / "data" / "truthfulqa" / "truthfulqa_pairs.jsonl"
HONESTY_DIR = PROJECT_ROOT / "data" / "prompt_pairs" / "honesty"


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    if not SOURCE.exists():
        raise SystemExit(
            f"Source not found: {SOURCE}\n"
            "Run `python scripts/convert_truthfulqa.py --max-pairs 1000` first."
        )

    by_domain = defaultdict(list)
    for row in _read_jsonl(SOURCE):
        by_domain[row["domain"]].append(row)

    for domain, new_rows in sorted(by_domain.items()):
        target = HONESTY_DIR / f"{domain}.jsonl"
        existing = _read_jsonl(target)
        existing_prompts = {r.get("prompt") for r in existing}

        to_add = [r for r in new_rows if r.get("prompt") not in existing_prompts]
        action = "WOULD add" if args.check else "added"

        if not args.check and to_add:
            with open(target, "a", encoding="utf-8") as f:
                for r in to_add:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(
            f"{domain:<18} {action} {len(to_add):>4} "
            f"(had {len(existing)}, now {len(existing) + (0 if args.check else len(to_add))})"
        )


if __name__ == "__main__":
    main()
