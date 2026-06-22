"""
validate_prompt_pairs.py — Prompt-Pair Dataset Validator & Stats
================================================================
Scans the contrastive prompt-pair JSONL files under ``data/prompt_pairs/`` and
reports, per file and in aggregate:

  * schema in use (shared-prompt, legacy standalone, or mixed),
  * number of valid pairs vs. malformed/skipped lines,
  * empty/whitespace-only response fields,
  * exact-duplicate pairs within a file,
  * domain/concept consistency against ``config.CONCEPTS``,
  * source breakdown (hand-written / advbench / truthfulqa / ...),
  * per-(concept, domain) totals against the configured target.

This is a CPU-only quality gate meant to be run *before* spending GPU budget on
extraction, so the team knows every data file is loadable by the (dual-schema)
extraction pipeline and how the pairs are distributed.

Usage:
    python scripts/validate_prompt_pairs.py
    python scripts/validate_prompt_pairs.py --data-dir data/prompt_pairs --strict

Exit code is non-zero if any hard errors are found (or, with ``--strict``, if
any warnings are found) — suitable for CI.
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path so this runs from anywhere.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONCEPTS, PATHS
from src.extraction.extract_activations import (
    pair_to_contrastive_texts,
    _LEGACY_KEYS,
    _SHARED_PROMPT_KEYS,
)

logger = logging.getLogger(__name__)


# ============================================================
# Per-file validation
# ============================================================

@dataclass
class FileReport:
    """Validation results for a single JSONL file."""
    path: Path
    n_lines: int = 0
    n_valid: int = 0
    schemas: Counter = field(default_factory=Counter)       # "shared"/"legacy" -> count
    sources: Counter = field(default_factory=Counter)
    domains: Counter = field(default_factory=Counter)
    concepts: Counter = field(default_factory=Counter)
    errors: List[str] = field(default_factory=list)         # hard problems
    warnings: List[str] = field(default_factory=list)       # soft problems
    n_unique_pos: int = 0                                    # distinct positive *responses*
    n_unique_neg: int = 0                                    # distinct negative *responses*


# Below this many valid pairs, response-diversity ratios are too noisy to judge.
MIN_PAIRS_FOR_DIVERSITY = 10
# Warn when one side's distinct-text ratio falls below this. Low diversity means
# the diff-of-means direction can encode the surface form of a handful of
# templated responses rather than the concept itself.
LOW_DIVERSITY_RATIO = 0.10


def _detect_schema(pair: dict) -> Optional[str]:
    """Return 'shared', 'legacy', or None for an individual pair dict."""
    if all(k in pair for k in _SHARED_PROMPT_KEYS):
        return "shared"
    if all(k in pair for k in _LEGACY_KEYS):
        return "legacy"
    return None


def _response_texts(pair: dict, schema: str) -> Tuple[str, str]:
    """Return the (positive, negative) *response* components for diversity checks.

    Unlike ``pair_to_contrastive_texts`` (which prepends the shared prompt), this
    isolates the response — the part that actually differs between the two sides.
    Under the shared-prompt schema the joined text is always unique because the
    prompt is unique, so diversity must be measured on the response alone to
    detect templated/canned responses.
    """
    if schema == "shared":
        return pair["positive_response"].strip(), pair["negative_response"].strip()
    return pair["positive"].strip(), pair["negative"].strip()


def validate_file(path: Path) -> FileReport:
    """Validate a single prompt-pair JSONL file."""
    report = FileReport(path=path)
    seen: Dict[Tuple[str, str], int] = {}
    pos_texts: set = set()
    neg_texts: set = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            report.n_lines += 1

            try:
                pair = json.loads(line)
            except json.JSONDecodeError as e:
                report.errors.append(f"line {line_num}: invalid JSON ({e})")
                continue

            schema = _detect_schema(pair)
            if schema is None:
                report.errors.append(
                    f"line {line_num}: matches neither schema "
                    f"({_LEGACY_KEYS} or {_SHARED_PROMPT_KEYS})"
                )
                continue

            # Resolve to contrastive texts and check they're non-empty.
            try:
                pos, neg = pair_to_contrastive_texts(pair)
            except KeyError as e:
                report.errors.append(f"line {line_num}: {e}")
                continue
            if not pos.strip() or not neg.strip():
                report.errors.append(
                    f"line {line_num}: empty positive or negative text"
                )
                continue
            if pos.strip() == neg.strip():
                report.warnings.append(
                    f"line {line_num}: positive and negative texts are identical"
                )

            # Track response-component diversity (isolated from the shared
            # prompt) — see _response_texts for why the joined text won't do.
            pos_resp, neg_resp = _response_texts(pair, schema)
            pos_texts.add(pos_resp)
            neg_texts.add(neg_resp)

            # Duplicate detection within the file (on the full contrastive text).
            key = (pos.strip(), neg.strip())
            if key in seen:
                report.warnings.append(
                    f"line {line_num}: duplicate of line {seen[key]}"
                )
            else:
                seen[key] = line_num

            # Domain / concept consistency against config.
            concept = pair.get("concept")
            domain = pair.get("domain")
            if concept is None:
                report.warnings.append(f"line {line_num}: missing 'concept' field")
            elif concept not in CONCEPTS:
                report.warnings.append(
                    f"line {line_num}: unknown concept '{concept}'"
                )
            elif domain is not None and domain not in CONCEPTS[concept].domains:
                report.warnings.append(
                    f"line {line_num}: domain '{domain}' not configured for "
                    f"concept '{concept}'"
                )
            if domain is None:
                report.warnings.append(f"line {line_num}: missing 'domain' field")

            # Tally.
            report.n_valid += 1
            report.schemas[schema] += 1
            report.sources[pair.get("source", "unknown")] += 1
            if domain is not None:
                report.domains[domain] += 1
            if concept is not None:
                report.concepts[concept] += 1

    # Response-diversity check: too few distinct positive/negative texts means
    # the diff-of-means direction risks encoding the surface form of a handful
    # of templated responses rather than the concept.
    report.n_unique_pos = len(pos_texts)
    report.n_unique_neg = len(neg_texts)
    if report.n_valid >= MIN_PAIRS_FOR_DIVERSITY:
        for side, n_unique in (("positive", report.n_unique_pos),
                               ("negative", report.n_unique_neg)):
            ratio = n_unique / report.n_valid
            if ratio < LOW_DIVERSITY_RATIO:
                report.warnings.append(
                    f"low {side} diversity: only {n_unique} distinct texts "
                    f"across {report.n_valid} pairs ({ratio:.1%}) — direction may "
                    f"encode response templates rather than the concept"
                )

    return report


# ============================================================
# Dataset-level aggregation
# ============================================================

def validate_dataset(data_dir: Path) -> List[FileReport]:
    """Validate every ``*.jsonl`` under ``data_dir`` (recursively)."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob("*.jsonl"))
    if not files:
        logger.warning("No .jsonl files found under %s", data_dir)
    return [validate_file(p) for p in files]


def print_report(reports: List[FileReport], data_dir: Path) -> Tuple[int, int]:
    """Print a human-readable summary. Returns (total_errors, total_warnings)."""
    total_errors = sum(len(r.errors) for r in reports)
    total_warnings = sum(len(r.warnings) for r in reports)
    total_valid = sum(r.n_valid for r in reports)

    print(f"\nPrompt-pair validation — {data_dir}")
    print("=" * 72)

    # Per-file table.
    print(f"{'file':<42}{'valid':>6}{'uniq+':>7}{'uniq-':>7}{'schema':>9}{'err':>5}{'warn':>5}")
    print("-" * 81)
    for r in reports:
        rel = r.path.relative_to(data_dir)
        if not r.schemas:
            schema = "-"
        elif len(r.schemas) == 1:
            schema = next(iter(r.schemas))
        else:
            schema = "mixed"
        print(
            f"{str(rel):<42}{r.n_valid:>6}{r.n_unique_pos:>7}{r.n_unique_neg:>7}"
            f"{schema:>9}{len(r.errors):>5}{len(r.warnings):>5}"
        )

    # Aggregate by (concept, domain).
    by_cd: Dict[str, Counter] = defaultdict(Counter)
    sources: Counter = Counter()
    for r in reports:
        sources.update(r.sources)
        # Reconstruct concept->domain counts from this file's tallies.
        # A file is single-concept in practice, but aggregate defensively.
        for concept in r.concepts:
            for domain, count in r.domains.items():
                by_cd[concept][domain] += count if len(r.concepts) == 1 else 0

    print("\nPairs per concept / domain (target in parentheses):")
    print("-" * 72)
    for concept, cfg in CONCEPTS.items():
        if not cfg.domains:
            continue
        target = cfg.target_pairs_per_domain
        print(f"  {concept}:")
        for domain in cfg.domains:
            have = by_cd.get(concept, Counter()).get(domain, 0)
            flag = "" if have >= target else "  <-- below target"
            print(f"    {domain:<22}{have:>5} / {target}{flag}")

    print("\nSource breakdown:")
    print("-" * 72)
    for source, count in sorted(sources.items(), key=lambda kv: -kv[1]):
        print(f"  {source:<22}{count:>6}")

    # Surface the first few issues so they're actionable.
    if total_errors or total_warnings:
        print("\nIssues:")
        print("-" * 72)
        for r in reports:
            rel = r.path.relative_to(data_dir)
            for msg in r.errors:
                print(f"  ERROR  {rel}: {msg}")
            for msg in r.warnings[:10]:
                print(f"  WARN   {rel}: {msg}")
            if len(r.warnings) > 10:
                print(f"  WARN   {rel}: ... +{len(r.warnings) - 10} more warnings")

    print("\n" + "=" * 72)
    print(
        f"Total: {total_valid} valid pairs across {len(reports)} files | "
        f"{total_errors} errors | {total_warnings} warnings"
    )
    return total_errors, total_warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate contrastive prompt-pair JSONL files."
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(PATHS.prompt_pairs),
        help="Directory to scan recursively for .jsonl files.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as failures (non-zero exit).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    data_dir = Path(args.data_dir)
    reports = validate_dataset(data_dir)
    total_errors, total_warnings = print_report(reports, data_dir)

    if total_errors or (args.strict and total_warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
