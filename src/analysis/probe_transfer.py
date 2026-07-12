"""
probe_transfer.py — Cross-Domain Probing Transfer (honesty's second pillar)
============================================================================
Cosine dispersion alone shows that per-domain difference-of-means directions
diverge; it cannot say whether the *information* the domains use is shared.
This module tests that functionally: train a linear probe (logistic
regression on the residual stream) to separate positive from negative
activations within one domain, then evaluate it on every other domain's
held-out split.

Interpretation:
  * High off-diagonal accuracy despite low direction cosines -> the domains
    share a decodable concept subspace and the cosine fragmentation is
    largely superficial (e.g., lexical).
  * Off-diagonal accuracy near chance while within-domain accuracy is high
    -> fragmentation is functional, not just geometric.

Either outcome is informative; the paper text must follow the result.

CPU-only; needs the cached activations produced by extract_activations.

Usage:
    python -m src.analysis.probe_transfer \
        --model qwen-2.5-3b-instruct --concept honesty --layers 5 18 30
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import CONCEPTS, MODELS, PATHS
from src.extraction.cache_utils import load_activations

logger = logging.getLogger(__name__)


def _domain_splits(
    activations_dir: Path,
    model_name: str,
    concept: str,
    domains: List[str],
    layer: int,
    test_frac: float,
    seed: int,
    balance_to: Optional[int] = None,
) -> Dict[str, dict]:
    """Load per-domain activations at one layer and make seeded train/test splits.

    Positive and negative rows are paired by index, so the same permutation is
    applied to both sides and a pair is never split across train and test.
    """
    splits = {}
    for di, domain in enumerate(sorted(domains)):
        pos = load_activations(activations_dir, model_name, concept, domain,
                               layers=[layer], prefix="pos_")
        neg = load_activations(activations_dir, model_name, concept, domain,
                               layers=[layer], prefix="neg_")
        if layer not in pos or layer not in neg:
            logger.warning("No activations for %s/%s layer %d - skipping domain",
                           concept, domain, layer)
            continue
        p, n = pos[layer].numpy(), neg[layer].numpy()
        n_pairs = min(len(p), len(n))
        rng = np.random.default_rng([seed, di])
        order = rng.permutation(n_pairs)
        if balance_to is not None:
            order = order[:balance_to]
        n_test = max(1, int(round(len(order) * test_frac)))
        test_idx, train_idx = order[:n_test], order[n_test:]

        def xy(idx):
            X = np.concatenate([p[idx], n[idx]])
            y = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
            return X, y

        splits[domain] = {"train": xy(train_idx), "test": xy(test_idx),
                          "n_pairs": len(order)}
    return splits


def probe_transfer_matrix(
    activations_dir: Path,
    model_name: str,
    concept: str,
    domains: List[str],
    layer: int,
    test_frac: float = 0.25,
    seed: int = 42,
    balance_to: Optional[int] = None,
    C: float = 1.0,
) -> Dict[str, Dict[str, dict]]:
    """Train a probe per domain; evaluate on every domain's held-out split.

    Returns {train_domain: {test_domain: {"acc": float, "auc": float}}}.
    The diagonal is within-domain generalization (same-domain held-out pairs).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    splits = _domain_splits(activations_dir, model_name, concept, domains,
                            layer, test_frac, seed, balance_to)
    matrix: Dict[str, Dict[str, dict]] = {}
    for train_d, s in splits.items():
        X_tr, y_tr = s["train"]
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(scaler.transform(X_tr), y_tr)
        matrix[train_d] = {}
        for test_d, t in splits.items():
            X_te, y_te = t["test"]
            X_te = scaler.transform(X_te)
            acc = float(clf.score(X_te, y_te))
            auc = float(roc_auc_score(y_te, clf.decision_function(X_te)))
            matrix[train_d][test_d] = {"acc": acc, "auc": auc}
            logger.info("layer %d | train %s -> test %s: acc=%.3f auc=%.3f",
                        layer, train_d, test_d, acc, auc)
    return matrix


def summarize(matrix: Dict[str, Dict[str, dict]]) -> dict:
    """Mean within-domain (diagonal) vs cross-domain (off-diagonal) accuracy."""
    diag, off = [], []
    for a, row in matrix.items():
        for b, m in row.items():
            (diag if a == b else off).append(m["acc"])
    return {
        "mean_within_acc": float(np.mean(diag)) if diag else None,
        "mean_cross_acc": float(np.mean(off)) if off else None,
        "min_cross_acc": float(np.min(off)) if off else None,
        "transfer_gap": (float(np.mean(diag) - np.mean(off))
                         if diag and off else None),
    }


def run(
    model_name: str,
    concept: str,
    domains: List[str],
    layers: List[int],
    activations_dir: Path,
    report_dir: Path,
    test_frac: float = 0.25,
    seed: int = 42,
    balance_to: Optional[int] = None,
) -> dict:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name, "concept": concept, "domains": sorted(domains),
        "layers": layers, "test_frac": test_frac, "seed": seed,
        "balance_to": balance_to, "matrix": {}, "summary": {},
    }
    for layer in layers:
        m = probe_transfer_matrix(activations_dir, model_name, concept,
                                  domains, layer, test_frac, seed, balance_to)
        report["matrix"][layer] = m
        report["summary"][layer] = summarize(m)
        logger.info("layer %d summary: %s", layer, report["summary"][layer])

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"probe_transfer_{concept}_{model_name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", out)

    md = [f"# Probe transfer — {concept} ({model_name})", ""]
    for layer in layers:
        md.append(f"## Layer {layer}")
        doms = sorted(report["matrix"][layer].keys())
        md.append("| train \\ test | " + " | ".join(doms) + " |")
        md.append("|---|" + "---|" * len(doms))
        for a in doms:
            row = [f"{report['matrix'][layer][a][b]['acc']:.3f}" for b in doms]
            md.append(f"| {a} | " + " | ".join(row) + " |")
        md.append("")
        md.append(f"summary: {report['summary'][layer]}")
        md.append("")
    with open(report_dir / f"probe_transfer_{concept}_{model_name}.md",
              "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Cross-domain linear-probe transfer from cached activations."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--concept", type=str, required=True,
                        choices=list(CONCEPTS.keys()))
    parser.add_argument("--domains", type=str, nargs="*", default=None)
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="e.g. 5 18 30 (fragmented, steering, late layer)")
    parser.add_argument("--activations", type=str, default=str(PATHS.activations))
    parser.add_argument("--report-dir", type=str, default=str(PATHS.results))
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--balance-to", type=int, default=None,
                        help="Subsample every domain to this many pairs "
                             "(e.g. 59 to match the smallest honesty domain).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    if not args.verbose:
        logging.getLogger("src.extraction.cache_utils").setLevel(logging.WARNING)

    domains = args.domains or CONCEPTS[args.concept].domains
    run(args.model, args.concept, domains, args.layers,
        Path(args.activations), Path(args.report_dir),
        test_frac=args.test_frac, seed=args.seed, balance_to=args.balance_to)


if __name__ == "__main__":
    main()
