"""
run_pipeline.py — End-to-End Analysis Pipeline Driver
======================================================
Ties the whole Phase 3 analysis chain together into a single command:

    directions  ->  angular dispersion  ->  cross-domain transfer
                ->  bootstrap CIs        ->  publication figures
                ->  summary report (JSON + Markdown)

It runs from cached activations produced by `extract_activations.py`. It also
ships a **mock mode** (`--mock`) that synthesises activations with a
controllable, layer-dependent amount of domain fragmentation. Mock mode lets
the team validate the entire downstream pipeline on CPU — verifying that the
analysis and figure code actually produces sensible numbers and plots — BEFORE
spending the limited GPU budget on real extraction.

Usage
-----
# Dry-run the whole pipeline end-to-end on synthetic data (no GPU, no model):
    python -m src.analysis.run_pipeline --mock --concept refusal

# Run on real cached activations:
    python -m src.analysis.run_pipeline \
        --model gemma-2-2b-it --concept refusal \
        --activations results/activations/ \
        --directions results/directions/ \
        --figures results/figures/

The mock mode is deterministic given --seed, so the produced report is
reproducible across machines.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import CONCEPTS, MODELS, PATHS
from src.extraction.cache_utils import save_activations, load_activations, load_direction
from src.analysis.directions import compute_all_directions, balanced_subsample_indices
from src.analysis.angular_dispersion import (
    compute_angular_dispersion,
    compute_cross_domain_transfer,
)
from src.analysis.bootstrap import (
    bootstrap_cosine_similarity_ci,
    bootstrap_angular_dispersion_ci,
    bootstrap_pairlevel_dispersion_ci,
)

logger = logging.getLogger(__name__)


# ============================================================
# Mock Activation Synthesis
# ============================================================

def synthesize_mock_activations(
    activations_dir: Path,
    model_name: str,
    concept: str,
    domains: List[str],
    layers: List[int],
    n_pairs: int = 40,
    d_model: int = 256,
    base_fragmentation: float = 0.15,
    fragmentation_growth: float = 0.05,
    signal_strength: float = 4.0,
    noise_std: float = 1.0,
    seed: int = 42,
) -> None:
    """Generate synthetic pos/neg activations with known fragmentation structure.

    The generative model is intentionally simple and interpretable so that the
    downstream analysis has a *ground truth* to recover:

      * A single global concept axis ``g`` is shared by all domains.
      * Each domain gets its own axis ``g_d = normalise(g + frac * u_d)`` where
        ``u_d`` is a random unit vector and ``frac`` is the fragmentation level.
      * Positive activations are shifted along ``g_d``; negatives are not.

    Difference-of-means therefore recovers ``~g_d`` per domain and ``~g`` for the
    global aggregate, so the measured cosine(domain, global) should decrease as
    ``frac`` grows. ``frac`` increases with layer depth to mimic the common
    finding that later layers fragment more.

    Args:
        activations_dir: Where to write the cached ``.pt`` files.
        model_name: Model identifier used in filenames.
        concept: Concept name.
        domains: Domain names to synthesise.
        layers: Layer indices to synthesise.
        n_pairs: Number of contrastive pairs per domain.
        d_model: Hidden dimension of the synthetic activations.
        base_fragmentation: Fragmentation level at the earliest layer.
        fragmentation_growth: Added fragmentation per layer index step.
        signal_strength: Magnitude of the positive-vs-negative mean shift.
        noise_std: Per-coordinate Gaussian noise std.
        seed: RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Shared global concept axis.
    g = rng.standard_normal(d_model)
    g /= np.linalg.norm(g)

    # A fixed per-domain perturbation direction (stable across layers so the
    # fragmentation grows smoothly rather than jumping around randomly).
    domain_perturbation = {}
    for domain in domains:
        u = rng.standard_normal(d_model)
        u -= u.dot(g) * g  # orthogonalise against g for a clean fragmentation knob
        u /= np.linalg.norm(u)
        domain_perturbation[domain] = u

    layer_min = min(layers)

    for domain in domains:
        u = domain_perturbation[domain]
        pos_by_layer: Dict[int, torch.Tensor] = {}
        neg_by_layer: Dict[int, torch.Tensor] = {}

        for layer in layers:
            frac = base_fragmentation + fragmentation_growth * (layer - layer_min)
            axis = g + frac * u
            axis /= np.linalg.norm(axis)

            base = rng.standard_normal((n_pairs, d_model)) * noise_std
            pos = base + signal_strength * axis[None, :]
            neg = (
                rng.standard_normal((n_pairs, d_model)) * noise_std
            )  # negatives: no concept shift

            pos_by_layer[layer] = torch.from_numpy(pos).float()
            neg_by_layer[layer] = torch.from_numpy(neg).float()

        save_activations(
            pos_by_layer, activations_dir, model_name, concept, domain, prefix="pos_"
        )
        save_activations(
            neg_by_layer, activations_dir, model_name, concept, domain, prefix="neg_"
        )

    logger.info(
        "Synthesised mock activations: model=%s concept=%s domains=%s layers=%s "
        "(n_pairs=%d, d_model=%d, seed=%d)",
        model_name, concept, domains, layers, n_pairs, d_model, seed,
    )


# ============================================================
# Figure Generation
# ============================================================

def generate_figures(
    dispersion_result: dict,
    directions: Dict[str, Dict[int, torch.Tensor]],
    transfer_layer: int,
    transfer_sims: dict,
    concept: str,
    model_name: str,
    domains: List[str],
    figures_dir: Path,
) -> List[str]:
    """Render all publication figures. Returns the list of figure base names.

    Imported lazily so the numerical pipeline can run in environments without a
    working matplotlib/seaborn stack.
    """
    from src.visualization.heatmaps import (
        plot_angular_dispersion_heatmap,
        plot_dispersion_profile,
    )
    from src.visualization.transfer_matrix import plot_transfer_matrix
    from src.visualization.pca_umap import plot_direction_geometry

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    produced: List[str] = []

    plot_angular_dispersion_heatmap(
        dispersion_result["domain_vs_global"],
        concept=concept, model_name=model_name, output_dir=figures_dir,
    )
    produced.append(f"heatmap_{concept}_{model_name}")

    plot_dispersion_profile(
        dispersion_result["dispersion"],
        concept=concept, model_name=model_name,
        metric="mean_cos", output_dir=figures_dir,
    )
    produced.append(f"dispersion_profile_{concept}_{model_name}")

    plot_transfer_matrix(
        transfer_sims, domains=domains, concept=concept,
        model_name=model_name, layer=transfer_layer, output_dir=figures_dir,
    )
    produced.append(f"transfer_matrix_{concept}_{model_name}_layer{transfer_layer:03d}")

    plot_direction_geometry(
        directions, concept=concept, model_name=model_name,
        method="pca", output_dir=figures_dir,
    )
    produced.append(f"geometry_pca_{concept}_{model_name}")

    import matplotlib.pyplot as plt
    plt.close("all")

    logger.info("Generated %d figures in %s", len(produced), figures_dir)
    return produced


# ============================================================
# Report Writing
# ============================================================

def write_report(
    report: dict,
    report_dir: Path,
    concept: str,
    model_name: str,
) -> Path:
    """Write the analysis summary as both JSON and a human-readable Markdown file."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / f"report_{concept}_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = report_dir / f"report_{concept}_{model_name}.md"
    lines = [
        f"# Analysis Report — {concept.title()} ({model_name})",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Mode: {'MOCK (synthetic activations)' if report['mock'] else 'real activations'}",
        f"- Domains: {', '.join(report['domains'])}",
        f"- Layers analysed: {report['layers']}",
        "",
        "## Angular Dispersion (domain direction vs. global direction)",
        "",
        "| Layer | Mean cosine | 95% CI | Mean angle (deg) |",
        "|------:|------------:|:-------|-----------------:|",
    ]
    for layer in report["layers"]:
        d = report["dispersion"].get(str(layer)) or report["dispersion"].get(layer)
        if d is None:
            continue
        ci = report["bootstrap"].get(str(layer)) or report["bootstrap"].get(layer)
        ci_txt = f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]" if ci else "—"
        lines.append(
            f"| {layer} | {d['mean_cos']:.3f} | {ci_txt} | {d['mean_angle_deg']:.1f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        f"- Lowest mean cosine-to-global: **{report['summary']['min_mean_cos']:.3f}** "
        f"at layer {report['summary']['min_mean_cos_layer']}.",
        f"- Highest mean angle-to-global: **{report['summary']['max_mean_angle_deg']:.1f}°** "
        f"at layer {report['summary']['max_mean_angle_layer']}.",
        f"- Late-layer (final third) mean cosine: **{report['summary']['late_layer_mean_cos']:.3f}** "
        f"(layers {report['summary']['late_layers'][0]}–{report['summary']['late_layers'][-1]}).",
        "",
        "Lower cosine / higher angle indicates stronger domain fragmentation "
        "(per-domain directions diverging from the global direction). Note: "
        "minima in early layers may reflect token/surface statistics rather "
        "than concept representations — contrast with the late-layer aggregate, "
        "where behavior is read out.",
        "",
        "## Figures",
        "",
    ]
    for name in report["figures"]:
        lines.append(f"- `{name}.pdf` / `{name}.png`")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Wrote report: %s and %s", json_path, md_path)
    return md_path


# ============================================================
# Pipeline Orchestration
# ============================================================

def run_pipeline(
    model_name: str,
    concept: str,
    domains: List[str],
    layers: List[int],
    activations_dir: Path,
    directions_dir: Path,
    figures_dir: Path,
    report_dir: Path,
    mock: bool = False,
    make_figures: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
    mock_kwargs: Optional[dict] = None,
    transfer_layer: Optional[int] = None,
    balance_domains: bool = False,
) -> dict:
    """Run the full analysis pipeline and return the report dict."""
    activations_dir = Path(activations_dir)
    directions_dir = Path(directions_dir)

    # 1. (Optional) synthesise activations.
    if mock:
        synthesize_mock_activations(
            activations_dir, model_name, concept, domains, layers,
            seed=seed, **(mock_kwargs or {}),
        )

    # 2. Compute per-domain + global directions.
    logger.info("Computing directions...")
    directions = compute_all_directions(
        activations_dir, directions_dir, model_name, concept, domains, layers,
        balance_domains=balance_domains, seed=seed,
    )

    # Drop any layers that failed to produce a global direction.
    usable_layers = [l for l in layers if l in directions.get("global", {})]
    if not usable_layers:
        raise RuntimeError(
            "No usable directions were computed. Check that activations exist "
            f"for {model_name}/{concept} in {activations_dir}."
        )

    # 3. Angular dispersion.
    logger.info("Computing angular dispersion...")
    disp = compute_angular_dispersion(
        directions_dir, model_name, concept, domains, usable_layers,
    )

    # 4. Bootstrap CIs on the mean cosine-to-global per layer. The pair-level
    # bootstrap resamples prompt pairs and recomputes directions each resample;
    # the domain-level fallback (bootstrapping the handful of per-domain
    # cosines) is statistically much weaker and is only used when the per-pair
    # activations cannot be loaded.
    logger.info("Bootstrapping confidence intervals (%d resamples, pair-level)...", n_bootstrap)
    # Same seeded subsample as compute_all_directions, so the bootstrap sees
    # exactly the pairs the directions were computed from.
    balance_idx = (
        balanced_subsample_indices(
            activations_dir, model_name, concept, domains, usable_layers, seed=seed,
        )
        if balance_domains else {}
    )
    bootstrap: Dict[int, dict] = {}
    for layer in usable_layers:
        pos_by_domain, neg_by_domain = {}, {}
        for domain in domains:
            pos = load_activations(
                activations_dir, model_name, concept, domain,
                layers=[layer], prefix="pos_",
            )
            neg = load_activations(
                activations_dir, model_name, concept, domain,
                layers=[layer], prefix="neg_",
            )
            if layer in pos and layer in neg:
                idx = balance_idx.get(domain)
                pos_by_domain[domain] = (
                    pos[layer][idx] if idx is not None else pos[layer]
                ).numpy()
                neg_by_domain[domain] = (
                    neg[layer][idx] if idx is not None else neg[layer]
                ).numpy()

        if pos_by_domain:
            bootstrap[layer] = bootstrap_pairlevel_dispersion_ci(
                pos_by_domain, neg_by_domain,
                n_bootstrap=n_bootstrap, seed=seed,
            )
        else:
            cos_vals = list(disp["domain_vs_global"][layer].values())
            ci = bootstrap_cosine_similarity_ci(cos_vals, n_bootstrap=n_bootstrap, seed=seed)
            ci_angle = bootstrap_angular_dispersion_ci(
                cos_vals, n_bootstrap=n_bootstrap, seed=seed
            )
            ci["angle_std_ci_lower"] = ci_angle["ci_lower"]
            ci["angle_std_ci_upper"] = ci_angle["ci_upper"]
            ci["method"] = "domain-level"
            bootstrap[layer] = ci

    # 5. Cross-domain transfer at the most-fragmented layer. Embedding-adjacent
    # layers are excluded from the argmin: their directions reflect token/surface
    # statistics rather than concept representations, so a minimum there is an
    # artifact (observed: refusal argmin landed on layer 0). An explicit
    # transfer_layer overrides the selection.
    if transfer_layer is not None:
        if transfer_layer not in usable_layers:
            raise ValueError(
                f"Requested transfer layer {transfer_layer} has no usable "
                f"directions (usable: {usable_layers})."
            )
    else:
        candidate_layers = [l for l in usable_layers if l >= 2] or usable_layers
        transfer_layer = min(
            candidate_layers, key=lambda l: disp["dispersion"][l]["mean_cos"]
        )
    logger.info("Computing cross-domain transfer at layer %d...", transfer_layer)
    transfer_sims = compute_cross_domain_transfer(
        directions_dir, model_name, concept, domains, transfer_layer,
    )

    # 6. Summary statistics.
    mean_cos_by_layer = {l: disp["dispersion"][l]["mean_cos"] for l in usable_layers}
    mean_angle_by_layer = {l: disp["dispersion"][l]["mean_angle_deg"] for l in usable_layers}
    min_cos_layer = min(mean_cos_by_layer, key=mean_cos_by_layer.get)
    max_angle_layer = max(mean_angle_by_layer, key=mean_angle_by_layer.get)

    # Early-layer minima can reflect token/surface statistics rather than
    # concept representations; the late-layer aggregate (final third of the
    # stack, where behavior is read out) is reported alongside the global
    # minimum so the two can be contrasted.
    late_layers = usable_layers[-max(1, len(usable_layers) // 3):]
    late_mean_cos = float(np.mean([mean_cos_by_layer[l] for l in late_layers]))

    # Extraction provenance sidecars (written by extract_activations), if any.
    model_short = model_name.replace("-", "_").replace(".", "")
    provenance = []
    for meta_path in sorted(Path(activations_dir).glob(f"meta_{model_short}_{concept}_*.json")):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                provenance.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read provenance sidecar %s: %s", meta_path, e)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "concept": concept,
        "domains": domains,
        "layers": usable_layers,
        "mock": mock,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "balance_domains": balance_domains,
        "provenance": provenance,
        "dispersion": {l: disp["dispersion"][l] for l in usable_layers},
        "domain_vs_global": {
            l: disp["domain_vs_global"][l] for l in usable_layers
        },
        "bootstrap": bootstrap,
        "transfer": {
            "layer": transfer_layer,
            "pairwise": {f"{a}->{b}": v for (a, b), v in transfer_sims.items()},
        },
        "summary": {
            "min_mean_cos": float(mean_cos_by_layer[min_cos_layer]),
            "min_mean_cos_layer": int(min_cos_layer),
            "max_mean_angle_deg": float(mean_angle_by_layer[max_angle_layer]),
            "max_mean_angle_layer": int(max_angle_layer),
            "late_layer_mean_cos": late_mean_cos,
            "late_layers": [int(l) for l in late_layers],
        },
        "figures": [],
    }

    # 7. Figures.
    if make_figures:
        try:
            report["figures"] = generate_figures(
                disp, directions, transfer_layer, transfer_sims,
                concept, model_name, domains, figures_dir,
            )
        except Exception as e:  # pragma: no cover - environment dependent
            logger.warning("Figure generation failed (%s). Continuing without figures.", e)

    # 8. Report.
    write_report(report, report_dir, concept, model_name)
    return report


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the end-to-end concept-direction analysis pipeline."
    )
    parser.add_argument("--model", type=str, default="gemma-2-2b-it",
                        help="Model key (used for filenames; need not exist for --mock).")
    parser.add_argument("--concept", type=str, default="refusal",
                        choices=list(CONCEPTS.keys()))
    parser.add_argument("--domains", type=str, nargs="*", default=None,
                        help="Domain names (default: concept's configured domains).")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layer indices to analyse (default: mock=[2,6,10,14,18,22], "
                             "real=all model layers).")
    parser.add_argument("--activations", type=str, default=str(PATHS.activations))
    parser.add_argument("--directions", type=str, default=str(PATHS.directions))
    parser.add_argument("--figures", type=str, default=str(PATHS.figures))
    parser.add_argument("--report-dir", type=str, default=str(PATHS.results))
    parser.add_argument("--mock", action="store_true",
                        help="Synthesise activations instead of loading cached ones.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation (numbers + report only).")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--transfer-layer", type=int, default=None,
                        help="Layer for the cross-domain transfer matrix "
                             "(default: most-fragmented layer, excluding layers 0-1).")
    parser.add_argument("--balance-domains", action="store_true",
                        help="Subsample every domain to the smallest domain's pair "
                             "count before computing directions (T4 sensitivity).")
    parser.add_argument("--n-pairs", type=int, default=40, help="(mock) pairs per domain.")
    parser.add_argument("--d-model", type=int, default=256, help="(mock) hidden size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    # Quiet very noisy third-party loggers (font subsetting, image libs).
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # The per-file cache load/save logs are useful for real runs but drown out
    # the pipeline progress; keep them only in --verbose mode.
    if not args.verbose:
        logging.getLogger("src.extraction.cache_utils").setLevel(logging.WARNING)

    domains = args.domains or CONCEPTS[args.concept].domains
    if not domains:
        parser.error(f"Concept '{args.concept}' has no configured domains; pass --domains.")

    if args.layers:
        layers = args.layers
    elif args.mock:
        layers = [2, 6, 10, 14, 18, 22]
    else:
        layers = list(range(MODELS[args.model].n_layers))

    run_pipeline(
        model_name=args.model,
        concept=args.concept,
        domains=domains,
        layers=layers,
        activations_dir=Path(args.activations),
        directions_dir=Path(args.directions),
        figures_dir=Path(args.figures),
        report_dir=Path(args.report_dir),
        mock=args.mock,
        make_figures=not args.no_figures,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        mock_kwargs={"n_pairs": args.n_pairs, "d_model": args.d_model},
        transfer_layer=args.transfer_layer,
        balance_domains=args.balance_domains,
    )


if __name__ == "__main__":
    main()
