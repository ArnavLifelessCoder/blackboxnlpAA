"""
heatmaps.py — Angular Dispersion Heatmap Figures
=================================================
Generates heatmap figures showing cosine similarity between
per-domain concept directions and the global direction across layers.

This is one of the key figures in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

from src.visualization.plot_utils import (
    setup_plotting, save_figure, format_domain_name, DOUBLE_COLUMN
)


def plot_angular_dispersion_heatmap(
    domain_vs_global: Dict[int, Dict[str, float]],
    concept: str,
    model_name: str,
    output_dir: Optional[Path] = None,
    figsize: tuple = DOUBLE_COLUMN,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "RdYlBu_r",
    layer_step: int = 1,
) -> plt.Figure:
    """Plot a heatmap of domain-vs-global cosine similarity across layers.

    X-axis: layers. Y-axis: domains.
    Color: cosine similarity (darker red = lower similarity = more fragmentation).

    Args:
        domain_vs_global: Dict from angular_dispersion.compute_angular_dispersion().
            {layer: {domain: cosine_sim}}
        concept: Concept name for the title.
        model_name: Model name for the title.
        output_dir: Where to save the figure.
        figsize: Figure dimensions.
        vmin, vmax: Color scale range.
        cmap: Colormap name.
        layer_step: Only show every Nth layer (for readability).

    Returns:
        The matplotlib Figure object.
    """
    setup_plotting()

    # Build the data matrix
    layers = sorted(domain_vs_global.keys())
    if layer_step > 1:
        layers = layers[::layer_step]

    # Get all domains from the first layer
    domains = sorted(next(iter(domain_vs_global.values())).keys())

    matrix = np.zeros((len(domains), len(layers)))
    for j, layer in enumerate(layers):
        for i, domain in enumerate(domains):
            matrix[i, j] = domain_vs_global.get(layer, {}).get(domain, np.nan)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[str(l) for l in layers],
        yticklabels=[format_domain_name(d) for d in domains],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=True if len(layers) <= 20 else False,
        fmt=".2f" if len(layers) <= 20 else "",
        cbar_kws={"label": "Cosine Similarity to Global Direction"},
        linewidths=0.5,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Domain")
    ax.set_title(
        f"Angular Dispersion: {concept.title()} — {model_name}",
        fontsize=10,
        pad=10,
    )

    fig.tight_layout()

    # Save
    if output_dir:
        save_figure(fig, f"heatmap_{concept}_{model_name}", output_dir, close=False)

    return fig


def plot_dispersion_profile(
    dispersion: Dict[int, Dict[str, float]],
    concept: str,
    model_name: str,
    metric: str = "mean_angle_deg",
    output_dir: Optional[Path] = None,
    figsize: tuple = (6.75, 3.0),
) -> plt.Figure:
    """Plot the layer-wise angular dispersion profile.

    Shows how angular dispersion (std of angles) changes across layers.
    Useful for identifying which layers show the most fragmentation.

    Args:
        dispersion: Dict from compute_angular_dispersion()["dispersion"].
            {layer: {"mean_cos": float, "std_cos": float, "mean_angle_deg": float, ...}}
        concept: Concept name.
        model_name: Model name.
        metric: Which metric to plot ("mean_angle_deg", "std_cos", etc.).
        output_dir: Where to save.
        figsize: Figure size.

    Returns:
        The matplotlib Figure.
    """
    setup_plotting()

    layers = sorted(dispersion.keys())
    values = [dispersion[l][metric] for l in layers]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(layers, values, color="#0072B2", linewidth=1.5, marker="o", markersize=3)
    ax.fill_between(layers, 0, values, alpha=0.15, color="#0072B2")

    ax.set_xlabel("Layer")
    ylabel_map = {
        "mean_angle_deg": "Mean Angle to Global (°)",
        "std_cos": "Std of Cosine Similarities",
        "mean_cos": "Mean Cosine Similarity",
        "max_angle_deg": "Max Angle to Global (°)",
    }
    ax.set_ylabel(ylabel_map.get(metric, metric))
    ax.set_title(
        f"Dispersion Profile: {concept.title()} — {model_name}",
        fontsize=10,
    )

    fig.tight_layout()

    if output_dir:
        save_figure(fig, f"dispersion_profile_{concept}_{model_name}", output_dir, close=False)

    return fig
