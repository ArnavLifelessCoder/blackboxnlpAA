"""
transfer_matrix.py — Cross-Domain Transfer Matrix Plots
========================================================
Generates heatmap figures showing pairwise cosine similarity between
domain-specific concept directions.

If domain A's direction has high cosine similarity with domain B's direction,
then steering with A's direction should also work for B's prompts.
Low similarity = steering doesn't transfer across domains.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.visualization.plot_utils import (
    setup_plotting, save_figure, format_domain_name, SQUARE
)


def plot_transfer_matrix(
    pairwise_sims: Dict[Tuple[str, str], float],
    domains: List[str],
    concept: str,
    model_name: str,
    layer: int,
    output_dir: Optional[Path] = None,
    figsize: tuple = SQUARE,
    cmap: str = "RdYlBu_r",
) -> plt.Figure:
    """Plot the cross-domain transfer matrix as a heatmap.

    Shows pairwise cosine similarity between all domain-specific directions
    at a given layer.

    Args:
        pairwise_sims: Dict from angular_dispersion.compute_cross_domain_transfer().
            {(domain_a, domain_b): cosine_sim}
        domains: Ordered list of domain names.
        concept: Concept name.
        model_name: Model name.
        layer: Layer index (for title).
        output_dir: Where to save the figure.
        figsize: Figure size.
        cmap: Colormap.

    Returns:
        Matplotlib Figure.
    """
    setup_plotting()

    # Build symmetric matrix
    n = len(domains)
    matrix = np.zeros((n, n))
    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            matrix[i, j] = pairwise_sims.get((d1, d2), np.nan)

    labels = [format_domain_name(d) for d in domains]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
        linewidths=1.0,
        linecolor="white",
    )

    ax.set_title(
        f"Cross-Domain Transfer: {concept.title()}\n"
        f"{model_name} — Layer {layer}",
        fontsize=10,
        pad=10,
    )

    # Rotate x labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()

    if output_dir:
        save_figure(
            fig,
            f"transfer_matrix_{concept}_{model_name}_layer{layer:03d}",
            output_dir,
            close=False,
        )

    return fig


def plot_multi_layer_transfer(
    pairwise_by_layer: Dict[int, Dict[Tuple[str, str], float]],
    domains: List[str],
    concept: str,
    model_name: str,
    layers_to_show: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    ncols: int = 3,
) -> plt.Figure:
    """Plot transfer matrices for multiple layers in a grid.

    Args:
        pairwise_by_layer: {layer: {(d1, d2): cosine_sim}}.
        domains: Domain names.
        concept: Concept name.
        model_name: Model name.
        layers_to_show: Which layers to plot. None = all.
        output_dir: Save directory.
        ncols: Columns in the grid.

    Returns:
        Matplotlib Figure.
    """
    setup_plotting()

    if layers_to_show is None:
        layers_to_show = sorted(pairwise_by_layer.keys())

    n = len(domains)
    nrows = (len(layers_to_show) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.5, nrows * 2.5),
        squeeze=False,
    )

    labels = [format_domain_name(d) for d in domains]

    for idx, layer in enumerate(layers_to_show):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        matrix = np.zeros((n, n))
        sims = pairwise_by_layer.get(layer, {})
        for i, d1 in enumerate(domains):
            for j, d2 in enumerate(domains):
                matrix[i, j] = sims.get((d1, d2), np.nan)

        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=labels if row == nrows - 1 else [],
            yticklabels=labels if col == 0 else [],
            vmin=0.0, vmax=1.0,
            cmap="RdYlBu_r",
            annot=False,
            square=True,
            cbar=False,
            linewidths=0.5,
        )
        ax.set_title(f"Layer {layer}", fontsize=8)

    # Remove unused subplots
    for idx in range(len(layers_to_show), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Cross-Domain Transfer: {concept.title()} — {model_name}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    if output_dir:
        save_figure(
            fig,
            f"transfer_matrix_grid_{concept}_{model_name}",
            output_dir,
            close=False,
        )

    return fig
