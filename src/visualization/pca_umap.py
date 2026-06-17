"""
pca_umap.py — PCA/UMAP Geometry Visualization
===============================================
Generates the "key geometric figure" for the paper (per the DuoPlan):
PCA/UMAP visualization of per-domain direction clusters.

This figure shows whether concept directions from different domains
cluster together (universal) or separate (fragmented) in the space
of possible directions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from src.visualization.plot_utils import (
    setup_plotting, save_figure, format_domain_name, get_color,
    DOUBLE_COLUMN, SQUARE, COLORS,
)

logger = logging.getLogger(__name__)


def plot_direction_geometry(
    directions: Dict[str, Dict[int, np.ndarray]],
    concept: str,
    model_name: str,
    method: str = "pca",
    n_pca_components: int = 50,
    umap_kwargs: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    figsize: tuple = DOUBLE_COLUMN,
) -> plt.Figure:
    """Plot 2D projection of direction vectors, colored by domain.

    Each point is a direction vector from a specific (domain, layer) pair.
    The visualization shows whether directions from different domains
    cluster together or separate.

    Args:
        directions: Dict mapping domain_name -> {layer: direction_vector (numpy)}.
            Include "global" as a special domain.
        concept: Concept name.
        model_name: Model name.
        method: "pca" for PCA, "umap" for UMAP, "pca+umap" for PCA then UMAP.
        n_pca_components: Intermediate PCA dimensions before UMAP.
        umap_kwargs: Override UMAP parameters.
        output_dir: Save directory.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    setup_plotting()

    # Collect all direction vectors with metadata
    all_vectors = []
    all_labels = []
    all_layers = []

    for domain, layer_dirs in directions.items():
        for layer, vec in layer_dirs.items():
            if isinstance(vec, np.ndarray):
                all_vectors.append(vec)
            else:
                all_vectors.append(vec.numpy())
            all_labels.append(domain)
            all_layers.append(layer)

    if len(all_vectors) < 3:
        logger.warning("Too few direction vectors for dimensionality reduction.")
        return plt.figure()

    X = np.stack(all_vectors)  # (n_directions, d_model)
    labels = np.array(all_labels)
    layers = np.array(all_layers)

    # Dimensionality reduction
    if method == "pca" or method == "pca+umap":
        from sklearn.decomposition import PCA
        n_components_pca = min(n_pca_components, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_components_pca)
        X_reduced = pca.fit_transform(X)

        if method == "pca":
            X_2d = X_reduced[:, :2]
            explained = pca.explained_variance_ratio_[:2]
            xlabel = f"PC1 ({explained[0]:.1%} var.)"
            ylabel = f"PC2 ({explained[1]:.1%} var.)"

    if method == "umap" or method == "pca+umap":
        try:
            import umap
        except ImportError:
            logger.error("umap-learn not installed — falling back to PCA.")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            xlabel, ylabel = "PC1", "PC2"
        else:
            default_umap = {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 2,
                "random_state": 42,
                "metric": "cosine",
            }
            if umap_kwargs:
                default_umap.update(umap_kwargs)

            input_data = X_reduced if method == "pca+umap" else X
            reducer = umap.UMAP(**default_umap)
            X_2d = reducer.fit_transform(input_data)
            xlabel = "UMAP-1"
            ylabel = "UMAP-2"

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    unique_domains = sorted(set(all_labels))

    for domain in unique_domains:
        mask = labels == domain
        color = get_color(domain)
        marker = "D" if domain == "global" else "o"
        size = 40 if domain == "global" else 20
        alpha = 1.0 if domain == "global" else 0.7
        zorder = 10 if domain == "global" else 5

        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            label=format_domain_name(domain),
            edgecolors="white" if domain != "global" else "black",
            linewidths=0.5,
            zorder=zorder,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Direction Geometry: {concept.title()} — {model_name}",
        fontsize=10,
    )
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left",
        framealpha=0.9, fontsize=7,
    )

    fig.tight_layout()

    if output_dir:
        method_label = method.replace("+", "_")
        save_figure(
            fig,
            f"geometry_{method_label}_{concept}_{model_name}",
            output_dir,
            close=False,
        )

    return fig


def plot_direction_geometry_by_layer_range(
    directions: Dict[str, Dict[int, np.ndarray]],
    concept: str,
    model_name: str,
    layer_ranges: List[Tuple[int, int]],
    method: str = "pca",
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """Plot direction geometry for different layer ranges side by side.

    Useful for showing that fragmentation varies by layer depth
    (e.g., early layers vs. middle vs. late).

    Args:
        directions: Full direction dict.
        concept: Concept name.
        model_name: Model name.
        layer_ranges: List of (start, end) layer ranges.
        method: Dimensionality reduction method.
        output_dir: Save directory.

    Returns:
        Matplotlib Figure.
    """
    setup_plotting()

    ncols = len(layer_ranges)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.0, 3.0))
    if ncols == 1:
        axes = [axes]

    for ax, (layer_start, layer_end) in zip(axes, layer_ranges):
        # Filter directions to this layer range
        filtered = {}
        for domain, layer_dirs in directions.items():
            filtered_dirs = {
                l: v for l, v in layer_dirs.items()
                if layer_start <= l <= layer_end
            }
            if filtered_dirs:
                filtered[domain] = filtered_dirs

        # Collect vectors
        all_vectors = []
        all_labels = []
        for domain, layer_dirs in filtered.items():
            for layer, vec in layer_dirs.items():
                v = vec if isinstance(vec, np.ndarray) else vec.numpy()
                all_vectors.append(v)
                all_labels.append(domain)

        if len(all_vectors) < 3:
            ax.text(0.5, 0.5, "Too few vectors", ha="center", transform=ax.transAxes)
            continue

        X = np.stack(all_vectors)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(2, X.shape[0] - 1))
        X_2d = pca.fit_transform(X)

        labels = np.array(all_labels)
        for domain in sorted(set(all_labels)):
            mask = labels == domain
            color = get_color(domain)
            marker = "D" if domain == "global" else "o"
            size = 30 if domain == "global" else 15
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=color, marker=marker, s=size,
                alpha=0.7, edgecolors="white", linewidths=0.3,
            )

        ax.set_title(f"Layers {layer_start}–{layer_end}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Direction Geometry by Depth: {concept.title()} — {model_name}",
        fontsize=10,
    )
    fig.tight_layout()

    if output_dir:
        save_figure(
            fig,
            f"geometry_by_depth_{concept}_{model_name}",
            output_dir,
            close=False,
        )

    return fig
