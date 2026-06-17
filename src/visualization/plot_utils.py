"""
plot_utils.py — Shared Plotting Configuration
==============================================
Publication-quality figure settings for ACL-formatted papers.
All figures use colorblind-safe palettes and 300 DPI.

Usage:
    from src.visualization.plot_utils import setup_plotting, save_figure, get_color
    setup_plotting()
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ...
    save_figure(fig, "my_figure")
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import VISUALIZATION, PATHS


# ============================================================
# Wong Colorblind-Safe Palette
# ============================================================
# From: Bang Wong, Nature Methods 8, 441 (2011)

COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "sky_blue": "#56B4E9",
    "orange": "#E69F00",
    "black": "#000000",
}

# Ordered list for cycling through domains/concepts
COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["vermillion"],
    COLORS["green"],
    COLORS["purple"],
    COLORS["orange"],
    COLORS["sky_blue"],
    COLORS["yellow"],
]

# Domain-specific color assignments (consistent across all figures)
DOMAIN_COLORS = {
    # Refusal domains
    "violence": COLORS["vermillion"],
    "illegal_activity": COLORS["blue"],
    "medical_legal": COLORS["green"],
    "privacy": COLORS["purple"],
    # Honesty domains
    "factual_trivia": COLORS["sky_blue"],
    "math": COLORS["orange"],
    "politics_opinion": COLORS["vermillion"],
    "personal_advice": COLORS["green"],
    # Special
    "global": COLORS["black"],
}

# ============================================================
# Figure Size Constants (ACL template compatible)
# ============================================================

SINGLE_COLUMN = (3.25, 2.5)    # Single column width
DOUBLE_COLUMN = (6.75, 4.0)    # Double column (full page width)
SQUARE = (3.25, 3.25)          # Square figure
TALL_SINGLE = (3.25, 4.0)      # Tall single column


def get_color(domain: str, fallback_idx: int = 0) -> str:
    """Get the color assigned to a domain name.

    Args:
        domain: Domain or concept name.
        fallback_idx: Index into COLOR_CYCLE if domain not in DOMAIN_COLORS.

    Returns:
        Hex color string.
    """
    if domain in DOMAIN_COLORS:
        return DOMAIN_COLORS[domain]
    return COLOR_CYCLE[fallback_idx % len(COLOR_CYCLE)]


# ============================================================
# Setup Function
# ============================================================

def setup_plotting():
    """Configure matplotlib for publication-quality ACL figures.

    Call this once at the start of your script/notebook.
    """
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": VISUALIZATION.font_size,

        # Figure
        "figure.dpi": VISUALIZATION.dpi,
        "savefig.dpi": VISUALIZATION.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Axes
        "axes.titlesize": VISUALIZATION.title_size,
        "axes.labelsize": VISUALIZATION.label_size,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Ticks
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # Legend
        "legend.fontsize": 8,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        # Color cycle
        "axes.prop_cycle": plt.cycler(color=COLOR_CYCLE),

        # PDF/PS backend for LaTeX
        "pdf.fonttype": 42,  # TrueType fonts in PDFs
        "ps.fonttype": 42,
    })


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Optional[Path] = None,
    formats: Optional[list] = None,
    close: bool = True,
) -> list:
    """Save a figure in publication-quality format(s).

    Args:
        fig: Matplotlib figure to save.
        name: Base filename (without extension).
        output_dir: Directory to save to. Defaults to results/figures/.
        formats: List of formats to save (e.g., ["pdf", "png"]). Defaults to ["pdf", "png"].
        close: Whether to close the figure after saving.

    Returns:
        List of saved file paths.
    """
    if output_dir is None:
        output_dir = PATHS.figures
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = ["pdf", "png"]

    saved = []
    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, bbox_inches="tight", pad_inches=0.05)
        saved.append(filepath)

    if close:
        plt.close(fig)

    return saved


def format_domain_name(domain: str) -> str:
    """Convert internal domain name to display-friendly label.

    E.g., "illegal_activity" -> "Illegal Activity"
    """
    return domain.replace("_", " ").title()


def add_significance_stars(
    ax: plt.Axes,
    x: float,
    y: float,
    p_value: float,
    fontsize: int = 8,
):
    """Add significance stars to a plot.

    Args:
        ax: Matplotlib axes.
        x, y: Position for the annotation.
        p_value: P-value to determine significance level.
        fontsize: Font size for the stars.
    """
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "n.s."

    ax.annotate(
        stars, (x, y),
        ha="center", va="bottom",
        fontsize=fontsize,
    )
