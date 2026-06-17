"""
bootstrap.py — Bootstrap Confidence Intervals
==============================================
Computes bootstrap confidence intervals for effect sizes in the paper.
Used for: cosine similarities, angular dispersions, refusal rate deltas,
and Base vs. Instruct comparison gaps.

Minimum 1000 bootstrap samples as specified in the project plan.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations.
        statistic_fn: Function that computes the statistic from a sample.
        n_bootstrap: Number of bootstrap resamples (minimum 1000 per plan).
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
          - "point_estimate": The statistic computed on the full data.
          - "ci_lower": Lower bound of the CI.
          - "ci_upper": Upper bound of the CI.
          - "se": Standard error of the bootstrap distribution.
          - "n_bootstrap": Number of resamples used.
    """
    rng = np.random.RandomState(seed)
    n = len(data)

    # Point estimate
    point_estimate = statistic_fn(data)

    # Bootstrap resamples
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_fn(resample)

    # Percentile CI
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    se = float(np.std(bootstrap_stats))

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_difference(
    data_a: np.ndarray,
    data_b: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute a bootstrap CI for the difference in a statistic between two groups.

    Useful for: Base vs. Instruct comparisons, domain A vs. domain B effects.

    Args:
        data_a: Observations from group A.
        data_b: Observations from group B.
        statistic_fn: Function to compute the statistic on each group.
        n_bootstrap: Number of resamples.
        ci_level: Confidence level.
        seed: Random seed.

    Returns:
        Dict with keys:
          - "point_estimate_a", "point_estimate_b": Group-level estimates.
          - "difference": B - A.
          - "ci_lower", "ci_upper": CI for the difference.
          - "p_value_approx": Approximate p-value (fraction of resamples with reversed sign).
    """
    rng = np.random.RandomState(seed)
    n_a = len(data_a)
    n_b = len(data_b)

    est_a = statistic_fn(data_a)
    est_b = statistic_fn(data_b)
    observed_diff = est_b - est_a

    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample_a = rng.choice(data_a, size=n_a, replace=True)
        resample_b = rng.choice(data_b, size=n_b, replace=True)
        bootstrap_diffs[i] = statistic_fn(resample_b) - statistic_fn(resample_a)

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))

    # Approximate two-sided p-value
    if observed_diff >= 0:
        p_value = float(np.mean(bootstrap_diffs <= 0))
    else:
        p_value = float(np.mean(bootstrap_diffs >= 0))
    p_value = min(2 * p_value, 1.0)  # Two-sided

    return {
        "point_estimate_a": float(est_a),
        "point_estimate_b": float(est_b),
        "difference": float(observed_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.std(bootstrap_diffs)),
        "p_value_approx": p_value,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_cosine_similarity_ci(
    cos_sims: List[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for the mean cosine similarity across domains.

    This is the primary statistic for measuring angular dispersion.

    Args:
        cos_sims: List of cosine similarity values (one per domain).
        n_bootstrap: Number of resamples.
        ci_level: Confidence level.
        seed: Random seed.

    Returns:
        Bootstrap CI result dict.
    """
    return bootstrap_ci(
        data=np.array(cos_sims),
        statistic_fn=np.mean,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )


def bootstrap_angular_dispersion_ci(
    cos_sims: List[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for angular dispersion (std of angles in degrees).

    Args:
        cos_sims: List of cosine similarities (domain vs. global).
        n_bootstrap: Number of resamples.
        ci_level: Confidence level.
        seed: Random seed.

    Returns:
        Bootstrap CI result dict where the statistic is std(angles_degrees).
    """
    def angle_std(sims):
        angles = np.degrees(np.arccos(np.clip(sims, -1.0, 1.0)))
        return np.std(angles)

    return bootstrap_ci(
        data=np.array(cos_sims),
        statistic_fn=angle_std,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )
