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


def bootstrap_pairlevel_dispersion_ci(
    pos_by_domain: Dict[str, "np.ndarray"],
    neg_by_domain: Dict[str, "np.ndarray"],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Pair-level bootstrap CI for the mean domain-vs-global cosine similarity.

    Resamples *prompt pairs* (not the handful of per-domain cosine values):
    each resample draws pairs with replacement within every domain, recomputes
    the per-domain and global difference-of-means directions from the resampled
    activations, and recomputes the mean cosine-to-global. The resulting CI
    reflects sampling variability in the underlying data — unlike bootstrapping
    the 4 domain cosines, which has too few observations to be meaningful.

    Positive and negative activations are paired by row index (same prompt
    pair), so a resample uses the same indices for both sides when the counts
    match; otherwise the sides are resampled independently.

    Args:
        pos_by_domain: {domain: array (n_pairs_d, d_model)} positive activations.
        neg_by_domain: {domain: array (n_pairs_d, d_model)} negative activations.
        n_bootstrap: Number of resamples.
        ci_level: Confidence level.
        seed: Random seed.

    Returns:
        Dict with "point_estimate", "ci_lower", "ci_upper", "se", "n_bootstrap"
        for the mean cosine, plus "angle_std_point", "angle_std_ci_lower",
        "angle_std_ci_upper" for the dispersion (std of angles, degrees), and
        "method": "pair-level".
    """
    domains = sorted(pos_by_domain.keys())
    pos = {d: np.asarray(pos_by_domain[d], dtype=np.float64) for d in domains}
    neg = {d: np.asarray(neg_by_domain[d], dtype=np.float64) for d in domains}

    def stats(idx_map=None) -> Tuple[float, float]:
        all_pos, all_neg, cos_vals = [], [], []
        global_parts_pos, global_parts_neg = [], []
        domain_dirs = {}
        for d in domains:
            p, n = pos[d], neg[d]
            if idx_map is not None:
                pi, ni = idx_map[d]
                p, n = p[pi], n[ni]
            v = p.mean(axis=0) - n.mean(axis=0)
            norm = np.linalg.norm(v)
            domain_dirs[d] = v / norm if norm > 0 else v
            global_parts_pos.append(p)
            global_parts_neg.append(n)
        g = (
            np.concatenate(global_parts_pos).mean(axis=0)
            - np.concatenate(global_parts_neg).mean(axis=0)
        )
        g_norm = np.linalg.norm(g)
        if g_norm > 0:
            g = g / g_norm
        cos_vals = [float(domain_dirs[d] @ g) for d in domains]
        angles = np.degrees(np.arccos(np.clip(cos_vals, -1.0, 1.0)))
        return float(np.mean(cos_vals)), float(np.std(angles))

    point_cos, point_angle_std = stats()

    rng = np.random.RandomState(seed)
    boot_cos = np.zeros(n_bootstrap)
    boot_angle_std = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx_map = {}
        for d in domains:
            n_p, n_n = pos[d].shape[0], neg[d].shape[0]
            if n_p == n_n:
                pi = rng.randint(0, n_p, size=n_p)
                idx_map[d] = (pi, pi)  # paired resample
            else:
                idx_map[d] = (
                    rng.randint(0, n_p, size=n_p),
                    rng.randint(0, n_n, size=n_n),
                )
        boot_cos[i], boot_angle_std[i] = stats(idx_map)

    alpha = 1 - ci_level
    return {
        "point_estimate": point_cos,
        "ci_lower": float(np.percentile(boot_cos, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_cos, 100 * (1 - alpha / 2))),
        "se": float(np.std(boot_cos)),
        "n_bootstrap": n_bootstrap,
        "angle_std_point": point_angle_std,
        "angle_std_ci_lower": float(np.percentile(boot_angle_std, 100 * alpha / 2)),
        "angle_std_ci_upper": float(np.percentile(boot_angle_std, 100 * (1 - alpha / 2))),
        "method": "pair-level",
    }


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
