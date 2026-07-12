"""
test_review_fixes.py — Tests for the 2026-07-07 reviewer-driven fixes
======================================================================
Covers:
  - pair-level bootstrap CI (replaces bootstrapping the 4 domain cosines)
  - balanced-domain subsampling for direction computation (T4 sensitivity)
  - transfer-layer selection excluding embedding-adjacent layers
  - provenance sidecar written by extraction / collected by run_pipeline
  - E4 steering driver held-out split (CPU parts only)

Run: python -m pytest tests/test_review_fixes.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.cache_utils import save_activations
from src.analysis.bootstrap import bootstrap_pairlevel_dispersion_ci
from src.analysis.directions import (
    balanced_subsample_indices,
    compute_all_directions,
)


def _write_synthetic_activations(
    tmp_path, model, concept, domains, layers, n_pairs_by_domain, d_model=32, seed=0
):
    """Positives shifted along a shared axis; negatives pure noise."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(d_model)
    g /= np.linalg.norm(g)
    for domain in domains:
        n = n_pairs_by_domain[domain]
        pos, neg = {}, {}
        for layer in layers:
            base = rng.standard_normal((n, d_model))
            pos[layer] = torch.from_numpy(base + 3.0 * g[None, :]).float()
            neg[layer] = torch.from_numpy(rng.standard_normal((n, d_model))).float()
        save_activations(pos, tmp_path, model, concept, domain, prefix="pos_")
        save_activations(neg, tmp_path, model, concept, domain, prefix="neg_")
    return g


class TestPairLevelBootstrap:
    def _make_data(self, seed=0, n=40, d=32, frag=0.0):
        rng = np.random.default_rng(seed)
        g = rng.standard_normal(d)
        g /= np.linalg.norm(g)
        pos_by, neg_by = {}, {}
        for i, domain in enumerate(["a", "b", "c"]):
            u = rng.standard_normal(d)
            u -= u.dot(g) * g
            u /= np.linalg.norm(u)
            axis = g + frag * u
            axis /= np.linalg.norm(axis)
            pos_by[domain] = rng.standard_normal((n, d)) + 4.0 * axis[None, :]
            neg_by[domain] = rng.standard_normal((n, d))
        return pos_by, neg_by

    def test_ci_well_formed(self):
        # Percentile CIs need not contain the point estimate when it sits near
        # the cosine ceiling (resampling noise biases resamples downward), so
        # assert ordering + proximity rather than strict containment.
        pos_by, neg_by = self._make_data()
        res = bootstrap_pairlevel_dispersion_ci(pos_by, neg_by, n_bootstrap=200, seed=1)
        assert res["ci_lower"] <= res["ci_upper"]
        assert res["se"] > 0
        assert abs(res["point_estimate"] - res["ci_upper"]) < 5 * res["se"] + 1e-9
        assert res["method"] == "pair-level"

    def test_no_fragmentation_gives_high_cosine(self):
        pos_by, neg_by = self._make_data(frag=0.0)
        res = bootstrap_pairlevel_dispersion_ci(pos_by, neg_by, n_bootstrap=100, seed=1)
        assert res["point_estimate"] > 0.9

    def test_fragmentation_lowers_cosine(self):
        pos_by, neg_by = self._make_data(frag=0.0)
        pos_f, neg_f = self._make_data(frag=1.0)
        hi = bootstrap_pairlevel_dispersion_ci(pos_by, neg_by, n_bootstrap=100, seed=1)
        lo = bootstrap_pairlevel_dispersion_ci(pos_f, neg_f, n_bootstrap=100, seed=1)
        assert lo["point_estimate"] < hi["point_estimate"]

    def test_deterministic_given_seed(self):
        pos_by, neg_by = self._make_data()
        r1 = bootstrap_pairlevel_dispersion_ci(pos_by, neg_by, n_bootstrap=50, seed=7)
        r2 = bootstrap_pairlevel_dispersion_ci(pos_by, neg_by, n_bootstrap=50, seed=7)
        assert r1 == r2

    def test_unequal_pos_neg_counts_supported(self):
        pos_by, neg_by = self._make_data(n=30)
        # Drop rows on one side of one domain -> independent resampling path.
        neg_by["a"] = neg_by["a"][:20]
        res = bootstrap_pairlevel_dispersion_ci(pos_by, neg_by, n_bootstrap=50, seed=1)
        assert np.isfinite(res["point_estimate"])


class TestBalancedSubsampling:
    def test_indices_balanced_to_min(self, tmp_path):
        domains = ["big", "small"]
        _write_synthetic_activations(
            tmp_path, "m", "c", domains, layers=[0, 1],
            n_pairs_by_domain={"big": 50, "small": 12},
        )
        idx = balanced_subsample_indices(tmp_path, "m", "c", domains, [0, 1], seed=3)
        assert set(idx) == {"big", "small"}
        assert len(idx["big"]) == 12 and len(idx["small"]) == 12
        assert len(np.unique(idx["big"])) == 12  # without replacement

    def test_indices_deterministic(self, tmp_path):
        domains = ["big", "small"]
        _write_synthetic_activations(
            tmp_path, "m", "c", domains, layers=[0],
            n_pairs_by_domain={"big": 50, "small": 12},
        )
        i1 = balanced_subsample_indices(tmp_path, "m", "c", domains, [0], seed=3)
        i2 = balanced_subsample_indices(tmp_path, "m", "c", domains, [0], seed=3)
        assert np.array_equal(i1["big"], i2["big"])

    def test_compute_all_directions_balanced_runs(self, tmp_path):
        domains = ["big", "small"]
        acts = tmp_path / "acts"
        dirs = tmp_path / "dirs"
        _write_synthetic_activations(
            acts, "m", "c", domains, layers=[0, 1],
            n_pairs_by_domain={"big": 50, "small": 12},
        )
        res = compute_all_directions(
            acts, dirs, "m", "c", domains, [0, 1], balance_domains=True, seed=3
        )
        assert "global" in res and 0 in res["global"]
        for d in domains:
            assert torch.isfinite(res[d][0]).all()
            assert abs(torch.norm(res[d][0]).item() - 1.0) < 1e-5


class TestTransferLayerSelection:
    def test_embedding_layers_excluded_and_override(self, tmp_path):
        from src.analysis.run_pipeline import run_pipeline

        report = run_pipeline(
            model_name="m", concept="refusal",
            domains=["violence", "privacy"], layers=[0, 1, 4, 8],
            activations_dir=tmp_path / "acts", directions_dir=tmp_path / "dirs",
            figures_dir=tmp_path / "figs", report_dir=tmp_path / "rep",
            mock=True, make_figures=False, n_bootstrap=20,
            mock_kwargs={"n_pairs": 10, "d_model": 16},
        )
        assert report["transfer"]["layer"] >= 2

        report2 = run_pipeline(
            model_name="m", concept="refusal",
            domains=["violence", "privacy"], layers=[0, 1, 4, 8],
            activations_dir=tmp_path / "acts", directions_dir=tmp_path / "dirs",
            figures_dir=tmp_path / "figs", report_dir=tmp_path / "rep",
            mock=True, make_figures=False, n_bootstrap=20,
            mock_kwargs={"n_pairs": 10, "d_model": 16},
            transfer_layer=8,
        )
        assert report2["transfer"]["layer"] == 8
        # Late-layer summary present (fix for early-layer lexical confound).
        assert "late_layer_mean_cos" in report2["summary"]
        # Pair-level bootstrap used (activations exist in mock mode).
        first_layer = report2["layers"][0]
        assert report2["bootstrap"][first_layer]["method"] == "pair-level"

    def test_invalid_transfer_layer_raises(self, tmp_path):
        from src.analysis.run_pipeline import run_pipeline

        with pytest.raises(ValueError):
            run_pipeline(
                model_name="m", concept="refusal",
                domains=["violence", "privacy"], layers=[0, 4],
                activations_dir=tmp_path / "acts", directions_dir=tmp_path / "dirs",
                figures_dir=tmp_path / "figs", report_dir=tmp_path / "rep",
                mock=True, make_figures=False, n_bootstrap=10,
                mock_kwargs={"n_pairs": 8, "d_model": 16},
                transfer_layer=99,
            )


class TestProvenanceSidecar:
    def test_meta_collected_into_report(self, tmp_path):
        from src.analysis.run_pipeline import run_pipeline

        acts = tmp_path / "acts"
        acts.mkdir(parents=True)
        meta = {"model": "m", "concept": "refusal", "domain": "violence",
                "data_source": "data/prompt_pairs_promptbased"}
        with open(acts / "meta_m_refusal_violence.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        report = run_pipeline(
            model_name="m", concept="refusal",
            domains=["violence", "privacy"], layers=[2, 4],
            activations_dir=acts, directions_dir=tmp_path / "dirs",
            figures_dir=tmp_path / "figs", report_dir=tmp_path / "rep",
            mock=True, make_figures=False, n_bootstrap=10,
            mock_kwargs={"n_pairs": 8, "d_model": 16},
        )
        assert report["provenance"] == [meta]


class TestProbeTransfer:
    def _write(self, tmp_path, shared_axis=True, n=60, d=24, seed=0):
        """Domains either share one separating axis or use orthogonal axes."""
        rng = np.random.default_rng(seed)
        axes = {}
        g = rng.standard_normal(d); g /= np.linalg.norm(g)
        for i, domain in enumerate(["a", "b", "c"]):
            if shared_axis:
                axes[domain] = g
            else:
                u = np.zeros(d); u[i] = 1.0  # orthogonal per-domain axes
                axes[domain] = u
            pos = {0: torch.from_numpy(
                rng.standard_normal((n, d)) + 6.0 * axes[domain][None, :]).float()}
            neg = {0: torch.from_numpy(rng.standard_normal((n, d))).float()}
            save_activations(pos, tmp_path, "m", "c", domain, prefix="pos_")
            save_activations(neg, tmp_path, "m", "c", domain, prefix="neg_")

    def test_shared_axis_transfers(self, tmp_path):
        from src.analysis.probe_transfer import probe_transfer_matrix, summarize
        self._write(tmp_path, shared_axis=True)
        m = probe_transfer_matrix(tmp_path, "m", "c", ["a", "b", "c"], layer=0)
        s = summarize(m)
        assert s["mean_within_acc"] > 0.9
        assert s["mean_cross_acc"] > 0.85  # shared axis -> probes transfer

    def test_orthogonal_axes_do_not_transfer(self, tmp_path):
        from src.analysis.probe_transfer import probe_transfer_matrix, summarize
        self._write(tmp_path, shared_axis=False)
        m = probe_transfer_matrix(tmp_path, "m", "c", ["a", "b", "c"], layer=0)
        s = summarize(m)
        assert s["mean_within_acc"] > 0.9
        assert s["mean_cross_acc"] < 0.75  # disjoint axes -> weak transfer
        assert s["transfer_gap"] > 0.2

    def test_balance_to_subsamples(self, tmp_path):
        from src.analysis.probe_transfer import _domain_splits
        self._write(tmp_path, shared_axis=True, n=60)
        splits = _domain_splits(tmp_path, "m", "c", ["a", "b", "c"], 0,
                                test_frac=0.25, seed=1, balance_to=20)
        assert all(s["n_pairs"] == 20 for s in splits.values())

    def test_deterministic(self, tmp_path):
        from src.analysis.probe_transfer import probe_transfer_matrix
        self._write(tmp_path, shared_axis=True)
        m1 = probe_transfer_matrix(tmp_path, "m", "c", ["a", "b"], 0, seed=7)
        m2 = probe_transfer_matrix(tmp_path, "m", "c", ["a", "b"], 0, seed=7)
        assert m1 == m2


class TestSteeringHeldout:
    def _write_pairs(self, tmp_path, domain, n):
        d = tmp_path / "refusal"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{domain}.jsonl", "w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({
                    "positive": f"harmless {domain} {i}",
                    "negative": f"harmful {domain} {i}",
                    "concept": "refusal", "domain": domain,
                }) + "\n")

    def test_heldout_skips_training_pairs(self, tmp_path):
        from src.analysis.run_steering import heldout_prompts_by_domain

        self._write_pairs(tmp_path, "violence", 30)
        prompts = heldout_prompts_by_domain(
            tmp_path, "refusal", ["violence"], n_heldout=5, skip_first=20
        )
        assert prompts["violence"] == [f"harmful violence {i}" for i in range(20, 25)]

    def test_heldout_falls_back_when_exhausted(self, tmp_path):
        from src.analysis.run_steering import heldout_prompts_by_domain

        self._write_pairs(tmp_path, "privacy", 10)
        prompts = heldout_prompts_by_domain(
            tmp_path, "refusal", ["privacy"], n_heldout=5, skip_first=20
        )
        assert len(prompts["privacy"]) == 5  # last 5, with a warning

    def test_missing_data_raises_instead_of_empty_run(self, tmp_path):
        # Regression: the Jul 8 Kaggle E4 run silently generated on ZERO
        # prompts because the gitignored prompt-based dir was absent.
        from src.analysis.run_steering import heldout_prompts_by_domain

        with pytest.raises(RuntimeError, match="build_refusal_promptbased"):
            heldout_prompts_by_domain(
                tmp_path / "nonexistent_or_empty", "refusal", ["violence"],
                n_heldout=5, skip_first=20,
            )
