"""
test_extraction.py — Smoke Tests for the Extraction Pipeline
=============================================================
Tests that don't require a GPU or model download.
Validates data loading, caching, direction computation, and analysis
on synthetic/dummy data.

Run: python -m pytest tests/ -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Test Data Loading
# ============================================================

class TestPromptPairLoading:
    """Tests for loading contrastive prompt pairs from JSONL files."""

    def _create_temp_jsonl(self, pairs, tmp_path):
        """Helper: write pairs to a temporary JSONL file."""
        filepath = tmp_path / "test_pairs.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        return filepath

    def test_load_valid_pairs(self, tmp_path):
        """Load a well-formed JSONL file with 3 pairs."""
        from src.extraction.extract_activations import load_prompt_pairs

        pairs = [
            {"positive": "Yes, here's how...", "negative": "I can't help with that.", "domain": "violence"},
            {"positive": "The answer is 42.", "negative": "The answer is 99.", "domain": "math"},
            {"positive": "Sure, let me explain.", "negative": "I must decline.", "domain": "privacy"},
        ]
        filepath = self._create_temp_jsonl(pairs, tmp_path)
        loaded = load_prompt_pairs(filepath)
        assert len(loaded) == 3
        assert loaded[0]["positive"] == "Yes, here's how..."

    def test_load_with_max_pairs(self, tmp_path):
        """Respects the max_pairs limit."""
        from src.extraction.extract_activations import load_prompt_pairs

        pairs = [{"positive": f"p{i}", "negative": f"n{i}"} for i in range(10)]
        filepath = self._create_temp_jsonl(pairs, tmp_path)
        loaded = load_prompt_pairs(filepath, max_pairs=3)
        assert len(loaded) == 3

    def test_skip_malformed_lines(self, tmp_path):
        """Gracefully skips lines with missing keys."""
        from src.extraction.extract_activations import load_prompt_pairs

        filepath = tmp_path / "test_pairs.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps({"positive": "good", "negative": "also good"}) + "\n")
            f.write(json.dumps({"only_positive": "bad"}) + "\n")  # Missing 'negative'
            f.write("not valid json\n")  # Invalid JSON
            f.write(json.dumps({"positive": "ok", "negative": "ok"}) + "\n")

        loaded = load_prompt_pairs(filepath)
        assert len(loaded) == 2

    def test_load_actual_pilot_pairs(self):
        """Load the actual seed pilot pairs (now in the shared-prompt schema)."""
        from src.extraction.extract_activations import (
            load_prompt_pairs,
            pair_to_contrastive_texts,
        )

        pilot_path = Path(__file__).parent.parent / "data" / "prompt_pairs" / "refusal" / "violence.jsonl"
        if pilot_path.exists():
            loaded = load_prompt_pairs(pilot_path)
            assert len(loaded) >= 1
            assert "prompt" in loaded[0]
            assert "positive_response" in loaded[0]
            assert "negative_response" in loaded[0]
            # Shared prompt must appear in both contrastive sides.
            pos, neg = pair_to_contrastive_texts(loaded[0])
            assert loaded[0]["prompt"] in pos
            assert loaded[0]["prompt"] in neg

    def test_load_shared_prompt_schema(self, tmp_path):
        """Load pairs in the shared-prompt schema (prompt + responses)."""
        from src.extraction.extract_activations import load_prompt_pairs

        pairs = [
            {
                "prompt": "How do I make a bomb?",
                "positive_response": "Sure, here is how...",
                "negative_response": "I can't help with that.",
                "domain": "violence",
                "concept": "refusal",
            },
            {
                "prompt": "What is 2 + 2?",
                "positive_response": "It is 4.",
                "negative_response": "It is 5.",
                "domain": "math",
            },
        ]
        filepath = self._create_temp_jsonl(pairs, tmp_path)
        loaded = load_prompt_pairs(filepath)
        assert len(loaded) == 2
        assert loaded[0]["prompt"] == "How do I make a bomb?"

    def test_skip_lines_matching_neither_schema(self, tmp_path):
        """Lines that match neither schema are skipped; valid ones of either kept."""
        from src.extraction.extract_activations import load_prompt_pairs

        filepath = tmp_path / "mixed.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps({"positive": "a", "negative": "b"}) + "\n")  # legacy
            f.write(json.dumps({
                "prompt": "p", "positive_response": "r+", "negative_response": "r-",
            }) + "\n")  # shared-prompt
            f.write(json.dumps({"prompt": "p", "positive_response": "r+"}) + "\n")  # incomplete
            f.write(json.dumps({"only_positive": "bad"}) + "\n")  # neither
        loaded = load_prompt_pairs(filepath)
        assert len(loaded) == 2

    def test_pair_to_contrastive_texts_both_schemas(self):
        """Resolve both schemas to (positive_text, negative_text)."""
        from src.extraction.extract_activations import (
            pair_to_contrastive_texts,
            PROMPT_RESPONSE_SEPARATOR,
        )

        legacy = {"positive": "yes", "negative": "no"}
        assert pair_to_contrastive_texts(legacy) == ("yes", "no")

        shared = {
            "prompt": "Q?",
            "positive_response": "good",
            "negative_response": "bad",
        }
        sep = PROMPT_RESPONSE_SEPARATOR
        assert pair_to_contrastive_texts(shared) == (f"Q?{sep}good", f"Q?{sep}bad")

        # Blank prompt falls back to bare responses.
        shared_blank = {"prompt": "  ", "positive_response": "g", "negative_response": "b"}
        assert pair_to_contrastive_texts(shared_blank) == ("g", "b")

        with pytest.raises(KeyError):
            pair_to_contrastive_texts({"foo": "bar"})

    def test_load_actual_advbench_pairs(self):
        """Load the real AdvBench-derived shared-prompt pairs if present."""
        from src.extraction.extract_activations import (
            load_prompt_pairs,
            pair_to_contrastive_texts,
        )

        path = (
            Path(__file__).parent.parent
            / "data" / "prompt_pairs" / "refusal_new" / "violence.jsonl"
        )
        if path.exists():
            loaded = load_prompt_pairs(path, max_pairs=5)
            assert len(loaded) >= 1
            assert "prompt" in loaded[0]
            pos, neg = pair_to_contrastive_texts(loaded[0])
            assert loaded[0]["prompt"] in pos
            assert loaded[0]["prompt"] in neg


# ============================================================
# Test Prompt-Pair Validation
# ============================================================

class TestPromptPairValidation:
    """Tests for scripts/validate_prompt_pairs.py."""

    def _write(self, lines, tmp_path, name="pairs.jsonl"):
        filepath = tmp_path / name
        with open(filepath, "w", encoding="utf-8") as f:
            for line in lines:
                f.write((json.dumps(line) if isinstance(line, dict) else line) + "\n")
        return filepath

    def test_clean_file_has_no_issues(self, tmp_path):
        from scripts.validate_prompt_pairs import validate_file

        path = self._write([
            {"prompt": "Q1?", "positive_response": "yes", "negative_response": "no",
             "concept": "refusal", "domain": "violence", "source": "advbench"},
            {"positive": "a", "negative": "b",
             "concept": "honesty", "domain": "math", "source": "hand-written"},
        ], tmp_path)
        r = validate_file(path)
        assert r.n_valid == 2
        assert not r.errors and not r.warnings
        assert r.schemas["shared"] == 1 and r.schemas["legacy"] == 1

    def test_invalid_json_and_bad_schema_are_errors(self, tmp_path):
        from scripts.validate_prompt_pairs import validate_file

        path = self._write([
            {"positive": "a", "negative": "b", "concept": "honesty", "domain": "math"},
            "not valid json",
            {"only_positive": "x"},
        ], tmp_path)
        r = validate_file(path)
        assert r.n_valid == 1
        assert len(r.errors) == 2

    def test_empty_text_is_error(self, tmp_path):
        from scripts.validate_prompt_pairs import validate_file

        path = self._write([
            {"prompt": "", "positive_response": "  ", "negative_response": "",
             "concept": "refusal", "domain": "violence"},
        ], tmp_path)
        r = validate_file(path)
        assert r.n_valid == 0
        assert any("empty" in e for e in r.errors)

    def test_duplicate_pair_is_warning(self, tmp_path):
        from scripts.validate_prompt_pairs import validate_file

        dup = {"positive": "same", "negative": "diff",
               "concept": "honesty", "domain": "math", "source": "hand-written"}
        path = self._write([dup, dict(dup)], tmp_path)
        r = validate_file(path)
        assert r.n_valid == 2
        assert any("duplicate" in w for w in r.warnings)

    def test_unknown_domain_is_warning(self, tmp_path):
        from scripts.validate_prompt_pairs import validate_file

        path = self._write([
            {"positive": "a", "negative": "b",
             "concept": "refusal", "domain": "not_a_domain"},
        ], tmp_path)
        r = validate_file(path)
        assert r.n_valid == 1
        assert any("not configured" in w for w in r.warnings)

    def test_low_response_diversity_warns_on_shared_schema(self, tmp_path):
        """Distinct prompts but 1 canned refusal => low negative diversity warning.

        Guards the subtle case: the joined prompt+response text is unique
        (prompts differ), but the *response* is templated. Diversity must be
        measured on the response component, not the joined text.
        """
        from scripts.validate_prompt_pairs import validate_file

        rows = [
            {"prompt": f"Harmful request number {i}?",
             "positive_response": f"Sure, here is harmful thing {i}.",
             "negative_response": "I can't help with that.",  # one canned refusal
             "concept": "refusal", "domain": "violence", "source": "advbench"}
            for i in range(20)
        ]
        r = validate_file(self._write(rows, tmp_path))
        assert r.n_valid == 20
        assert r.n_unique_neg == 1            # response component, not joined text
        assert r.n_unique_pos == 20
        assert any("low negative diversity" in w for w in r.warnings)
        assert not any("low positive diversity" in w for w in r.warnings)


# ============================================================
# Test Activation Caching
# ============================================================

class TestCaching:
    """Tests for saving and loading activation tensors."""

    def test_save_and_load_activations(self, tmp_path):
        """Round-trip test: save activations, load them back, verify equality."""
        from src.extraction.cache_utils import save_activations, load_activations

        activations = {
            0: torch.randn(5, 128),   # 5 pairs, d_model=128
            5: torch.randn(5, 128),
            10: torch.randn(5, 128),
        }

        save_activations(
            activations, tmp_path,
            model_name="test-model",
            concept="refusal",
            domain="violence",
            prefix="pos_",
        )

        loaded = load_activations(
            tmp_path,
            model_name="test-model",
            concept="refusal",
            domain="violence",
            prefix="pos_",
        )

        assert set(loaded.keys()) == {0, 5, 10}
        for layer in activations:
            torch.testing.assert_close(loaded[layer], activations[layer])

    def test_load_specific_layers(self, tmp_path):
        """Loading a subset of layers works correctly."""
        from src.extraction.cache_utils import save_activations, load_activations

        activations = {i: torch.randn(3, 64) for i in range(10)}
        save_activations(activations, tmp_path, "test", "refusal", "violence", "pos_")

        loaded = load_activations(
            tmp_path, "test", "refusal", "violence",
            layers=[0, 5, 9], prefix="pos_",
        )
        assert set(loaded.keys()) == {0, 5, 9}

    def test_direction_save_load(self, tmp_path):
        """Round-trip test for direction vectors."""
        from src.extraction.cache_utils import save_direction, load_direction

        direction = torch.randn(256)
        direction = direction / torch.norm(direction)  # Normalize

        save_direction(direction, tmp_path, "test", "refusal", "violence", layer=5)
        loaded = load_direction(tmp_path, "test", "refusal", "violence", layer=5)

        torch.testing.assert_close(loaded, direction)


# ============================================================
# Test Direction Computation
# ============================================================

class TestDirections:
    """Tests for difference-of-means direction computation."""

    def test_difference_of_means_basic(self):
        """Direction should point from negative mean to positive mean."""
        from src.analysis.directions import difference_of_means

        pos = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        neg = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]])

        direction = difference_of_means(pos, neg, normalize=True)

        # Should point in the positive x direction
        assert direction[0].item() > 0.99
        assert abs(direction[1].item()) < 0.01

    def test_normalized_direction_is_unit_vector(self):
        """Normalized direction should have L2 norm ≈ 1."""
        from src.analysis.directions import difference_of_means

        pos = torch.randn(50, 128)
        neg = torch.randn(50, 128)

        direction = difference_of_means(pos, neg, normalize=True)
        norm = torch.norm(direction)

        assert abs(norm.item() - 1.0) < 1e-5

    def test_unnormalized_direction_magnitude(self):
        """Unnormalized direction should preserve the actual difference magnitude."""
        from src.analysis.directions import difference_of_means

        pos = torch.ones(10, 64) * 3.0
        neg = torch.ones(10, 64) * 1.0

        direction = difference_of_means(pos, neg, normalize=False)
        expected = torch.ones(64) * 2.0

        torch.testing.assert_close(direction, expected)


# ============================================================
# Test Angular Dispersion
# ============================================================

class TestAngularDispersion:
    """Tests for cosine similarity and angular dispersion computation."""

    def test_cosine_similarity_identical(self):
        """Identical vectors should have cosine similarity = 1.0."""
        from src.analysis.angular_dispersion import cosine_similarity

        v = torch.randn(128)
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have cosine similarity ≈ 0."""
        from src.analysis.angular_dispersion import cosine_similarity

        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim) < 1e-5

    def test_cosine_similarity_opposite(self):
        """Opposite vectors should have cosine similarity = -1.0."""
        from src.analysis.angular_dispersion import cosine_similarity

        v = torch.randn(64)
        sim = cosine_similarity(v, -v)
        assert abs(sim + 1.0) < 1e-5

    def test_pairwise_is_symmetric(self):
        """Pairwise cosine similarity matrix should be symmetric."""
        from src.analysis.angular_dispersion import pairwise_cosine_similarity

        directions = {
            "a": torch.randn(64),
            "b": torch.randn(64),
            "c": torch.randn(64),
        }
        sims = pairwise_cosine_similarity(directions)

        assert abs(sims[("a", "b")] - sims[("b", "a")]) < 1e-5
        assert abs(sims[("a", "a")] - 1.0) < 1e-5


# ============================================================
# Test Bootstrap
# ============================================================

class TestBootstrap:
    """Tests for bootstrap confidence intervals."""

    def test_ci_contains_point_estimate(self):
        """The CI should contain the point estimate."""
        from src.analysis.bootstrap import bootstrap_ci

        data = np.random.randn(100)
        result = bootstrap_ci(data, np.mean, n_bootstrap=500, seed=42)

        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_ci_width_decreases_with_more_data(self):
        """More data should produce narrower CIs."""
        from src.analysis.bootstrap import bootstrap_ci

        small = np.random.randn(10)
        large = np.random.randn(1000)

        ci_small = bootstrap_ci(small, np.mean, n_bootstrap=500, seed=42)
        ci_large = bootstrap_ci(large, np.mean, n_bootstrap=500, seed=42)

        width_small = ci_small["ci_upper"] - ci_small["ci_lower"]
        width_large = ci_large["ci_upper"] - ci_large["ci_lower"]

        assert width_large < width_small

    def test_bootstrap_difference(self):
        """Test bootstrap difference between two groups."""
        from src.analysis.bootstrap import bootstrap_difference

        # Group B has higher mean
        group_a = np.random.randn(50)
        group_b = np.random.randn(50) + 2.0

        result = bootstrap_difference(group_a, group_b, np.mean, n_bootstrap=500, seed=42)

        # Difference should be positive (B > A)
        assert result["difference"] > 0
        assert result["ci_lower"] > 0  # Significant effect


# ============================================================
# Test Hooks (without model loading)
# ============================================================

class TestHooks:
    """Tests for the hook infrastructure (no model required)."""

    def test_residual_stream_hook_last_token(self):
        """ResidualStreamHook correctly extracts the last token position."""
        from src.extraction.hooks import ResidualStreamHook

        hook = ResidualStreamHook(token_position="last")
        # Simulate activations: batch=2, seq_len=5, d_model=8
        hook.activations[0] = torch.randn(2, 5, 8)

        result = hook.get_activation_at_position(0)
        assert result.shape == (2, 8)
        torch.testing.assert_close(result, hook.activations[0][:, -1, :])

    def test_residual_stream_hook_first_token(self):
        """ResidualStreamHook correctly extracts the first token position."""
        from src.extraction.hooks import ResidualStreamHook

        hook = ResidualStreamHook(token_position="first")
        hook.activations[0] = torch.randn(3, 10, 16)

        result = hook.get_activation_at_position(0)
        assert result.shape == (3, 16)
        torch.testing.assert_close(result, hook.activations[0][:, 0, :])

    def test_residual_stream_hook_mean(self):
        """ResidualStreamHook correctly computes mean across positions."""
        from src.extraction.hooks import ResidualStreamHook

        hook = ResidualStreamHook(token_position="mean")
        act = torch.ones(2, 4, 8)  # All ones
        hook.activations[0] = act

        result = hook.get_activation_at_position(0)
        assert result.shape == (2, 8)
        torch.testing.assert_close(result, torch.ones(2, 8))

    def test_residual_stream_hook_with_attention_mask(self):
        """Last token extraction respects attention mask (padding)."""
        from src.extraction.hooks import ResidualStreamHook

        hook = ResidualStreamHook(token_position="last")
        # batch=2, seq_len=5, d_model=4
        act = torch.zeros(2, 5, 4)
        act[0, 2, :] = 1.0  # Last real token for example 0 (len=3)
        act[1, 4, :] = 2.0  # Last real token for example 1 (len=5)
        hook.activations[0] = act

        mask = torch.tensor([
            [1, 1, 1, 0, 0],  # 3 real tokens
            [1, 1, 1, 1, 1],  # 5 real tokens
        ])

        result = hook.get_activation_at_position(0, attention_mask=mask)
        assert result.shape == (2, 4)
        torch.testing.assert_close(result[0], torch.ones(4))
        torch.testing.assert_close(result[1], torch.ones(4) * 2.0)

    def test_clear_activations(self):
        """Clear should remove all stored activations."""
        from src.extraction.hooks import ResidualStreamHook

        hook = ResidualStreamHook()
        hook.activations[0] = torch.randn(1, 1, 1)
        hook.activations[5] = torch.randn(1, 1, 1)

        hook.clear()
        assert len(hook.activations) == 0


# ============================================================
# Test Config
# ============================================================

class TestConfig:
    """Tests for configuration module."""

    def test_models_defined(self):
        """All expected models should be in the MODELS dict."""
        from config import MODELS

        assert "gemma-2-2b" in MODELS
        assert "gemma-2-2b-it" in MODELS
        assert "qwen-2.5-3b-instruct" in MODELS
        assert "qwen-2.5-3b" in MODELS

    def test_concepts_defined(self):
        """All expected concepts should be in CONCEPTS."""
        from config import CONCEPTS

        assert "refusal" in CONCEPTS
        assert "honesty" in CONCEPTS
        assert len(CONCEPTS["refusal"].domains) == 4
        assert len(CONCEPTS["honesty"].domains) == 4

    def test_activation_filename_format(self):
        """Activation filenames follow the expected pattern."""
        from config import activation_filename

        name = activation_filename("qwen-2.5-3b-instruct", "refusal", "violence", 12)
        assert name == "qwen_25_3b_instruct_refusal_violence_layer012.pt"

    def test_paths_exist_or_can_be_created(self):
        """ProjectPaths.ensure_all() should not raise."""
        from config import PATHS
        # Don't actually create — just verify the method exists
        assert hasattr(PATHS, "ensure_all")
