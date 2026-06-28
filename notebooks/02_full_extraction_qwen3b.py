# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 — Full Extraction: Qwen 2.5 3B (Phase 2)
#
# **BlackboxNLP 2026** — Phase 2 full run, on **Kaggle** with GPU (T4/P100).
#
# Unlike the pilot (notebook 01, which inlined everything), this notebook drives
# the project's actual pipeline modules so the run matches the committed code:
#
# 1. `src.extraction.batch_extract` — extract activations for a whole concept
#    (all domains, aggregating `refusal/` + `refusal_new/`), cached to disk.
# 2. `src.analysis.run_pipeline`   — directions → angular dispersion →
#    cross-domain transfer → bootstrap CIs → figures + report.
#
# Run it once for **Qwen 2.5 3B Instruct** (primary) and once for **Qwen 2.5 3B
# Base** (the RLHF comparison axis), for both `refusal` and `honesty`.

# %% [markdown]
# ## 0. Setup & repo
#
# On Kaggle, make the project importable. Easiest: add this repo as a Kaggle
# *dataset/utility script*, or clone it, then `cd` in so `config.py` and `src/`
# are on the path.

# %%
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["transformer_lens", "transformers>=4.40.0", "bitsandbytes", "einops",
            "umap-learn", "matplotlib", "seaborn"]:
    install(pkg)

# %%
import os
from pathlib import Path

# Point this at the repo root (the folder containing config.py).
REPO_ROOT = Path(os.environ.get("BBNLP_ROOT", "/kaggle/working/blackboxnlp"))
assert (REPO_ROOT / "config.py").exists(), (
    f"config.py not found under {REPO_ROOT}. Clone the repo there or set BBNLP_ROOT."
)
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import torch
print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "Enable GPU in Kaggle: Settings -> Accelerator -> GPU."
print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 1. Configuration
#
# `quantization='4bit'` is already set for the Qwen configs in `config.py` to fit
# Kaggle memory. Backend `huggingface` is the safe default; try `transformer_lens`
# first if the model is supported.

# %%
from config import MODELS, CONCEPTS

MODEL_KEY = "qwen-2.5-3b-instruct"   # then re-run with "qwen-2.5-3b" for RLHF axis
CONCEPTS_TO_RUN = ["refusal", "honesty"]
BACKEND = "huggingface"              # or "transformer_lens"
# None = all layers. Subsample (e.g. list(range(0, 36, 2))) to save time/space.
LAYERS = None
BATCH_SIZE = 8

cfg = MODELS[MODEL_KEY]
print(f"Model: {cfg.name} | {cfg.n_layers} layers | d_model={cfg.d_model} | quant={cfg.quantization}")
for c in CONCEPTS_TO_RUN:
    print(f"  concept '{c}': domains = {CONCEPTS[c].domains}")

# %% [markdown]
# ## 2. Sanity check — what will be extracted (no model load)
#
# Confirms the committed dataset is discovered and aggregated as expected.

# %%
from src.extraction.batch_extract import discover_pairs_by_domain
from config import PATHS

for c in CONCEPTS_TO_RUN:
    by_domain = discover_pairs_by_domain(PATHS.prompt_pairs, concept=c)
    total = sum(len(v) for v in by_domain.values())
    print(f"\n{c}: {total} pairs")
    for d, pairs in sorted(by_domain.items()):
        print(f"  {d:<20} {len(pairs):>5}")

# %% [markdown]
# ## 3. Extract activations (per concept, all domains)
#
# Caches to `results/activations/` with the standard naming convention. This is
# the GPU-heavy step (~6h per checkpoint in the budget).

# %%
from src.extraction.batch_extract import run_batch

for c in CONCEPTS_TO_RUN:
    print(f"\n=== Extracting {MODEL_KEY} / {c} ===")
    run_batch(
        model_key=MODEL_KEY,
        concept=c,
        data_dir=PATHS.prompt_pairs,
        output_dir=PATHS.activations,
        backend=BACKEND,
        target_layers=LAYERS,
        batch_size=BATCH_SIZE,
    )

# %% [markdown]
# ## 4. Run the analysis pipeline (per concept)
#
# Computes per-domain + global directions, angular dispersion, cross-domain
# transfer, bootstrap CIs, and writes figures + a JSON/Markdown report.

# %%
from src.analysis.run_pipeline import run_pipeline

for c in CONCEPTS_TO_RUN:
    print(f"\n=== Analysing {MODEL_KEY} / {c} ===")
    report = run_pipeline(
        model_name=MODEL_KEY,
        concept=c,
        domains=CONCEPTS[c].domains,
        layers=list(range(cfg.n_layers)) if LAYERS is None else LAYERS,
        activations_dir=PATHS.activations,
        directions_dir=PATHS.directions,
        figures_dir=PATHS.figures,
        report_dir=PATHS.results,
        mock=False,
    )
    s = report["summary"]
    print(f"  min mean cos-to-global: {s['min_mean_cos']:.3f} @ layer {s['min_mean_cos_layer']}")
    print(f"  max mean angle-to-global: {s['max_mean_angle_deg']:.1f} deg @ layer {s['max_mean_angle_layer']}")

# %% [markdown]
# ## 5. RLHF comparison
#
# Re-run this whole notebook with `MODEL_KEY = "qwen-2.5-3b"` (Base). Then compare
# the Base vs. Instruct angular-dispersion profiles — hypothesis #3 predicts RLHF
# *increases* dispersion (directions become less universal).
#
# **Note on `refusal_new` responses:** they are synthetic (diversified) templates.
# For the most faithful refusal results, consider generating the model's *own*
# refusals/compliances from these prompts before extraction — see
# `docs/dataset_notes.md`.

# %% [markdown]
# ## Checklist
#
# - [ ] GPU verified, repo importable
# - [ ] Dataset discovered (refusal ~835, honesty ~885)
# - [ ] Instruct: activations extracted + analysis report/figures written
# - [ ] Base: same, for the RLHF comparison
# - [ ] Compare Base vs. Instruct dispersion
