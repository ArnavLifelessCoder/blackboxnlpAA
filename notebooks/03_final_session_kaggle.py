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
# # 03 — Final GPU Session: Analysis Rerun + Steering (E4) + E2′
#
# **BlackboxNLP 2026** — the last Kaggle session before writing (~8h budget).
#
# Three jobs, in order of importance:
#
# 1. **Analysis rerun** (CPU-only, needs the cached activations from the Jul 6
#    session) — picks up the 2026-07-07 pipeline fixes: pair-level bootstrap
#    CIs, corrected transfer-layer selection, late-layer stats, provenance,
#    and the `--balance-domains` sensitivity run.
# 2. **E4 steering** (GPU, ~6h) — the behavioral test (Hypothesis 2 / threat
#    T7). Global vs per-domain direction on held-out prompts.
# 3. **E2′ robustness** (GPU, ~2h) — response-based `refusal_new/` on Qwen
#    Instruct. Fragmentation here + universality prompt-based on the SAME
#    model makes the confound story airtight.
#
# If quota runs short: E4 refusal-only at one coeff > E2′ > extra coeffs.

# %% [markdown]
# ## 0. Setup
#
# Attach the Jul 6 activations as a Kaggle dataset (or rerun notebook 02 first).
# `results/activations/` must contain the `pos_/neg_ qwen_25_3b*` .pt files.

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
REPO = Path("/kaggle/working/blackboxnlpAA")
if not REPO.exists():
    subprocess.check_call(
        ["git", "clone", "https://github.com/ArnavLifelessCoder/blackboxnlpAA.git",
         str(REPO)]
    )
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# If activations were attached as a Kaggle dataset, link/copy them in:
# !cp /kaggle/input/<your-activations-dataset>/*.pt results/activations/

import torch
print("CUDA:", torch.cuda.is_available())
n_act = len(list(Path("results/activations").glob("*.pt")))
print(f"activation files present: {n_act}")
assert n_act > 0, "No cached activations — attach the Jul 6 outputs or rerun notebook 02."

# data/prompt_pairs_promptbased/ is GITIGNORED — a fresh clone does not have
# it, and steering needs it for held-out prompts. Rebuild it (deterministic):
subprocess.check_call([sys.executable, "scripts/build_refusal_promptbased.py",
                       "--max-per-domain", "120"])
n_pb = len(list(Path("data/prompt_pairs_promptbased/refusal").glob("*.jsonl")))
assert n_pb == 4, f"prompt-based rebuild produced {n_pb}/4 domain files"

# %% [markdown]
# ## 1. Analysis rerun (CPU) — fixed pipeline, all 4 reports

# %%
def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)

PY = [sys.executable, "-m"]

for model in ["qwen-2.5-3b-instruct", "qwen-2.5-3b"]:
    for concept in ["honesty", "refusal"]:
        run(PY + ["src.analysis.run_pipeline",
                  "--model", model, "--concept", concept,
                  "--figures", "results/figures/qwen"])

# %% [markdown]
# ### 1b. Domain-balance sensitivity (T4) — reports to a separate dir

# %%
for model in ["qwen-2.5-3b-instruct", "qwen-2.5-3b"]:
    for concept in ["honesty", "refusal"]:
        run(PY + ["src.analysis.run_pipeline",
                  "--model", model, "--concept", concept,
                  "--balance-domains", "--no-figures",
                  "--report-dir", "results/balanced"])

# %% [markdown]
# ## 2. E4 — Steering (GPU, the deciding experiment)
#
# Pick the steering layer from the fresh reports: a mid/late layer where the
# concept is represented (for refusal, dispersion is flat-high, so a late-mid
# layer like 18–24 of 36; for the paper also confirm with one alternative).
# Positive coeff pushes toward compliance (see run_steering docstring).

# %%
# Dry-run first: verifies directions + held-out split, no model load.
run(PY + ["src.analysis.run_steering",
          "--model", "qwen-2.5-3b-instruct", "--concept", "refusal",
          "--layer", "18", "--skip-first", "120", "--dry-run"])

# %%
# Real run + a small coefficient sweep. ~20 held-out prompts x 4 domains x
# (baseline + global + own) generations per coeff.
for coeff in ["2.0", "4.0"]:
    run(PY + ["src.analysis.run_steering",
              "--model", "qwen-2.5-3b-instruct", "--concept", "refusal",
              "--layer", "18", "--coeff", coeff,
              "--skip-first", "120", "--n-heldout", "20",
              "--cross-domain"])

# Optional (budget permitting): repeat on the Base checkpoint for the RLHF axis.
# run(PY + ["src.analysis.run_steering", "--model", "qwen-2.5-3b",
#           "--concept", "refusal", "--layer", "18", "--coeff", "4.0",
#           "--skip-first", "120"])

# %% [markdown]
# ## 3. E2′ — Response-based refusal on Qwen (robustness, ~2h)
#
# Same model, same pipeline, but the response-based dataset (default
# `data/prompt_pairs`, which aggregates `refusal/` + `refusal_new/`).
# Output goes to a separate activations dir so it can't collide with the
# prompt-based cache.

# %%
run(PY + ["src.extraction.batch_extract",
          "--model", "qwen-2.5-3b-instruct", "--concept", "refusal",
          "--max-pairs-per-domain", "120",
          "--output", "results/activations_respbased"])

run(PY + ["src.analysis.run_pipeline",
          "--model", "qwen-2.5-3b-instruct", "--concept", "refusal",
          "--activations", "results/activations_respbased",
          "--directions", "results/directions_respbased",
          "--figures", "results/figures/qwen_respbased",
          "--report-dir", "results/respbased"])

# %% [markdown]
# ## 4. Package outputs
#
# Download everything under `results/` EXCEPT the big `.pt` activation files
# (those stay as a Kaggle dataset for future reruns):
# reports (`results/*.json|md`, `results/balanced/`, `results/respbased/`),
# `results/steering_*.json`, and `results/figures/`.

# %%
import shutil
shutil.make_archive("/kaggle/working/results_final", "zip",
                    root_dir="results",
                    base_dir=".")
print("Wrote /kaggle/working/results_final.zip — download from the Output tab.")
