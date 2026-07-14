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
# # 04 — Reviewer Cheap Wins (pre-submission hardening)
#
# Four analyses answering the mock-review's strongest objections, in priority
# order. Items 1–2 are **CPU-only** (no GPU quota needed — run them even in a
# CPU session); items 3–4 need GPU.
#
# | # | Item | Answers | Needs |
# |---|---|---|---|
# | 1 | Probe transfer matrix (honesty) | "fragmentation has no functional leg" | CPU + activations |
# | 2 | n=59 balanced honesty rerun | "math result rests on the least data" | CPU + activations |
# | 3 | fp16 spot-check (layers 5, 18) | "4-bit may distort the geometry you measure" | GPU ~1h |
# | 4 | Steering rerun w/ saved generations + LLM judge | "keyword refusal detection" | GPU ~4h |
#
# NOTE on item 1: the probe result cuts either way. High cross-domain probe
# accuracy = cosine fragmentation is superficial (lexical); low = functional
# fragmentation. The paper text must follow whichever comes out.

# %% [markdown]
# ## 0. Setup — clone repo, attach activations, rebuild gitignored data

# %%
import subprocess, sys, os
from pathlib import Path

REPO = Path("/kaggle/working/concept-directions")
if not REPO.exists():
    subprocess.check_call(["git", "clone",
        "<ANONYMIZED_REPO_URL>", str(REPO)])
os.chdir(REPO)
sys.path.insert(0, str(REPO))

def run(cmd):
    print(">>>", " ".join(cmd)); subprocess.check_call(cmd)

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["scikit-learn", "transformers>=4.40.0", "bitsandbytes", "einops",
            "matplotlib", "seaborn"]:
    install(pkg)

# Attach the Jul 6 activations dataset, then copy the .pt files in:
# !cp /kaggle/input/<your-activations-dataset>/*.pt results/activations/
n_act = len(list(Path("results/activations").glob("*.pt")))
print(f"activation files present: {n_act}")
assert n_act > 0, "Attach the activations dataset first."

# Gitignored on fresh clones; needed by item 4:
run([sys.executable, "scripts/build_refusal_promptbased.py", "--max-per-domain", "120"])

PY = [sys.executable, "-m"]

# %% [markdown]
# ## 1. Probe transfer matrix — honesty (CPU, ~10 min)
# Layers: 5 (fragmentation peak), 18 (mid), 30 (late third).
# Both checkpoints; plus an n=59 balanced variant of the key layer.

# %%
for model in ["qwen-2.5-3b-instruct", "qwen-2.5-3b"]:
    run(PY + ["src.analysis.probe_transfer", "--model", model,
              "--concept", "honesty", "--layers", "5", "18", "30"])

# Balanced variant (every domain subsampled to math's n):
run(PY + ["src.analysis.probe_transfer", "--model", "qwen-2.5-3b-instruct",
          "--concept", "honesty", "--layers", "5", "30",
          "--balance-to", "59",
          "--report-dir", "results/probe_balanced"])

# Refusal too — cheap, and symmetry helps the paper:
run(PY + ["src.analysis.probe_transfer", "--model", "qwen-2.5-3b-instruct",
          "--concept", "refusal", "--layers", "5", "18", "30"])

# %% [markdown]
# ## 2. n=59 balanced honesty dispersion rerun (CPU, ~20 min)

# %%
for model in ["qwen-2.5-3b-instruct", "qwen-2.5-3b"]:
    run(PY + ["src.analysis.run_pipeline", "--model", model,
              "--concept", "honesty", "--balance-domains", "--no-figures",
              "--report-dir", "results/balanced"])

# %% [markdown]
# ## 3. fp16 spot-check (GPU ~1h) — layers 5 and 18, honesty
# Extracts unquantized fp16 activations into a separate cache, then compares
# direction geometry against the 4-bit cache.

# %%
run(PY + ["src.extraction.batch_extract", "--model", "qwen-2.5-3b-instruct",
          "--concept", "honesty", "--backend", "huggingface",
          "--max-pairs-per-domain", "120", "--layers", "5", "18",
          "--no-quantization", "--output", "results/activations_fp16"])

run([sys.executable, "scripts/compare_quantization.py",
     "--model", "qwen-2.5-3b-instruct", "--concept", "honesty",
     "--layers", "5", "18", "--acts-fp16", "results/activations_fp16"])

# %% [markdown]
# ## 4. Steering rerun with saved generations + LLM judge (GPU ~4h)
# The patched driver writes a *_generations.jsonl sidecar; the judge script
# re-scores it. Directions must exist in results/directions (produced by any
# run_pipeline pass over the refusal activations).

# %%
# Ensure directions exist for the steering layer:
run(PY + ["src.analysis.run_pipeline", "--model", "qwen-2.5-3b-instruct",
          "--concept", "refusal", "--no-figures"])

run(PY + ["src.analysis.run_steering", "--model", "qwen-2.5-3b-instruct",
          "--concept", "refusal", "--layer", "18", "--coeff", "4.0",
          "--skip-first", "120", "--n-heldout", "20", "--cross-domain"])

run([sys.executable, "scripts/judge_steering.py", "--generations",
     "results/steering_refusal_qwen-2.5-3b-instruct_layer018_coeff4_generations.jsonl"])

# %% [markdown]
# ## 5. Package
# Grab results_cheapwins.zip from the Output tab. Verify it contains:
# probe_transfer_*.json/md, results/balanced/, quant_check_*.json,
# steering_*_coeff4.json + _generations.jsonl + _judged.json.

# %%
import shutil
shutil.make_archive("/kaggle/working/results_cheapwins", "zip", root_dir="results")
print("Wrote /kaggle/working/results_cheapwins.zip")
