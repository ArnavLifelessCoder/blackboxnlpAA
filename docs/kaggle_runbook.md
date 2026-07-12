# Kaggle Runbook — Reviewer Cheap-Wins Session

> Self-contained instructions. Anyone with a Kaggle account can run this.
> Goal: produce `results_cheapwins.zip` with 4 analyses that harden the paper.
> Priority if time/quota runs out: **Cell 4 > Cell 5 > Cell 6 > Cell 7.**
> Cells 4–5 are CPU-only (work even without GPU quota).

## A. Notebook setup

1. kaggle.com → sign in → **Create → New Notebook**.
2. **Settings → Accelerator → GPU T4 x2** (skip if only running cells 4–5).
3. **Settings → Internet → On** (requires phone-verified account).
4. Delete the default boilerplate cell.

## B. Attach the activations dataset

The analyses read cached activation tensors (`.pt` files) produced by the
July extraction runs. They are NOT in the GitHub repo.

- If a dataset with the `.pt` files exists (profile → **Your Work →
  Datasets**): in the notebook, right panel → **+ Add Input** → filter
  **Your Work** → add it.
- If not: open the old extraction notebook → **Output** tab → if the `.pt`
  files are listed there, click **New Dataset**, name it
  `blackbox-activations`, then Add Input as above.
- If no `.pt` files exist anywhere: activations must be re-extracted first
  (see notebook `02_full_extraction_qwen3b.py`, ~6h GPU) — do that before
  this runbook.

## C. Cells — paste and run ONE AT A TIME, in order

### Cell 1 — clone + install (~2 min)
```python
import subprocess, sys, os, glob, shutil
subprocess.check_call(["git", "clone",
    "https://github.com/ArnavLifelessCoder/blackboxnlpAA.git",
    "/kaggle/working/blackboxnlpAA"])
os.chdir("/kaggle/working/blackboxnlpAA")
sys.path.insert(0, "/kaggle/working/blackboxnlpAA")
for pkg in ["scikit-learn", "transformers>=4.40.0", "bitsandbytes", "einops"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print(os.listdir("/kaggle/input"))   # note the dataset folder name for Cell 2
```

### Cell 2 — copy activations in (EDIT the dataset name first)
```python
DATASET = "YOUR-DATASET-NAME"   # <-- folder name printed by Cell 1
for f in glob.glob(f"/kaggle/input/{DATASET}/**/*.pt", recursive=True):
    shutil.copy(f, "results/activations/")
n = len(glob.glob("results/activations/*.pt"))
print("activation files:", n)
assert n > 0, "wrong dataset name/path - check the Input panel"
```

### Cell 3 — rebuild gitignored data (~1 min, needed by Cell 7)
```python
!python scripts/build_refusal_promptbased.py --max-per-domain 120
```

### Cell 4 — probe transfer matrices [CPU, ~10 min] — TOP PRIORITY
```python
!python -m src.analysis.probe_transfer --model qwen-2.5-3b-instruct --concept honesty --layers 5 18 30
!python -m src.analysis.probe_transfer --model qwen-2.5-3b --concept honesty --layers 5 18 30
!python -m src.analysis.probe_transfer --model qwen-2.5-3b-instruct --concept honesty --layers 5 30 --balance-to 59 --report-dir results/probe_balanced
!python -m src.analysis.probe_transfer --model qwen-2.5-3b-instruct --concept refusal --layers 5 18 30
```

### Cell 5 — balanced (n=59) honesty rerun [CPU, ~20 min]
```python
!python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept honesty --balance-domains --no-figures --report-dir results/balanced
!python -m src.analysis.run_pipeline --model qwen-2.5-3b --concept honesty --balance-domains --no-figures --report-dir results/balanced
```

### Cell 6 — fp16 quantization spot-check [GPU, ~1h]
```python
!python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept honesty --backend huggingface --max-pairs-per-domain 120 --layers 5 18 --no-quantization --output results/activations_fp16
!python scripts/compare_quantization.py --model qwen-2.5-3b-instruct --concept honesty --layers 5 18 --acts-fp16 results/activations_fp16
```

### Cell 7 — steering rerun with saved generations + LLM judge [GPU, ~4h]
```python
!python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept refusal --no-figures
!python -m src.analysis.run_steering --model qwen-2.5-3b-instruct --concept refusal --layer 18 --coeff 4.0 --skip-first 120 --n-heldout 20 --cross-domain
!python scripts/judge_steering.py --generations results/steering_refusal_qwen-2.5-3b-instruct_layer018_coeff4_generations.jsonl
```

### Cell 8 — package
```python
shutil.make_archive("/kaggle/working/results_cheapwins", "zip", root_dir="results")
print("download results_cheapwins.zip from the Output tab")
```

## D. Before closing the session — VERIFY (30 seconds, do not skip)

Open the zip listing (or the Output tab) and confirm these exist:

- `probe_transfer_honesty_qwen-2.5-3b-instruct.json` (+ Base, + refusal, + `.md`)
- `probe_balanced/probe_transfer_honesty_*.json`
- `balanced/report_honesty_*.json`
- `quant_check_honesty_qwen-2.5-3b-instruct.json`
- `steering_refusal_*_coeff4.json`
- `steering_refusal_*_coeff4_generations.jsonl`  ← must be non-trivial size (MBs)
- `steering_refusal_*_coeff4_judged.json`

Previous sessions lost outputs twice by skipping this check.

## E. Sanity signals while running

- Cell 4 prints per-layer lines ending in a summary like
  `{'mean_within_acc': ..., 'mean_cross_acc': ..., 'transfer_gap': ...}` —
  if `mean_within_acc` is near 0.5, something is wrong (probes should
  separate pos/neg within-domain easily).
- Cell 7's steering step must print `E4 setup OK: 4 domains x 20 held-out
  prompts` before loading the model. If it errors about missing prompts,
  Cell 3 was skipped.
- If any cell errors, stop and report the error - do not improvise.

Deliver `results_cheapwins.zip` back into the project root.
