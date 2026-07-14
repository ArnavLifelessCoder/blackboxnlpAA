# Command Reference

All commands run from the project root. Labels:
**[CPU]** runs on your laptop · **[GPU/KAGGLE]** needs a GPU session ·
**[LOCAL]** git/latex on your machine.

> The GPU sections below are kept for reproduction. Fresh runs write to
> `results/` (auto-created).

---

## Setup — [LOCAL, CPU] (run once)

```powershell
cd <project-root>
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
```

## Verify code & data — [CPU]

```powershell
python -m pytest tests/ -q                          # 38 tests
python scripts/validate_prompt_pairs.py             # dataset health (counts/diversity)
python scripts/validate_prompt_pairs.py --strict    # fails on any warning (CI gate)
```

## See the pipeline work end-to-end — [CPU] (synthetic data, no model)

```powershell
python -m src.analysis.run_pipeline --mock --concept refusal
python -m src.analysis.run_pipeline --mock --concept honesty
ls results/figures/
cat results/report_refusal_gemma-2-2b-it.md
```

## Rebuild / inspect the dataset — [CPU] (`--check` = preview only)

```powershell
python scripts/convert_truthfulqa.py --max-pairs 1000   # downloads TruthfulQA
python scripts/convert_advbench.py                      # needs AdvBench CSV
python scripts/merge_truthfulqa_into_honesty.py --check
python scripts/migrate_seed_to_shared_schema.py --check
python scripts/add_math_honesty_pairs.py --check
python scripts/add_refusal_pairs.py --check
python scripts/finalize_refusal_dataset.py --check
```

## Build the prompt-based refusal set — [CPU] (design-faithful refusal data)

Canonical method: harmful vs. harmless *prompts*, no synthetic responses,
balanced per domain (see `docs/experimental_design.md`). Output is regenerable
(gitignored).

```powershell
python scripts/build_refusal_promptbased.py --max-per-domain 100
python scripts/validate_prompt_pairs.py --data-dir data/prompt_pairs_promptbased
```

## Preview extraction (what would run on GPU) — [CPU]

```powershell
# Response-based (legacy) discovery:
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept refusal --dry-run
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept honesty --dry-run

# Design-faithful: prompt-based refusal + balanced caps:
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept refusal \
    --data-dir data/prompt_pairs_promptbased --max-pairs-per-domain 120 --dry-run
```

## Real extraction — [GPU / KAGGLE] (~6 hrs per checkpoint)

The notebook `notebooks/02_full_extraction_qwen3b.py` already does the
design-faithful run (prompt-based refusal + balanced caps). Equivalent CLI:

```bash
# Kaggle notebook with GPU enabled, after cloning the repo:
!git clone <ANONYMIZED_REPO_URL> /kaggle/working/concept-directions

# Refusal — prompt-based, balanced (primary refusal condition)
python scripts/build_refusal_promptbased.py --max-per-domain 120
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept refusal \
    --data-dir data/prompt_pairs_promptbased --max-pairs-per-domain 120

# Honesty — balanced (primary, clean single-source experiment)
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept honesty \
    --max-pairs-per-domain 120

# Repeat both with --model qwen-2.5-3b for the Base (RLHF) axis.
```

## Real analysis (after extraction) — [CPU] (runs locally once activations exist)

⚠️ `results/activations/` is gitignored and empty locally — the `.pt` files live
in the Kaggle session outputs. Download them first or run these on Kaggle.

```powershell
python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept refusal
python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept honesty
python -m src.analysis.run_pipeline --model qwen-2.5-3b           --concept refusal
python -m src.analysis.run_pipeline --model qwen-2.5-3b           --concept honesty
# Options added 2026-07-07:
#   --transfer-layer N     pin the transfer matrix to a specific layer
#   --balance-domains      subsample all domains to the smallest domain's n (T4)
```

## Steering (E4, behavioral test) — [GPU / KAGGLE] (~6 hrs)

```bash
# CPU dry-run first (verifies directions + held-out split, no model load):
python -m src.analysis.run_steering --model qwen-2.5-3b-instruct --concept refusal \
    --layer 18 --skip-first 120 --dry-run

# Real run: baseline vs global vs own-domain (vs cross-domain) on held-out prompts
python -m src.analysis.run_steering --model qwen-2.5-3b-instruct --concept refusal \
    --layer 18 --coeff 4.0 --skip-first 120 --n-heldout 20 --cross-domain
```

Output filename embeds the coefficient (`steering_*_layer018_coeff4.json`) so
sweep runs don't overwrite each other. **Prompt-based data must be rebuilt on a
fresh clone first** (gitignored): `python scripts/build_refusal_promptbased.py --max-per-domain 120`.

The full final GPU session (analysis rerun + E4 + E2′) is scripted in
`notebooks/03_final_session_kaggle.py`.

## Git — [LOCAL]

```powershell
git status
gh repo create <repo-name> --private --source=. --remote=origin --push
```

## Paper — [LOCAL]

```bash
# acl.sty lives at the repo ROOT (the old latex/ path 404s); grab the .bst too
curl -L -o paper/acl.sty https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty
curl -L -o paper/acl_natbib.bst https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl_natbib.bst
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

**Minimum path to "see it working" now:** Setup → Verify → Run pipeline (`--mock`).
Only commands with a real model (`--model qwen-*` without `--dry-run`) need a GPU.
