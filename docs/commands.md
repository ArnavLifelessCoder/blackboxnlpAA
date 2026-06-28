# Command Reference

All commands run from the project root. Labels:
**[CPU]** runs on your laptop · **[GPU/KAGGLE]** needs a GPU session ·
**[LOCAL]** git/latex on your machine.

---

## Setup — [LOCAL, CPU] (run once)

```powershell
cd "C:\Users\Arnav Gawade(pro)\Downloads\blackboxnlpAA"
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

## Preview extraction (what would run on GPU) — [CPU]

```powershell
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept refusal --dry-run
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept honesty --dry-run
```

## Real extraction — [GPU / KAGGLE] (~6 hrs per checkpoint)

```bash
# Kaggle notebook with GPU enabled, after cloning the repo:
!git clone https://github.com/<you>/blackboxnlp-2026 /kaggle/working/blackboxnlp

# Run notebooks/02_full_extraction_qwen3b.py, or these directly:
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept refusal
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept honesty
python -m src.extraction.batch_extract --model qwen-2.5-3b           --concept refusal   # Base (RLHF axis)
python -m src.extraction.batch_extract --model qwen-2.5-3b           --concept honesty
```

## Real analysis (after extraction) — [CPU] (runs locally once activations exist)

```powershell
python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept refusal
python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept honesty
python -m src.analysis.run_pipeline --model qwen-2.5-3b           --concept refusal
python -m src.analysis.run_pipeline --model qwen-2.5-3b           --concept honesty
```

## Git — [LOCAL]

```powershell
git status
gh repo create blackboxnlp-2026 --private --source=. --remote=origin --push
```

## Paper — [LOCAL]

```bash
curl -L -o paper/acl.sty https://raw.githubusercontent.com/acl-org/acl-style-files/master/latex/acl.sty
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

**Minimum path to "see it working" now:** Setup → Verify → Run pipeline (`--mock`).
Only commands with a real model (`--model qwen-*` without `--dry-run`) need a GPU.
