# Latest Run Summary & Next Steps

> Snapshot updated: 2026-07-09. **Freshest numbers are in `results_final/`**
> (Jul 8 rerun with the fixed pipeline: pair-level CIs, corrected transfer
> layers, late-layer stats). Session outcome table: `docs/whats_left.md`.
> ⚠️ The Jul 8 E4 steering JSON in `results_final/` is INVALID (n=0 — see
> `docs/whats_left.md`); the driver is fixed, rerun on next GPU session.
> Companion to `next_steps.md`, `docs/remaining_work.md`, `docs/experimental_design.md`.

## What has run

| Run | Model | Concept | Data | Date | Status |
|---|---|---|---|---|---|
| Pilot | Gemma 2 2B Base + Instruct | refusal | response-based (`refusal_new/`, confounded) | Jun 30 | ✅ real activations |
| **Primary** | **Qwen 2.5 3B Base + Instruct** | **honesty** | TruthfulQA (single-source, clean) | Jul 6 | ✅ real activations |
| **Primary** | **Qwen 2.5 3B Base + Instruct** | **refusal** | prompt-based (confound-controlled)* | Jul 6 | ✅ real activations |

All runs: all layers, 1000 bootstrap resamples, `mock: False`. Figures:
`results/figures/` (Gemma) and `results/figures/qwen/` (Qwen).
\* Provenance not recorded in the report JSON — **confirm** the Kaggle run pointed
at `data/prompt_pairs_promptbased` (high cosines are consistent with it).

## Qwen results (the ones the paper claims rest on)

### E1 — Honesty (clean primary experiment)
- Fragmentation is real but **layer-dependent**: mean cosine-to-global bottoms at
  **0.522 @ layer 5** (~57°), then recovers to **0.69–0.77** in late layers.
- Cross-domain transfer at L5 is poor and uneven: **`math` is near-orthogonal to
  every other domain** (−0.10…−0.02); best pair `factual_trivia ↔ personal_advice`
  only 0.34.

### E2 — Refusal (prompt-based, confound-controlled)
- **Largely universal**: mean cosine **0.83–0.94 across all 36 layers**
  (argmin lands on layer 0 = embeddings, an artifact — deeper layers sit at 0.90+).
- **Contrast with the Gemma pilot** (min cos 0.56 on response-based data): once the
  source/template confounds (T1/T2) are removed, refusal "fragmentation" mostly
  disappears. The pilot fragmentation was plausibly the confound, not the concept.

### E3 — RLHF axis: **hypothesis NOT supported**
- Instruct is slightly *more* universal than Base, not less: mean Δcos
  **+0.011 (honesty)**, **+0.007 (refusal)**; gap widens in late layers
  (honesty L35: 0.690 → 0.769). Only 10/36 layers go the other way, weakly.
- Sanity check passed: Base–Instruct differences grow with depth (max 0.20),
  the expected signature of a genuine fine-tune — not a caching bug.

### Headline (revised framing)
> Prompt-based refusal directions largely **reproduce** RepE's universality
> assumption once dataset confounds are controlled; **honesty fragments
> substantially in early-mid layers** (math near-orthogonal); **RLHF does not
> amplify fragmentation** — if anything it slightly consolidates directions; and
> **naive dataset construction manufactures false fragmentation** (Gemma pilot
> vs. Qwen prompt-based contrast).

This is the pre-planned null/reproducibility framing (`next_steps.md` risk table).
H1 partially supported (honesty, early layers); H3 answered "no".

## Reviewer-driven fixes (applied 2026-07-07)

1. ✅ **Transfer-layer artifact** — embedding-adjacent layers (0–1) excluded from
   the argmin; `--transfer-layer` override added (`run_pipeline.py`). Re-run E
   to re-render the refusal transfer matrices.
2. ✅ **Pair-level bootstrap** — `bootstrap_pairlevel_dispersion_ci` resamples
   *prompt pairs* and recomputes directions per resample; `run_pipeline` now
   uses it whenever per-pair activations are loadable (falls back to the old
   domain-level CI with a `method` tag otherwise). Re-run E for headline CIs.
3. ✅ **Provenance** — extraction writes a `meta_*.json` sidecar (data dir,
   schemas, framings, sources, n_pairs); `run_pipeline` folds it into the
   report. Takes effect on the next extraction — for the existing Jul 6 runs,
   still manually confirm the Kaggle notebook pointed at
   `data/prompt_pairs_promptbased`.
4. ✅ **Domain-imbalance sensitivity** — `--balance-domains` subsamples every
   domain to the smallest domain's n (seeded, consistent across layers and
   bootstrap). Run E once with and once without to report sensitivity.
5. ✅ **Early-vs-late layers** — report now includes `late_layer_mean_cos`
   (final third of the stack) + an interpretation note that early-layer minima
   may be lexical. The behavioral check is still E4.
6. ✅ **Title mismatch** — paper + README retitled to *"How Universal Are
   Concept Directions? A Controlled Stress-Test of Representation Engineering
   Under RLHF"*; README hypotheses annotated with outcomes.
7. ✅ **E4 driver ready** — `src/analysis/run_steering.py`: held-out prompts
   (skips the pairs used for directions), baseline vs global vs own-domain
   (vs cross-domain with `--cross-domain`), refusal-rate deltas, JSON report.
   `--dry-run` verifies the setup on CPU.
8. ✅ **MedSafetyBench license confirmed** — MIT, research-use-only condition;
   citation recorded in `docs/dataset_notes.md`.
9. ✅ **acl.sty + acl_natbib.bst downloaded** into `paper/` (repo-root path; the
   old `latex/` URL 404s — `docs/commands.md` corrected). ⚠️ `pdflatex` is not
   installed on this machine — compile still unverified.

Tests: 51 passing (13 new in `tests/test_review_fixes.py`).

## Next steps (revised, ~10 days to deadline)

1. **Steering experiment (E4)** — the deciding evidence; spend remaining GPU
   (~6h) here. On Kaggle:
   `python -m src.analysis.run_steering --model qwen-2.5-3b-instruct --concept refusal --layer 18 --coeff 4.0 --skip-first 120`
   (sweep `--coeff` over 0.5–5; `--dry-run` first).
2. **E2′ robustness (cheap, high value)** — run response-based `refusal_new/` on
   Qwen too: fragmentation there + universality prompt-based *on the same model*
   makes the confound story airtight.
3. **Re-run analysis (step E)** with the fixed pipeline (needs the cached
   activations from Kaggle): corrected transfer layer, pair-level CIs,
   `--balance-domains` sensitivity, late-layer stats.
4. **Figures** — transfer matrices at the corrected layer, RLHF comparison
   (`compare_base_vs_instruct`), PCA/UMAP key figure.
5. **Write** — Methods/Results from the reports, Discussion of the null,
   Limitations (4-bit quant T5, single family T6, keyword-based refusal
   detection in E4). Install TeX or use Overleaf to verify compile.
6. **Admin** — GitHub remote; operational-definitions sign-offs; framing
   decision (pool vs split) for the response-based robustness condition.

## Commands to run

Labels: **[CPU]** laptop · **[GPU]** Kaggle session · **[LOCAL]** git/latex.
All from project root. Verified against script argparse on 2026-07-06.

### A. Verify code & data — [CPU]
```powershell
python -m pytest tests/ -q                                    # smoke tests
python scripts/validate_prompt_pairs.py                       # dataset health
python scripts/validate_prompt_pairs.py --strict              # CI gate (fail on warning)
```

### B. See the pipeline end-to-end on synthetic data — [CPU] (no GPU/model)
```powershell
python -m src.analysis.run_pipeline --mock --concept honesty
python -m src.analysis.run_pipeline --mock --concept refusal
```

### C. Build design-faithful refusal data + preview extraction — [CPU]
```powershell
python scripts/build_refusal_promptbased.py --max-per-domain 120
python scripts/validate_prompt_pairs.py --data-dir data/prompt_pairs_promptbased
# Preview what the GPU run would do (no model loaded):
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept refusal `
    --data-dir data/prompt_pairs_promptbased --max-pairs-per-domain 120 --dry-run
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept honesty `
    --max-pairs-per-domain 120 --dry-run
```

### D. Real extraction — [GPU / KAGGLE] ✅ done Jul 6 for Qwen (both checkpoints)
Notebook `notebooks/02_full_extraction_qwen3b.py`. Equivalent CLI:
```bash
python scripts/build_refusal_promptbased.py --max-per-domain 120
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept refusal \
    --data-dir data/prompt_pairs_promptbased --max-pairs-per-domain 120
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept honesty \
    --max-pairs-per-domain 120
# Base (RLHF axis): repeat both with --model qwen-2.5-3b
```
**Still to run on GPU:** steering (E4) and optionally E2′ (response-based refusal
on Qwen: same commands without `--data-dir`, i.e. default `data/prompt_pairs`).

### E. Real analysis after extraction — [CPU] ✅ done Jul 6; re-run after transfer-layer fix
```powershell
python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept honesty
python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept refusal
python -m src.analysis.run_pipeline --model qwen-2.5-3b           --concept honesty
python -m src.analysis.run_pipeline --model qwen-2.5-3b           --concept refusal
# Pin the transfer matrix to a meaningful layer explicitly if desired:
#   ... --transfer-layer 18
# Outputs: results/report_<concept>_<model>.{json,md} + results/figures/
```
(Requires the cached activation `.pt` files locally — they are gitignored, so
re-runs need the Kaggle outputs downloaded into `results/activations/`.)

### F. Paper & git — [LOCAL]
```bash
curl -L -o paper/acl.sty https://raw.githubusercontent.com/acl-org/acl-style-files/master/latex/acl.sty
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
gh repo create blackboxnlp-2026 --private --source=. --remote=origin --push
```

> Note: PowerShell line continuation is a backtick (`` ` ``); the `bash` blocks use `\`.
> `build_refusal_promptbased.py` uses `--check` (not `--dry-run`) for preview.

## Key dates

| Date | Milestone |
|---|---|
| July 11 | Figures delivered |
| July 14 | Internal deadline |
| **July 17** | **OpenReview deadline** |
| July 27 | Reproducibility Challenge |
