# What's Left — Final Stretch Checklist

> Written: 2026-07-07 · Deadline: **July 17** (internal: July 14, figures: July 11)
> Companions: `results/latest_run_summary.md` (results + reviewer fixes),
> `docs/experimental_design.md` (validity), `docs/commands.md` (full reference).

## Where things stand (30-second version)

- ✅ All extraction done (Gemma pilot Jun 30; **Qwen both checkpoints × both
  concepts Jul 6** — the primary study data).
- ✅ Findings in: refusal largely universal (prompt-based), honesty fragments
  early-mid layers, RLHF does **not** amplify. Paper reframed accordingly
  (retitled Jul 7).
- ✅ All 9 reviewer-driven fixes applied Jul 7 (pair-level bootstrap, transfer
  layer, balance sensitivity, provenance, E4 driver, title, license, acl.sty).
  51 tests passing.
- ⏳ Left: **one Kaggle session** (notebook 03) + **writing**. Nothing else is
  blocked by code.

## ⚠️ Which folders are empty locally (and where the data actually is)

| Folder | Local state | Where to check / get it |
|---|---|---|
| `results/activations/` | **EMPTY** (gitignored, only `.gitkeep`) | The `.pt` files live in the **Kaggle session outputs** from Jul 6. Save them as a Kaggle dataset so notebook 03 can attach them. Without them, `run_pipeline --model qwen-*` fails locally. |
| `results/directions/` | **EMPTY** (gitignored) | Regenerated automatically by `run_pipeline` from activations — nothing to fetch. |
| `data/advbench/` | **EMPTY** | Raw AdvBench CSV was consumed during dataset build; only needed if rebuilding from scratch (`scripts/convert_advbench.py`). |
| `data/prompt_pairs/harmlessness/` | **EMPTY** | Stretch concept — dropped, intentionally empty. |
| `data/prompt_pairs_promptbased/` | populated but **regenerable/gitignored** | Rebuild anytime: `python scripts/build_refusal_promptbased.py --max-per-domain 120` |
| `results/` reports + `results/figures/` | ✅ present | Gemma reports at `results/report_refusal_gemma-2-2b*.md`, Qwen at `results/report_{concept}_qwen-2.5-3b*.md`, Qwen figures under `results/figures/qwen/`. |

**Rule of thumb:** reports/figures are committed; tensors are not. Anything
`.pt` must come from Kaggle.

## 1. Kaggle session (~8h GPU) — `notebooks/03_final_session_kaggle.py`

Everything below is already scripted in notebook 03; run it top-to-bottom.
Priority if quota runs short: **1 > 2 > 3**.

### 1a. Analysis rerun (CPU-only, needs activations attached)
Picks up the Jul 7 pipeline fixes (pair-level CIs, corrected transfer layer,
late-layer stats):
```bash
python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept honesty
python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept refusal
python -m src.analysis.run_pipeline --model qwen-2.5-3b           --concept honesty
python -m src.analysis.run_pipeline --model qwen-2.5-3b           --concept refusal
# T4 sensitivity (separate report dir):
python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept honesty \
    --balance-domains --no-figures --report-dir results/balanced
```

### 1b. E4 steering — THE deciding experiment (GPU ~6h)
```bash
# verify setup first (CPU):
python -m src.analysis.run_steering --model qwen-2.5-3b-instruct --concept refusal \
    --layer 18 --skip-first 120 --dry-run
# real runs (sweep coeff):
python -m src.analysis.run_steering --model qwen-2.5-3b-instruct --concept refusal \
    --layer 18 --coeff 4.0 --skip-first 120 --n-heldout 20 --cross-domain
```
Output: `results/steering_refusal_qwen-2.5-3b-instruct_layer018.json` —
`summary.own_minus_global` per domain is the paper's E4 number.

### 1c. E2′ robustness (GPU ~2h)
Response-based refusal on the same model (separate output dirs, no collision):
```bash
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept refusal \
    --max-pairs-per-domain 120 --output results/activations_respbased
python -m src.analysis.run_pipeline --model qwen-2.5-3b-instruct --concept refusal \
    --activations results/activations_respbased --directions results/directions_respbased \
    --figures results/figures/qwen_respbased --report-dir results/respbased
```
If this fragments while prompt-based doesn't → confound story is airtight.

### 1d. Download `results_final.zip` (notebook 03 packages it) and drop into `results/` locally.

## 2. Figures polish (~half day, after 1a)
- Transfer matrices regenerate at the corrected layer automatically.
- RLHF comparison figure: use `compare_base_vs_instruct()`
  (`src/analysis/angular_dispersion.py`) — needs a small plotting cell, the
  only figure not fully automated.
- PCA geometry (`geometry_pca_*`) = key figure; check colorblind-safety, 300 DPI.

## 3. Writing (~4–6 days) — the actual critical path now
- [ ] Verify `paper/main.tex` compiles — `acl.sty` + `acl_natbib.bst` are in
  `paper/` now, but **pdflatex is not installed locally**; use Overleaf or
  install TeX Live. Uncomment `\usepackage[review]{acl}` when switching to the
  official style.
- [ ] Abstract + Intro around the reframed story (title already updated).
- [ ] Methods: from `docs/experimental_design.md` + pipeline params
  (`config.py`: resid_post, last token, diff-of-means, 4-bit).
- [ ] Results: numbers are in the report JSONs; `results/latest_run_summary.md`
  has the narrative skeleton.
- [ ] Discussion: what the controlled null means for RepE; confound lesson.
- [ ] Limitations: 4-bit quant (T5), single family (T6), keyword refusal
  detection in E4, math domain n=59.
- [ ] Review sprint July 12–14; submit on OpenReview by July 17.

## 4. Small admin (~1h)
- [ ] Confirm the Jul 6 Kaggle notebook used `data/prompt_pairs_promptbased`
  for refusal (provenance sidecars only exist for NEW extractions).
- [ ] Sign off `docs/operational_definitions.md` (both members).
- [ ] Decide pooled-vs-split framing reporting for the E2′ condition
  (recommend: split + report agreement).
- [x] MedSafetyBench license — confirmed MIT (research-only), citation in
  `docs/dataset_notes.md`.
- [x] GitHub remote — exists (`ArnavLifelessCoder/blackboxnlpAA`); keep pushing.
