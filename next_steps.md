# Next Steps — BlackboxNLP 2026

> Last updated: June 16, 2026  
> Paper deadline: **July 17, 2026** (internal target: July 14)

---

## ✅ Done (Phase 1 — Infrastructure)

- [x] Project structure & directory scaffolding
- [x] `config.py` — models, concepts, domains, hyperparameters
- [x] Extraction pipeline — `hooks.py`, `extract_activations.py`, `cache_utils.py`
- [x] Analysis modules — `directions.py`, `angular_dispersion.py`, `steering.py`, `bootstrap.py`
- [x] Visualization modules — `plot_utils.py`, `heatmaps.py`, `transfer_matrix.py`, `pca_umap.py`
- [x] ~~10~~ → 80 seed prompt pairs (40 refusal + 40 honesty, 10 per domain)
- [x] 26 smoke tests — all passing
- [x] README, requirements.txt, .gitignore
- [x] Git repo initialized — commit `725ded0` (45 files, 5222 insertions)
- [x] Operational definitions for refusal & honesty (`docs/operational_definitions.md`)
- [x] ACL LaTeX template (`paper/main.tex`, `paper/references.bib`)
- [x] Dataset adapter scripts (`scripts/convert_truthfulqa.py`, `scripts/convert_advbench.py`)
- [x] Kaggle notebook scaffold (`notebooks/01_pilot_gemma2b.py`)
- [x] End-to-end analysis pipeline driver (`src/analysis/run_pipeline.py`) — ties directions → angular dispersion → cross-domain transfer → bootstrap CIs → figures + report into one command, with a `--mock` mode for CPU dry-runs (validated end-to-end on synthetic data for both concepts)
- [x] **Prompt-pair format migration** — extraction pipeline now reads the new shared-prompt schema (`prompt` + `positive_response`/`negative_response`) used by `data/prompt_pairs/refusal_new/` (728 AdvBench-derived pairs), in addition to the legacy `positive`/`negative` schema. `pair_to_contrastive_texts()` resolves either schema, building `prompt + response` per side so the prompt is held constant. Converter scripts (`convert_advbench.py`, `convert_truthfulqa.py`) updated to emit the new schema; 4 new loader tests added; README format docs updated.
- [x] **Canonical schema across whole dataset** — migrated the 80 hand-written seed pairs (`refusal/`, `honesty/`) from the legacy `positive`/`negative` schema to the shared-prompt schema via `scripts/migrate_seed_to_shared_schema.py` (idempotent; responses preserved verbatim, implicit shared prompt reconstructed per pair). All 12 files now use one schema. Loader still accepts legacy for backward compatibility. The seed files have full response diversity (10/10 unique) — they are the clean contrast set, unlike templated `refusal_new/`.
- [x] **Prompt-pair dataset validator** (`scripts/validate_prompt_pairs.py`) — CPU quality gate that confirms every JSONL file is loadable by the (dual-schema) pipeline and reports schema, valid/malformed/empty/duplicate lines, domain/concept consistency vs. `config.py`, per-domain totals vs. target, source breakdown, and **response-diversity** (distinct positive/negative *responses*). `--strict` makes warnings fail (CI gate). 6 new tests (36 passing total).

---

## 🔴 Immediate — Before June 27

### 1. Git Repository
- [x] Run `git init` and make the first commit — ✅ `725ded0` (June 17)
- [ ] Set up remote on GitHub (private repo)
- [ ] Invite both team members

### 2. Kaggle Environment Setup
- [x] Create Kaggle notebook — scaffolded as `notebooks/01_pilot_gemma2b.py` (jupytext format)
- [ ] Upload to Kaggle and convert to `.ipynb`
- [ ] Install dependencies: `transformer_lens`, `transformers`, `bitsandbytes`
- [ ] Verify GPU access (T4 or P100)
- [ ] Confirm `torch.cuda.is_available()` returns True

### 3. Model Loading Verification (needs GPU)
- [ ] Load **Gemma 2 2B Base** via TransformerLens — verify forward pass
- [ ] Load **Gemma 2 2B Instruct** — verify forward pass
- [ ] Test activation extraction on 1 dummy prompt — verify output shapes
- [ ] If TransformerLens fails → test manual HuggingFace hook fallback

### 4. Pilot Run — 10 Pairs on Gemma 2 2B
- [x] **Analysis pipeline validated on CPU** via `python -m src.analysis.run_pipeline --mock` (de-risks Phase 3 before GPU time)
- [ ] Run `extract_activations.py` on the 10 seed pilot pairs
- [ ] Verify cached activation file naming & shapes
- [ ] Compute pilot directions (global + per-domain) — sanity check
- [ ] Compute pilot cosine similarities — are directions even distinguishable?

### 5. Expand Seed Prompt Pairs → 80 Total ✅
- [x] Refusal/violence: 2 → 10 pairs
- [x] Refusal/illegal_activity: 1 → 10 pairs
- [x] Refusal/medical_legal: 1 → 10 pairs
- [x] Refusal/privacy: 1 → 10 pairs
- [x] Honesty/factual_trivia: 2 → 10 pairs
- [x] Honesty/math: 1 → 10 pairs
- [x] Honesty/politics_opinion: 1 → 10 pairs
- [x] Honesty/personal_advice: 1 → 10 pairs

### 6. Download External Datasets
- [ ] **TruthfulQA** — `datasets.load_dataset("truthful_qa", "generation")` → save to `data/truthfulqa/`
- [ ] **AdvBench** — download from the [AdvBench repo](https://github.com/llm-attacks/llm-attacks) → save to `data/advbench/`
- [x] Write adapter scripts to convert these into our JSONL prompt pair format — `scripts/convert_truthfulqa.py`, `scripts/convert_advbench.py`

### 7. Expression of Interest (June 27 deadline)
- [ ] Draft 2–3 paragraphs: research question, method, model family, expected contribution
- [ ] Both team members review
- [ ] Submit to BlackboxNLP Reproducibility Challenge

### 8. ACL LaTeX Template
- [ ] Download ACL 2025/2026 `acl.sty` from [ACL Anthology GitHub](https://github.com/acl-org/acl-style-files)
- [x] Set up in `paper/` directory
- [x] Create `paper/main.tex` with title, abstract placeholder, section stubs
- [x] Create `paper/references.bib` with 8 key references
- [ ] Verify it compiles (`pdflatex main.tex`) — needs `acl.sty` first

### 9. Concept Operational Definitions
- [x] Write 1-paragraph operational definition: **refusal** — what counts, what doesn't
- [x] Write 1-paragraph operational definition: **honesty** — what counts, what doesn't
- [x] Annotator guidelines added — see `docs/operational_definitions.md`
- [ ] Both team members agree and sign off

---

## 🟡 Phase 2 — Full Extraction (June 28 → July 10)

### 10. Scale Up Prompt Pairs → 300–500 Total

**Current status** (from `python scripts/validate_prompt_pairs.py`, target 200/domain) — **1,720 valid pairs total**:
- Refusal — violence **121**, illegal_activity **191**, medical_legal **456** ✅, privacy **67** (all 4 domains aggregate `refusal/` + `refusal_new/`; +67 hand-written over-refusal pairs added via `scripts/add_refusal_pairs.py`)
- Honesty — factual_trivia **526** ✅, politics_opinion **179**, personal_advice **121**, math **59** (TruthfulQA pulled & merged + 30 hand-written math pairs — real, diverse answers, 0 diversity warnings)
- Sources present: `truthfulqa` (815), `medsafetybench` (445), `advbench` (283), `hand-written` (80). Provenance + caveats in [`docs/dataset_notes.md`](docs/dataset_notes.md).

- [x] **Honesty data pulled from TruthfulQA** — `convert_truthfulqa.py` (817 pairs, shared-prompt schema) + `merge_truthfulqa_into_honesty.py` (split by domain into `honesty/*.jsonl`, deduped, idempotent). Real diverse responses, unlike templated `refusal_new/`.

> ✅ **`refusal_new/` templating — mitigated (interim).** `scripts/finalize_refusal_dataset.py` diversified the 3 canned refusals into natural, domain-aware variants (now 26–168 unique per file); validator warnings cleared. Responses are **still synthetic** — the recommended Phase-2 upgrade is to regenerate the study model's own refusals/compliances on GPU from these genuine prompts (option 2 in `docs/dataset_notes.md`).

- [x] **Honesty pull from TruthfulQA** — done (factual_trivia/politics/personal_advice now at target or above)
- [x] **honesty/math boosted** — added 30 hand-written math honesty pairs (`scripts/add_math_honesty_pairs.py`), now 59. Below the 200 target but a clean, diverse contrast set; add more if aiming for full target.
- [x] **refusal/privacy boosted** — +28 hand-written over-refusal pairs (now 67 aggregate). Below 200 target; add more to close fully.
- [x] **refusal/violence & illegal_activity boosted** — +20 / +19 hand-written pairs (now 121 / 191 aggregate)
- [x] 🎯 **Validity controls in place** — see [`docs/experimental_design.md`](docs/experimental_design.md). Honesty is the clean primary experiment (single-source TruthfulQA → domain not confounded with source). Refusal now has a **canonical prompt-based** path (`scripts/build_refusal_promptbased.py`: harmful vs. harmless instructions, no synthetic responses) under `data/prompt_pairs_promptbased/`, with the response-based set kept as robustness. Balanced extraction via `--max-pairs-per-domain`. Notebook 02 wired to the design-faithful run.
- [x] ⚠️ **Two refusal framings tagged** — seed `refusal/` (*over-refusal*) vs. `refusal_new/` (*harmful-request*) now carry a `framing` field (`over_refusal` / `harmful_request`) so analysis can pool or split them. **Still decide** whether to report them pooled, separately, or both (their direction agreement is itself a fragmentation signal). See `docs/dataset_notes.md`.
- [x] Pull pairs from TruthfulQA and AdvBench — tag source in JSONL
- [x] Document which pairs are sourced vs. hand-written — `scripts/validate_prompt_pairs.py` reports the source breakdown
- [ ] Write domain-split annotation guide

### 11. Harmlessness (Stretch Goal)
- [ ] Define harmlessness domains (3–4 domains)
- [ ] Write 100+ contrastive pairs
- [ ] Only if Person A can deliver before July 3

### 12. Full Extraction — Qwen 2.5 3B Instruct
- [x] Whole-concept extraction driver (`src/extraction/batch_extract.py`) — discovers + aggregates all domain files per concept (refusal ~835, honesty ~885), GPU-free `--dry-run`
- [x] Create `notebooks/02_full_extraction_qwen3b.py` (jupytext) — drives `batch_extract` → `run_pipeline` on the real dataset; re-run with Base checkpoint for the RLHF axis. Convert to `.ipynb` on Kaggle.
- [ ] Run extraction: refusal × 4 domains × all layers (~6h GPU)
- [ ] Run extraction: honesty × 4 domains × all layers (~6h GPU)
- [ ] Cache all activations with proper naming convention
- [ ] Verify shapes, check for NaN/inf

### 13. Full Extraction — Qwen 2.5 3B Base
- [ ] Run same extraction pipeline on the Base checkpoint (~6h GPU)
- [ ] This is the RLHF comparison axis

### 14. Direction Computation
- [ ] Compute **per-domain** directions: 4 domains × all layers × 2 concepts
- [ ] Compute **global** directions: all-domain aggregate × all layers × 2 concepts
- [ ] Save all direction vectors to `results/directions/`

### 15. Midpoint Check (July 3–4)
- [ ] Both review extraction results
- [ ] Sanity check direction vectors — are they non-trivial?
- [ ] **Pivot decision**: if null results emerging → consider Logit Lens fallback (#6)

---

## 🔴 Phase 3 — Analysis & Writing (July 10 → July 14)

### 16. Core Analysis
- [ ] Angular dispersion: cosine sim between per-domain vs. global directions → heatmap
- [ ] Steering transfer test: global direction on held-out prompts per domain → rate shift
- [ ] Cross-domain transfer: domain A direction on domain B prompts → transfer matrix
- [ ] Bootstrap CIs on all effect sizes (1000+ resamples)
- [ ] RLHF comparison: Base vs. Instruct angular dispersion gap

### 17. Key Figures (Publication Quality, 300 DPI)
- [ ] Angular dispersion heatmap (domain × layer)
- [ ] Steering transfer test results
- [ ] Cross-domain transfer matrix
- [ ] RLHF comparison figure (Base vs. Instruct)
- [ ] **PCA/UMAP geometry figure** ← *most important figure in the paper*
- [ ] Deliver all figures to Person A by **July 11**

### 18. Paper Writing
- [ ] Introduction + motivation (Person A)
- [ ] Related Work: Zou et al., Hernandez et al., Marks & Tegmark + 2–3 steering papers
- [ ] Methods: contrastive prompts, extraction, direction computation, steering protocol
- [ ] Results section (as soon as figures are ready)
- [ ] Discussion: what fragmentation/null result means for the field
- [ ] Conclusion + Limitations
- [ ] Format in ACL LaTeX, check page limits (8 pages + refs)

### 19. Review Sprint (July 12–14)
- [ ] Both read full draft end-to-end
- [ ] Check consistency between claims and figures
- [ ] Full proofreading pass (each person)
- [ ] Cross-check all citations against actual papers
- [ ] Final OpenReview submission — both approve before submitting

---

## 🟢 Phase 4 — Reproducibility Track (July 15–27, Optional)

- [ ] Only if main paper submitted AND interesting failure modes found
- [ ] **Logit Lens fallback**: run on same Qwen checkpoints using cached activations
- [ ] OR: reproducibility appendix + code/data release on GitHub
- [ ] Submit by **July 27**

---

## 🟢 Phase 5 — Camera Ready (If Accepted, Sep 17)

- [ ] Address reviewer comments
- [ ] Final figure polish (font sizes, labels, colorblind-safe)
- [ ] Acknowledgments section
- [ ] Submit camera-ready by **Sep 17**

---

## ⚡ Compute Budget Remaining

| Session | Task | Est. GPU hrs | Phase |
|---|---|---|---|
| 1 | Pilot — Gemma 2 2B, 10 pairs | ~5h | Phase 1 |
| 2 | Full extraction — Qwen 3B Instruct | ~6h | Phase 2 |
| 3 | Direction computation (CPU) | ~3h | Phase 3 |
| 4 | Steering + cross-domain transfer | ~6h | Phase 3 |
| 5 | Base checkpoint extraction (RLHF) | ~6h | Phase 2–3 |
| 6 | Buffer / Logit Lens fallback | ~4–6h | Flex |
| **Total** | | **~26–32h** | |

---

## ⚠️ Key Risks to Watch

| Risk | What to Do |
|---|---|
| Null result — global direction works fine everywhere | Frame as positive reproducibility result → submit to Repro Track |
| Kaggle GPU quota exhausted | Switch to Gemma 2 2B for remaining runs, use 4-bit quant, cache aggressively |
| Prompt pairs lack quality | Pull more from TruthfulQA/AdvBench; Person B reviews while Person A writes new |
| TransformerLens doesn't support model | Fallback to manual HuggingFace hooks (already implemented in `hooks.py`) |
| Team bandwidth mismatch | Scripts are documented — either person can run extractions. Scope to 2 concepts if tight. |

---

## 📅 Key Dates

| Date | Milestone |
|---|---|
| **June 27** | Expression of Interest (Repro Challenge) |
| **July 3–4** | Midpoint check — pivot decision |
| **July 11** | All figures delivered |
| **July 14** | Internal submission deadline |
| **July 17** | Official OpenReview deadline |
| **July 27** | Reproducibility Challenge submission |
| **Sep 8** | Acceptance notification |
| **Sep 17** | Camera-ready due |
