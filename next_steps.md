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
- [x] 10 seed pilot prompt pairs (5 refusal + 5 honesty)
- [x] 26 smoke tests — all passing
- [x] README, requirements.txt, .gitignore

---

## 🔴 Immediate — Before June 27

### 1. Git Repository
- [ ] Run `git init` and make the first commit
- [ ] Set up remote on GitHub (private repo)
- [ ] Invite both team members

### 2. Kaggle Environment Setup
- [ ] Create Kaggle notebook `notebooks/01_pilot_gemma2b.ipynb`
- [ ] Install dependencies: `transformer_lens`, `transformers`, `bitsandbytes`
- [ ] Verify GPU access (T4 or P100)
- [ ] Confirm `torch.cuda.is_available()` returns True

### 3. Model Loading Verification (needs GPU)
- [ ] Load **Gemma 2 2B Base** via TransformerLens — verify forward pass
- [ ] Load **Gemma 2 2B Instruct** — verify forward pass
- [ ] Test activation extraction on 1 dummy prompt — verify output shapes
- [ ] If TransformerLens fails → test manual HuggingFace hook fallback

### 4. Pilot Run — 10 Pairs on Gemma 2 2B
- [ ] Run `extract_activations.py` on the 10 seed pilot pairs
- [ ] Verify cached activation file naming & shapes
- [ ] Compute pilot directions (global + per-domain) — sanity check
- [ ] Compute pilot cosine similarities — are directions even distinguishable?

### 5. Expand Seed Prompt Pairs → 80 Total
- [ ] Refusal/violence: 2 → 10 pairs
- [ ] Refusal/illegal_activity: 1 → 10 pairs
- [ ] Refusal/medical_legal: 1 → 10 pairs
- [ ] Refusal/privacy: 1 → 10 pairs
- [ ] Honesty/factual_trivia: 2 → 10 pairs
- [ ] Honesty/math: 1 → 10 pairs
- [ ] Honesty/politics_opinion: 1 → 10 pairs
- [ ] Honesty/personal_advice: 1 → 10 pairs

### 6. Download External Datasets
- [ ] **TruthfulQA** — `datasets.load_dataset("truthful_qa", "generation")` → save to `data/truthfulqa/`
- [ ] **AdvBench** — download from the [AdvBench repo](https://github.com/llm-attacks/llm-attacks) → save to `data/advbench/`
- [ ] Write adapter scripts to convert these into our JSONL prompt pair format

### 7. Expression of Interest (June 27 deadline)
- [ ] Draft 2–3 paragraphs: research question, method, model family, expected contribution
- [ ] Both team members review
- [ ] Submit to BlackboxNLP Reproducibility Challenge

### 8. ACL LaTeX Template
- [ ] Download ACL 2025/2026 template from [ACL Anthology GitHub](https://github.com/acl-org/acl-style-files)
- [ ] Set up in `paper/` directory
- [ ] Create `paper/main.tex` with title, abstract placeholder, section stubs
- [ ] Verify it compiles (`pdflatex main.tex`)

### 9. Concept Operational Definitions
- [ ] Write 1-paragraph operational definition: **refusal** — what counts, what doesn't
- [ ] Write 1-paragraph operational definition: **honesty** — what counts, what doesn't
- [ ] Both team members agree and sign off

---

## 🟡 Phase 2 — Full Extraction (June 28 → July 10)

### 10. Scale Up Prompt Pairs → 300–500 Total
- [ ] Refusal: 150–250 pairs across 4 domains
- [ ] Honesty: 150–250 pairs across 4 domains
- [ ] Pull pairs from TruthfulQA and AdvBench — tag source in JSONL
- [ ] Document which pairs are sourced vs. hand-written
- [ ] Write domain-split annotation guide

### 11. Harmlessness (Stretch Goal)
- [ ] Define harmlessness domains (3–4 domains)
- [ ] Write 100+ contrastive pairs
- [ ] Only if Person A can deliver before July 3

### 12. Full Extraction — Qwen 2.5 3B Instruct
- [ ] Create `notebooks/02_full_extraction_qwen3b.ipynb`
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
