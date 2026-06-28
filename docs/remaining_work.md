# Remaining Work & Time Estimates

> Snapshot: June 28, 2026. Companion to `next_steps.md` (full plan) and
> `docs/dataset_notes.md` (data caveats). This file is the short "what's left
> and how long" view.

## Where we are

- ✅ Full pipeline coded + tested on CPU/mock: extraction (single + batch) →
  directions → angular dispersion → cross-domain transfer → bootstrap CIs →
  figures + report.
- ✅ Dataset: **1,720 pairs**, one canonical shared-prompt schema, 0 validator
  warnings. Refusal ~835 (4 domains), honesty ~885 (4 domains).
- ✅ Phase-2 Kaggle notebook (`notebooks/02_full_extraction_qwen3b.py`) wired to
  the real pipeline.
- ⏳ Everything below needs a **GPU**, a **human decision**, or **writing** — none
  of it is blocked by code.

Legend — **Effort**: hands-on time. **GPU**: needs a GPU session. Estimates assume
one person familiar with the repo.

---

## 1. Decisions to make first (do before GPU) — ~2–3 hrs total

These are cheap but **gate** the GPU runs, so settle them first.

| Task | Effort | Notes |
|---|---|---|
| Refusal **framing**: pool `over_refusal` + `harmful_request`, or analyze separately (or both) | 30 min | Field already tagged. Recommend "both" — agreement is a fragmentation signal. |
| `refusal_new` responses: keep diversified synthetic, **or** regenerate real ones on GPU | 30 min | See §2. Decision drives whether to add the generation step. |
| Confirm **MedSafetyBench** license/citation (445 pairs) | 30–60 min | Needed before publishing. |
| Harmlessness stretch concept: in or out? | 30 min | Only if Phase-2 bandwidth allows. |
| Sign off `docs/operational_definitions.md` (both members) | 30 min | Currently "draft — needs sign-off". |

---

## 2. GPU work (Kaggle T4/P100) — ~20–30 GPU hrs

Run `notebooks/02_full_extraction_qwen3b.py`. Budget mirrors `next_steps.md`.

| Task | Effort | GPU | Notes |
|---|---|---|---|
| Upload repo to Kaggle, convert notebook to `.ipynb`, verify GPU + model load | 1–2 hrs | ✅ ~1 hr | Gemma/Qwen forward-pass sanity check. |
| *(optional)* Generate model's own refusal/compliance responses for `refusal_new` | 2–3 hrs | ✅ ~3–4 hr | Only if §1 chose "regenerate". Most faithful refusal data. |
| Extract — **Qwen 3B Instruct**, both concepts, all layers | ~30 min setup | ✅ ~6 hr | `batch_extract` → cached activations. |
| Extract — **Qwen 3B Base** (RLHF axis) | ~15 min | ✅ ~6 hr | Same run, Base checkpoint. |
| Run `run_pipeline` for each concept × checkpoint (directions, dispersion, transfer, CIs, figures) | ~1 hr | ◑ ~3 hr | Mostly CPU; light GPU if steering. |
| Steering transfer experiments | 1–2 hrs | ✅ ~6 hr | Global vs per-domain direction on held-out prompts. |
| **Buffer / re-runs** | — | ✅ ~4–6 hr | Quota hiccups, fixes. |

**Subtotal: ~20–30 GPU hrs over ~3–5 working sessions.**

---

## 3. Analysis & figures — ~1–2 days

Largely automated by `run_pipeline`; this is interpretation + polish.

| Task | Effort | Notes |
|---|---|---|
| Midpoint sanity check — are directions non-trivial? Pivot if null | 2–3 hrs | If null → frame as reproducibility result. |
| Finalize 5 publication figures (heatmap, dispersion, transfer matrix, RLHF, PCA/UMAP) | 4–6 hrs | PCA/UMAP geometry = key figure. 300 DPI, colorblind-safe. |
| RLHF comparison: Base vs Instruct dispersion gap | 2–3 hrs | Hypothesis #3. |
| Per-framing direction agreement (if "both" chosen) | 1–2 hrs | Extra evidence for fragmentation. |

---

## 4. Writing & submission — ~4–6 days

| Task | Effort | Notes |
|---|---|---|
| Download `acl.sty`, confirm `paper/main.tex` compiles | 30 min | Known blocker; trivial once downloaded. |
| Expression of Interest (Repro Challenge) | 2–3 hrs | **Deadline was June 27** — check if still open. |
| Intro + Related Work | 1 day | Zou, Marks & Tegmark, Hernandez, + steering papers. |
| Methods + Results | 1–2 days | Results flow from §3 figures. |
| Discussion + Limitations + Conclusion | 1 day | What fragmentation/null means for the field. |
| Review sprint (both read end-to-end, citations, proofread) | 1–2 days | Before submission. |

---

## 5. Admin — ~1–2 hrs

| Task | Effort |
|---|---|
| Create private GitHub remote + invite both members | 30 min |
| Push committed history (4 recent commits on `main`) | 15 min |

---

## Optional / stretch (only if ahead)

| Task | Effort | Notes |
|---|---|---|
| Push thin domains to 200 target (math 59, privacy 67, violence 121) | 3–6 hrs | More hand-written pairs via existing `add_*` scripts. |
| Harmlessness concept (3–4 domains, 100+ pairs) | 1 day | Stretch goal. |
| Logit Lens fallback (Repro Track) | 1 day | Reuses cached activations. |

---

## Critical path & key dates

```
Decisions (§1) ─▶ GPU extraction (§2) ─▶ Analysis/figures (§3) ─▶ Writing (§4) ─▶ Submit
```

| Date | Milestone |
|---|---|
| **July 3–4** | Midpoint check / pivot decision |
| **July 11** | All figures delivered |
| **July 14** | Internal deadline |
| **July 17** | OpenReview deadline |
| **July 27** | Reproducibility Challenge |

**Rough total to submission: ~8–12 focused person-days + ~20–30 GPU hrs.**
The bottleneck is GPU availability and writing time, not code — the pipeline is
ready to run.
