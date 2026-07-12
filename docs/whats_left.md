# What's Left — Final 5 Days (Writing Sprint)

> Updated: 2026-07-12 · **Deadline: July 17 (OpenReview)** · internal: July 14
> Numbers: [`docs/results_summary.md`](results_summary.md) (canonical) ·
> Raw reports: `new_results_final/` · Validity: [`docs/experimental_design.md`](experimental_design.md)

## Status: ALL EXPERIMENTS DONE ✅ — only writing remains

E1 (honesty), E2 (prompt-based refusal), E2′ (response-based robustness),
E3 (RLHF axis), E4 (behavioral steering) are all complete and valid.
See `results_summary.md` for every number the paper needs.

## Folder map (what exists where)

| Location | Contents |
|---|---|
| `new_results_final/` | **Canonical results**: 4 Qwen + 2 Gemma reports (JSON+MD), E4 steering JSON, E2′ under `respbased/`, all figures (`figures/`, `figures/qwen/`, `figures/qwen respbased/`) |
| Kaggle dataset | Activation tensors (`.pt`) — only copy; keep the dataset alive |
| `results/` | **No longer exists** — deleted; code still writes there by default on fresh runs |
| `data/prompt_pairs_promptbased/` | Gitignored/regenerable: `python scripts/build_refusal_promptbased.py --max-per-domain 120` |

## Day-by-day plan (Jul 12 → 17)

### Day 1–2 (Jul 12–13): Methods + Results
- [ ] Methods from `experimental_design.md` + `config.py` (resid_post, last token,
  diff-of-means, 4-bit, pair-level bootstrap, chat-template steering protocol)
- [ ] Results: tables/figures straight from `results_summary.md` +
  `new_results_final/figures/qwen*/`
- [ ] Set up Overleaf; `paper/acl.sty` + `acl_natbib.bst` already vendored;
  switch to `\usepackage[review]{acl}`

### Day 3 (Jul 14 — internal deadline): Intro, Related Work, Discussion
- [ ] Intro: question form matches findings (title already updated)
- [ ] Related Work: Zou 2023, Marks & Tegmark, Hernandez, Arditi 2024, Turner
  2023, Li 2024, Wei 2024 (`paper/references.bib` has stubs)
- [ ] Discussion: universality is concept- and depth-dependent; RLHF
  consolidates; Gemma non-replication = scale/family question (do NOT claim
  proven dataset confound — E2′ killed that framing)

### Day 4 (Jul 15): Limitations, Abstract, full pass
- [ ] Limitations checklist is pre-written in `results_summary.md`
- [ ] Abstract last; both authors read end-to-end; citation check

### Day 5 (Jul 16–17): Buffer + submit
- [ ] Page limit check (8 + refs), figures ≥300 DPI colorblind-safe
- [ ] OpenReview submission — both approve first

## Reviewer hardening (built 2026-07-12) — run via [`docs/kaggle_runbook.md`](kaggle_runbook.md)

Step-by-step instructions for whoever runs the Kaggle session are in the
runbook (self-contained, no other context needed). Equivalent notebook:
`notebooks/04_cheap_wins_kaggle.py`.

Mock review flagged four weaknesses; all four analyses are now coded and
tested (56 tests). Priority: 1 > 2 > 3 > 4; items 1–2 are CPU-only.

- [ ] **1. Probe transfer matrix** (`src/analysis/probe_transfer.py`) — trains
  a linear probe per honesty domain, tests cross-domain; gives honesty a
  functional evidence pillar. ⚠️ Result cuts either way: high transfer means
  the cosine fragmentation is superficial/lexical — the paper text follows
  the data.
- [ ] **2. n=59 balanced honesty rerun** — `run_pipeline --balance-domains`
  (already built Jul 7), defends the math-orthogonality number.
- [ ] **3. fp16 spot-check** — `batch_extract --no-quantization` (layers 5,
  18) + `scripts/compare_quantization.py`; turns the T5 limitation into a
  measured non-issue (~1h GPU).
- [ ] **4. Steering rerun + LLM judge** — `run_steering` now persists all
  generations to a `*_generations.jsonl` sidecar (the Jul 12 run kept only 3
  truncated examples, so judging required a rerun); `scripts/judge_steering.py`
  re-scores them with the study model as judge (~4h GPU).
- [ ] Integrate results into the paper (probe matrix table in §5.2, one
  sentence each for items 2–4).

## Admin still open
- [ ] Sign off `docs/operational_definitions.md` (both members)
- [x] MedSafetyBench license (MIT, research-only — in `dataset_notes.md`)
- [x] GitHub remote up to date
