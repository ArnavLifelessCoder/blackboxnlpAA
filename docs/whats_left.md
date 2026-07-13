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

## Reviewer hardening — DONE (Jul 13, results in `results_cheapwins/`)

All four analyses ran. Outcomes (canonical numbers in `results_summary.md`):

- ✅ **1. Probe transfer (E5)** — the big win. Honesty probes fail to transfer
  across domains (cross 0.56–0.63 vs within 0.73–0.83); refusal probes
  transfer near-perfectly (0.88–0.98). Fragmentation is **functional, not
  lexical** — §5.2 keeps "honesty fragments" and gets stronger. New table +
  paragraph for the paper.
- ✅ **2. n=59 balanced honesty** — fragmentation persists (min cos 0.536 @ L4).
  Math-orthogonality is not a small-sample artifact. One sentence in §5.2.
- ✅ **3. fp16 check** — cross-precision cos ≥0.977, dispersion Δ ≤0.004.
  Quantization is a measured non-issue; moves from Limitations to a checked
  robustness sentence.
- ⚠️ **4. Judged steering** — did NOT cleanly validate. The judge under-counts
  refusals (labels "can't help with X but can with Y" as compliance), scoring
  obvious-refusal baselines at 0.10–0.35; keyword scorer is more accurate.
  BUT the global≥own ordering holds under both scorers → use as a
  scorer-robustness sentence in §5.3, keyword rates stay primary.

## Remaining before submission (Jul 13 → 17)

- [ ] Integrate the four into `paper/main.tex`: probe-transfer table +
  paragraph (§5.2), balanced sentence (§5.2), fp16 sentence (§5.2/Limitations),
  judge robustness sentence (§5.3). Update Limitations.
- [ ] Recompile on Overleaf; check page count (probe table adds ~0.3 pg).
- [ ] Both authors read end-to-end; submit by Jul 17.

## Admin still open
- [ ] Sign off `docs/operational_definitions.md` (both members)
- [x] MedSafetyBench license (MIT, research-only — in `dataset_notes.md`)
- [x] GitHub remote up to date
