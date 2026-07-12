# Final Results Summary — Canonical Numbers for the Paper

> Updated: 2026-07-12. **This is the single source of truth for results.**
> Raw reports: `new_results_final/` (Jul 12 session — fixed pipeline: pair-level
> bootstrap CIs, corrected transfer layers, late-layer stats, chat-template E4).
> Activation tensors live only in the Kaggle dataset (gitignored locally).

## Experiment outcomes at a glance

| # | Claim tested | Outcome |
|---|---|---|
| **E1** | Honesty directions fragment across domains | **Partially supported** — fragments early-mid layers (min cos 0.522 @ L5, ~57°), partial late convergence (late-third mean 0.69/0.67) |
| **E2** | Refusal directions fragment (prompt-based) | **Not supported** — universal (cos 0.83–0.94; late-third 0.93/0.91) |
| **E2′** | Refusal fragments (response-based, robustness) | **Not supported** — also universal on Qwen (0.85–0.93, late 0.904; CI at min layer [0.79, 0.89]) |
| **E3** | RLHF (Base→Instruct) amplifies fragmentation | **Not supported** — slight consolidation (Δcos +0.011 honesty / +0.007 refusal; only 10/36 layers negative, weakly) |
| **E4** | Global steering < per-domain steering (behavioral) | **Not supported** — global ≥ own-domain in every refusal domain |

## E1 — Honesty (clean primary; TruthfulQA single-source)

- Min mean cos-to-global **0.522 @ layer 5** (~57° mean angle), Instruct and Base alike.
- Late-third mean cos: **0.694** (Instruct) / **0.667** (Base) — partial convergence.
- Cross-domain transfer at L5: **math near-orthogonal to all** (−0.10…−0.03);
  best pair factual_trivia↔personal_advice 0.34.
- Caveat: early-layer minima may reflect surface/token statistics (noted in
  reports); math domain n=59 (< 120 cap) — balance sensitivity outstanding.

## E2 / E2′ — Refusal (both operationalizations universal on Qwen)

| | Prompt-based (E2) | Response-based (E2′) |
|---|---|---|
| Mean cos range across layers | 0.83–0.94 | 0.85–0.93 |
| Late-third mean cos | 0.928 (Instr) / 0.911 (Base) | 0.904 (Instr) |
| Pair-level CI at min layer | — | [0.791, 0.885] |

## E3 — RLHF axis

Instruct is slightly *more* universal than Base: mean Δcos **+0.011** (honesty),
**+0.007** (refusal); gap widens late (honesty L35: 0.690→0.769). Base–Instruct
per-domain differences grow with depth (max 0.20) — the expected fine-tune
signature, confirming distinct checkpoints.

## E4 — Behavioral steering (Qwen Instruct, layer 18, coeff 4.0, n=20/domain, chat template)

| Domain | Baseline refusal | Global Δ | Own Δ | own − global |
|---|---:|---:|---:|---:|
| violence | 0.95 | 0.00 | 0.00 | 0.00 |
| illegal_activity | 0.90 | −0.05 | −0.05 | 0.00 |
| medical_legal | 0.50 | −0.15 | 0.00 | +0.15 |
| privacy | 0.50 | −0.40 | −0.15 | +0.25 |

Global direction steers **as well or better** than domain-specific directions
everywhere; cross-domain directions also transfer (privacy steered −0.40 by
illegal_activity/violence directions). Violence is behaviorally immovable at
this coefficient. Only coeff 4.0 survives (2.0 run was overwritten before the
filename fix); refusal detection is a keyword heuristic — both go in Limitations.

## Gemma 2B pilot — did NOT replicate on Qwen

The Jun 30 pilot showed refusal fragmentation (min cos 0.56) on response-based
data; the same data recipe on Qwen gives 0.85+. **The fragmentation was
model-specific (Gemma 2 2B), not dataset-driven.** Frame as a scale/family
open question — do not claim a proven dataset-confound demonstration.

## Paper storyline (one paragraph)

> On Qwen 2.5 3B we find refusal concept directions robustly universal across
> semantic domains under both prompt-based and response-based
> operationalizations, confirmed behaviorally: a global steering direction
> matches or beats per-domain directions on held-out prompts. Honesty, by
> contrast, fragments substantially in early-mid layers (math near-orthogonal
> to other domains) with only partial late-layer convergence — universality is
> concept- and depth-dependent. RLHF does not amplify fragmentation; alignment
> slightly consolidates directions. Fragmentation observed in a smaller pilot
> model (Gemma 2 2B) did not replicate, suggesting model capacity/family as a
> variable deserving further study.

## Limitations checklist for the paper

- 4-bit quantization (T5); single model family at scale (T6)
- Keyword-based refusal detection in E4; single surviving coefficient
- honesty/math n=59; domain-balance sensitivity not run (`--balance-domains`)
- Early-layer honesty fragmentation may be partly lexical
- E4 held-out prompts partially overlap direction data in domains with <140 pairs
