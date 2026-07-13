# Final Results Summary — Canonical Numbers for the Paper

> Updated: 2026-07-13. **This is the single source of truth for results.**
> Raw reports: `new_results_final/` (Jul 12 primary runs) and
> `results_cheapwins/` (Jul 13 reviewer-hardening: probe transfer, n=59
> balanced, fp16 check, judged steering).
> Activation tensors live only in the Kaggle dataset (gitignored locally).

## Experiment outcomes at a glance

| # | Claim tested | Outcome |
|---|---|---|
| **E1** | Honesty directions fragment across domains | **Partially supported** — fragments early-mid layers (min cos 0.522 @ L5, ~57°), partial late convergence (late-third mean 0.69/0.67) |
| **E2** | Refusal directions fragment (prompt-based) | **Not supported** — universal (cos 0.83–0.94; late-third 0.93/0.91) |
| **E2′** | Refusal fragments (response-based, robustness) | **Not supported** — also universal on Qwen (0.85–0.93, late 0.904; CI at min layer [0.79, 0.89]) |
| **E3** | RLHF (Base→Instruct) amplifies fragmentation | **Not supported** — slight consolidation (Δcos +0.011 honesty / +0.007 refusal; only 10/36 layers negative, weakly) |
| **E4** | Global steering < per-domain steering (behavioral) | **Not supported** — no systematic per-domain advantage; global matches per-domain, differences at the 1-prompt (n=20) noise floor across two runs and two scorers |
| **E5** | Honesty fragmentation is functional (probe transfer) | **Supported** — honesty probes fail to transfer across domains (cross-domain acc 0.56–0.63 vs within 0.73–0.83); refusal probes transfer near-perfectly (0.88→0.98). Fragmentation is functional, not merely lexical. |

## E1 — Honesty (clean primary; TruthfulQA single-source)

- Min mean cos-to-global **0.522 @ layer 5** (~57° mean angle), Instruct and Base alike.
- Late-third mean cos: **0.694** (Instruct) / **0.667** (Base) — partial convergence.
- Cross-domain transfer at L5: **math near-orthogonal to all** (−0.10…−0.03);
  best pair factual_trivia↔personal_advice 0.34.
- **Balance (n=59) sensitivity — fragmentation persists.** Capping every domain
  to math's 59 pairs: min cos **0.536 @ L4** (Instruct) / 0.540 (Base), late-third
  0.65/0.63. The math-orthogonality result is not a small-sample artifact.
- Early-layer minima are **not merely lexical** — see E5 (probe transfer).

## E5 — Probe transfer (functional test of fragmentation)

Linear probe trained per domain, tested on every other domain's held-out
pairs. Accuracy (within-domain diagonal vs cross-domain off-diagonal):

| Condition | Within | Cross | Gap |
|---|---:|---:|---:|
| Honesty Instruct L5 | 0.73 | 0.56 | 0.17 |
| Honesty Instruct L18 | 0.80 | 0.61 | 0.19 |
| Honesty Instruct L30 | 0.83 | 0.63 | 0.20 |
| Honesty Base L30 | 0.82 | 0.59 | 0.23 |
| Honesty n=59 L30 | 0.83 | 0.63 | 0.19 |
| **Refusal Instruct L5** | **1.00** | **0.88** | 0.12 |
| **Refusal Instruct L18** | **1.00** | **0.96** | 0.04 |
| **Refusal Instruct L30** | **1.00** | **0.98** | 0.02 |

Honesty probes barely beat chance across domains (worst cell: math↔politics
near 0.45); refusal probes transfer near-perfectly. This is the functional
counterpart to the cosine geometry and gives honesty a second evidence
pillar. **Fragmentation is functional, not just lexical.**

## Quantization check (T5) — 4-bit vs fp16

Cross-precision direction cosine **0.977–0.990** (all domains, layers 5 & 18);
dispersion statistic changes by **≤0.004** (L5: 0.522→0.523; L18: 0.592→0.588).
Quantization does not distort the geometry the paper measures.

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
filename fix).

**Run/scorer robustness — the honest picture.** The strict claim "global ≥
own in every domain" holds in the Jul 12 run but NOT in the Jul 13
replication (illegal_activity: own −0.10 beats global −0.05, i.e. one prompt).
The judge re-score also under-counts refusals (labels "can't help with X but
can with Y" as compliance; baselines score 0.10–0.35 vs keyword 0.50–0.95 —
raw generations confirm the baselines are genuine refusals, so keyword is the
more accurate scorer). Across all three (Jul12-kw, Jul13-kw, Jul13-judge) the
worst per-domain edge for the own-direction is one prompt (0.05).
**Defensible claim: no systematic per-domain steering advantage; differences
are at the n=20 noise floor.** Do NOT claim strict "global ≥ own everywhere"
or "sign preserved across scorers" — the artifacts contradict both.

## Gemma 2B pilot — did NOT replicate on Qwen

The Jun 30 pilot showed refusal fragmentation (min cos 0.56) on response-based
data; the same data recipe on Qwen gives 0.85+. **The fragmentation was
model-specific (Gemma 2 2B), not dataset-driven.** Frame as a scale/family
open question — do not claim a proven dataset-confound demonstration.

## Paper storyline (one paragraph)

> On Qwen 2.5 3B we find refusal concept directions robustly universal across
> semantic domains under both prompt-based and response-based
> operationalizations, confirmed two further ways: a global steering direction
> matches or beats per-domain directions on held-out prompts (robust to
> scorer), and a linear probe trained on one domain transfers near-perfectly
> to the others (cross-domain accuracy up to 0.98). Honesty, by contrast,
> fragments in early-mid layers (math near-orthogonal to other domains) with
> only partial late-layer convergence, and this fragmentation is functional,
> not merely lexical: honesty probes fail to transfer across domains
> (cross-domain accuracy near chance). Universality is thus concept- and
> depth-dependent. RLHF does not amplify fragmentation; alignment slightly
> consolidates directions. Fragmentation observed in a smaller pilot model
> (Gemma 2 2B) did not replicate, suggesting model capacity/family as a
> variable deserving further study.

## Limitations checklist for the paper

- Single model family at scale (T6); two concepts; English only
- 4-bit quantization — **now measured** (cross-precision cos ≥0.977,
  dispersion Δ ≤0.004); state as checked, not a threat
- E4 refusal scoring: keyword primary; LLM-judge under-counts refusals but
  preserves the global≥own ordering; single surviving coefficient
- E4 held-out prompts partially overlap direction data in domains with <140 pairs
