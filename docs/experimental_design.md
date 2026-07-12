# Experimental Design & Validity

> Purpose: lock the study to claims that survive review. Read with
> `docs/dataset_notes.md` (data caveats) and `next_steps.md` (plan).

## Research question

> Do concept directions (refusal, honesty) generalize uniformly across semantic
> **domains**, or fragment into domain-specific sub-directions — and does RLHF
> (Base→Instruct) amplify the fragmentation?

For any fragmentation claim to hold, **domain must be the only thing varying**
between the buckets we compare. The threats below are where that breaks, and how
we control each one.

---

## Threats to validity & controls

### T1 — Domain confounded with *source dataset* (most dangerous, refusal only)

Refusal domains came from different datasets: `medical_legal` from
MedSafetyBench, the others from AdvBench. So a "medical vs violence" direction
difference could be a *MedSafetyBench vs AdvBench* difference (authorship,
phrasing, length), not domain fragmentation.

**Controls:**
- **Honesty is the clean primary experiment.** Its domains (`factual_trivia`,
  `politics_opinion`, `personal_advice`) all come from **one source**
  (TruthfulQA categories), real Q+A. Domain is *not* confounded with source.
  **Lead the paper with honesty.**
- For refusal, treat the source confound as an explicit limitation, and reduce
  it by using a **shared harmless baseline** (see T2) so at least the control
  side is constant across domains.

### T2 — Synthetic / templated responses (refusal)

`refusal_new/` pairs real prompts with **synthetic** responses (echo positives,
templated refusals). Reading activations on `prompt + synthetic_response` risks
encoding the templates, not refusal behavior.

**Control — use the canonical prompt-based method.**
`scripts/build_refusal_promptbased.py` builds the refusal contrast from
**harmful vs. harmless instructions** (Arditi et al. 2024; Zou et al. 2023),
read at the prompt's last token, **no responses**. Output:
`data/prompt_pairs_promptbased/refusal/`. This is the method reviewers expect and
removes the template confound entirely.

> The response-based `refusal_new/` set (with diversified refusals) is kept as a
> **secondary / robustness** condition, not the headline.

### T3 — Two refusal *framings* pooled

`refusal/` seeds are *over-refusal* (benign→refused); `refusal_new/` is
*harmful-request* (harmful→refused). These are different axes. Each pair carries
a `framing` tag (`over_refusal` / `harmful_request` / `prompt_contrast`).
**Analyze framings separately**; do not pool them into one "refusal direction."

### T4 — Domain imbalance rigs the global direction

Raw counts are lopsided (medical 456, factual_trivia 526). The global direction
would be dominated by the largest domain, pre-determining "global vs per-domain."

**Control — balance by capping** each domain to a common n at extraction time:
`batch_extract --max-pairs-per-domain N`, and the prompt-based builder caps
(`--max-per-domain`). Privacy was the limiter (only 29 real harmful prompts), so
it was topped up with hand-written harmful privacy requests
(`data/harmful_prompts_supplement.jsonl`) — prompt-based refusal is now balanced
at **~91–100 per domain** (privacy 100). Still report sensitivity to the cap.

### T5 — Quantization (hardware)

Qwen 3B runs 4-bit on Kaggle. 4-bit may distort activation geometry.

**Control:** state it as a limitation; if budget allows, replicate one domain in
fp16 to show the dispersion pattern is not a quantization artifact.

### T6 — Single model family (hardware)

Qwen (primary) + Gemma 2 2B (pilot) = limited external validity.

**Control — scope the claim.** Don't claim a universal law from one family.
Frame as a focused **reproduction / stress-test of RepE's universality
assumption** under controlled conditions — exactly the Reproducibility /
Generalizability track's remit (our EoI theme). A clean, well-controlled
*fragmentation* (or *null*) result is publishable there; a confounded positive
is not.

### T7 — Geometry without behavior

Cosine-similarity heatmaps alone invite "so what?" The fragmentation claim needs
a **behavioral** test.

**Control — spend the scarce GPU on steering** (Hypothesis 2): show a global
direction steers worse than a domain-specific one on held-out prompts. A
direction that *changes behavior* is the credibility anchor; prioritize it over
extra extraction.

---

## The experiments we actually claim

| # | Claim | Data | Outcome (2026-07-12, see `docs/results_summary.md`) |
|---|---|---|---|
| **E1 (primary)** | Honesty directions fragment across domains | TruthfulQA (single source, real) | **Partially supported** — early-mid layers (0.522 @ L5), partial late convergence |
| **E2** | Refusal directions fragment | **prompt-based** (harmful vs harmless) | **Not supported** — universal (0.83–0.94) |
| **E2′ (robustness)** | Same, response-based | `refusal_new/` diversified | **Not supported** — also universal on Qwen (0.85–0.93); Gemma pilot fragmentation did not replicate |
| **E3** | RLHF amplifies fragmentation | E1/E2 on Base vs Instruct | **Not supported** — slight consolidation (Δcos +0.01) |
| **E4 (behavioral)** | Global steering < per-domain steering | held-out prompts | **Not supported** — global ≥ own-domain everywhere |

**Honest framing for the paper:** "We reproduce and stress-test the universality
of representation-engineering concept directions on a controlled, single-source
setup (honesty), with a canonical prompt-based refusal replication and a Base-vs-
Instruct RLHF comparison, validated behaviorally via steering." That sentence is
defensible. "We proved concept directions aren't universal" is not.

---

## What to run (design-faithful)

```bash
# Refusal — canonical prompt-based, balanced
python scripts/build_refusal_promptbased.py --max-per-domain 100
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept refusal \
    --data-dir data/prompt_pairs_promptbased

# Honesty — primary, balanced
python -m src.extraction.batch_extract --model qwen-2.5-3b-instruct --concept honesty \
    --max-pairs-per-domain 120

# Repeat both for --model qwen-2.5-3b (Base) -> E3
# Then run_pipeline per concept/checkpoint; analyze framings separately.
```
