# Dataset Notes — Prompt-Pair Provenance & Quality

> **Status**: Living document. Run `python scripts/validate_prompt_pairs.py` to
> regenerate the current numbers.
> **Last updated**: June 22, 2026

This document records where each prompt-pair set comes from and known quality
caveats that affect direction extraction. **Read this before running extraction.**

---

## Sources

| `source` tag | Dataset | Used for | Notes |
|---|---|---|---|
| `hand-written` | Authored by the team | `refusal/`, `honesty/` seed pilots (80 pairs) | Migrated to the shared-prompt schema (`scripts/migrate_seed_to_shared_schema.py`); responses preserved verbatim, shared prompt reconstructed. **Full response diversity (10/10 unique) — the clean contrast set.** |
| `truthfulqa` | [TruthfulQA](https://github.com/sylinrl/TruthfulQA) (generation split, 817 Qs) | `honesty/` (all 4 domains) | Real Q + best/incorrect answers → prompt + positive/negative_response. **High response diversity** (≈745/786 unique). Pulled via `convert_truthfulqa.py` + `merge_truthfulqa_into_honesty.py`. |
| `advbench` | [AdvBench](https://github.com/llm-attacks/llm-attacks) harmful behaviors | `refusal_new/` (violence, illegal_activity, privacy) | Real, diverse harmful prompts. **Responses are synthetic templates** (see below). |
| `medsafetybench` | [MedSafetyBench](https://github.com/AI4LIFE-GROUP/med-safety-bench) (NeurIPS 2024) — harmful medical requests derived from the AMA Principles of Medical Ethics | `refusal_new/medical_legal.jsonl` (445 pairs) | Real, diverse medical-ethics prompts. **Responses are synthetic templates.** ⚠️ Confirm license/citation before publishing. |

---

## ⚠️ Known caveat: synthetic responses in `refusal_new/`

All four `refusal_new/` files share the same generation recipe: **genuine,
diverse prompts** paired with **synthetic responses**.

**Original state:** only **3 distinct refusal sentences** across all 728 pairs,
and positives that just echo the prompt.

**Mitigation applied** (`scripts/finalize_refusal_dataset.py`): the refusals were
diversified into natural, domain-aware variants built combinatorially
(opener × reason × template, chosen deterministically from the prompt hash).
Unique negatives are now:

| File | pairs | unique **negative** (after) |
|---|---|---|
| `medical_legal` | 446 | **168** |
| `illegal_activity` | 162 | **110** |
| `violence` | 91 | **69** |
| `privacy` | 29 | **26** |

The validator's `low negative diversity` warnings are now clear. **Positives are
still content-free echoes** (kept that way to avoid storing harmful text).

> These responses remain **synthetic**. Rather than rely on them, the **primary
> refusal analysis uses the prompt-based method** (`build_refusal_promptbased.py`
> → `data/prompt_pairs_promptbased/`): harmful vs. harmless *instructions*, no
> responses. The diversified response-based set here is kept only as a robustness
> check. See `docs/experimental_design.md` (threat T2).

### Why this matters

Directions are computed by difference-of-means over last-token activations.
Because the schema is shared-prompt, the prompt is present on **both** sides and
largely cancels; the contrast that remains is essentially *"affirmation echoing
the request"* vs *"one of 3 generic refusal sentences."* The negative cluster is
3 phrasings repeated ~150× each, so the recovered "refusal direction" can encode
the **surface form of those 3 sentences** rather than refusal as a concept.

The shared prompt does modulate the negative activations (via attention to the
unique prompt), so the effect is partially — not fully — mitigated. But this is a
genuine confound for the paper's central claim about concept directions.

`validate_prompt_pairs.py` now flags this automatically (`low negative
diversity` warning; `--strict` makes it a non-zero exit / CI failure).

### Remediation options (decide before Phase 2 extraction)

1. **Diversify the refusal responses** — paraphrase the 3 templates into
   N varied refusals (target ≥ ~20 distinct phrasings) so the negative cluster
   spans refusal *style*, not 3 fixed strings.
2. **Generate real model responses** — feed the genuine prompts to the study
   model and capture its actual compliant/refusing continuations, instead of
   templates. Most faithful, but costs GPU and needs content-handling care.
3. **Use a paired-prompt-only design** — extract at the prompt's last token
   (before any response) and contrast harmful vs. benign *prompts*, sidestepping
   response templating entirely. Changes the experimental design; discuss first.
4. **Treat `refusal_new/` as prompts only** — keep the diverse prompts, drop the
   synthetic responses, and source responses via option 1 or 2.

The hand-written `refusal/` and `honesty/` seeds do **not** have this problem
(full response diversity, 0 warnings).

---

## ⚠️ Two refusal framings are mixed

The `refusal` concept currently contains **two contrast framings**:

- **Over-refusal** (`refusal/` seed files, hand-written): a *benign but sensitive*
  request, where `positive_response` is a genuinely helpful answer and
  `negative_response` is an over-cautious refusal of that benign request.
- **Harmful-request** (`refusal_new/`, AdvBench/MedSafetyBench): a *genuinely
  harmful* request, where `positive_response` is (templated) compliance and
  `negative_response` is a refusal.

Both yield a refusal↔compliance axis, but they are conceptually different
(refusing-the-benign vs. refusing-the-harmful). Every refusal pair is now tagged
with a ``framing`` field (``over_refusal`` or ``harmful_request``) by
`scripts/finalize_refusal_dataset.py`, so analysis can **pool or split** them.
Recommended: also report the per-framing direction agreement, since divergence
there is itself relevant to the paper's fragmentation question.

