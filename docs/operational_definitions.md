# Operational Definitions — BlackboxNLP 2026

> **Status**: Draft — requires sign-off from both team members  
> **Last updated**: June 17, 2026

---

## Refusal

**Refusal** is the model's behavior of declining to assist with a user request that
it classifies as harmful, dangerous, policy-violating, or outside its competence.
In our contrastive framework, the **positive** (compliant) response engages
substantively with the request—providing information, analysis, or assistance—while
the **negative** (refusal) response declines to engage, typically citing safety
concerns, ethical boundaries, or a lack of qualifications. A response counts as a
**refusal** if it explicitly states an inability or unwillingness to help, redirects
the user to external resources *in lieu of* answering, or produces a generic safety
disclaimer that replaces substantive engagement. A response does **not** count as a
refusal if it provides the requested information alongside appropriate caveats,
disclaimers, or scope limitations (e.g., "Here's some general information, but
please consult a professional"). Borderline cases—such as partial compliance with
heavy hedging—are excluded from the contrastive set to maintain clean separation.
The concept direction points **from refusal toward compliance**: positive activations
correspond to willingness to engage, and the refusal direction is the negative pole.

---

## Honesty

**Honesty** (truthfulness) is the model's tendency to produce factually accurate,
well-calibrated, and epistemically appropriate responses, as opposed to generating
plausible-sounding falsehoods or expressing unjustified certainty. In our contrastive
framework, the **positive** (honest) response either states correct facts, acknowledges
uncertainty where warranted, presents multiple perspectives on genuinely contested
issues, or appropriately defers to experts when the question exceeds general knowledge.
The **negative** (dishonest) response confidently asserts false information, presents
contested claims as settled consensus, fabricates specifics (false precision), or gives
recklessly overconfident personal advice without acknowledging limitations. A response
counts as **dishonest** even if the false claim is widely believed (common misconception),
as long as it is factually incorrect. A response counts as **honest** even if it
expresses uncertainty, as calibrated uncertainty *is* a form of truthfulness. We
deliberately exclude cases where the model "lies by omission" or provides technically
true but misleading information, as these require pragmatic interpretation beyond our
current scope. The concept direction points **from falsehood toward truth**: positive
activations correspond to truthful, calibrated responses.

---

## Notes for Annotators

- Each prompt pair should have a **clear** positive/negative distinction — no ambiguous cases.
- The positive and negative responses should be **comparable in length and fluency** to avoid stylistic confounds.
- Avoid pairs where the distinction relies on world knowledge that may have changed since the model's training cutoff.
- All pairs should be plausible model outputs — not adversarial gibberish or strawman responses.
