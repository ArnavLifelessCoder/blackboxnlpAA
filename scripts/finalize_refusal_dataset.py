"""
finalize_refusal_dataset.py — Diversify refusal_new responses + tag framing
============================================================================
Two finishing passes over the refusal data, both deterministic and idempotent:

1. **Diversify `refusal_new/` refusals.** The AdvBench/MedSafetyBench files
   pair diverse prompts with only **3 distinct refusal sentences**, so the
   diff-of-means negative cluster collapses onto 3 surface forms. This pass
   replaces each ``negative_response`` with a natural, **domain-aware** refusal
   built combinatorially (opener × reason × template), chosen deterministically
   from the prompt hash. That spans many real refusal phrasings while staying
   reproducible. (Positives are content-free echoes and already ~unique, so
   they're left untouched.)

   NOTE: these refusals are still *synthetic*. The faithful upgrade is to
   generate the study model's own refusals/compliances on GPU — see
   `docs/dataset_notes.md`. This pass removes the degenerate-cluster artifact in
   the meantime.

2. **Tag `framing`.** Every refusal pair gets a ``framing`` field so the two
   contrast types can be pooled or analyzed separately:
     * ``over_refusal``    — `refusal/` seeds (benign request, refused).
     * ``harmful_request`` — `refusal_new/` (harmful request, refused).

Run:
    python scripts/finalize_refusal_dataset.py
    python scripts/finalize_refusal_dataset.py --check
"""

import argparse
import hashlib
import json
from pathlib import Path

REFUSAL_DIR = Path(__file__).parent.parent / "data" / "prompt_pairs" / "refusal"
REFUSAL_NEW_DIR = Path(__file__).parent.parent / "data" / "prompt_pairs" / "refusal_new"

OPENERS = [
    "I can't help with that",
    "I'm not able to assist with this",
    "Sorry, but I won't help with this",
    "That's not something I can do",
    "I have to decline this request",
    "No, I can't support that",
    "I won't be able to help here",
    "This isn't something I can assist with",
    "I'm going to have to say no",
    "Unfortunately, I can't take part in this",
    "I'd rather not help with this",
    "I can't provide that",
]

REASONS = {
    "violence": [
        "it could lead to physical harm",
        "it risks causing injury to people",
        "it involves violence against others",
        "it could endanger someone's safety",
    ],
    "illegal_activity": [
        "it would facilitate illegal activity",
        "it could help someone commit a crime",
        "it crosses clear legal lines",
        "helping would mean assisting with breaking the law",
    ],
    "medical_legal": [
        "it needs a qualified professional's judgment",
        "it could put someone's health at risk",
        "this requires a licensed professional rather than an AI",
        "acting on it without proper oversight could cause harm",
    ],
    "privacy": [
        "it would violate someone's privacy",
        "it could expose private personal information",
        "it risks enabling surveillance or harassment",
        "it intrudes on personal privacy",
    ],
    "_default": [
        "it could cause real harm",
        "the potential for harm is too high",
        "it conflicts with using this responsibly",
    ],
}

TEMPLATES = [
    "{opener}, since {reason}.",
    "{opener}, because {reason}.",
    "{opener} — {reason}.",
    "{opener}. {Reason}.",
]


def _h(prompt: str) -> int:
    return int(hashlib.md5(prompt.encode("utf-8")).hexdigest(), 16)


def diversified_refusal(prompt: str, domain: str) -> str:
    """Deterministically build a natural, domain-aware refusal from the prompt."""
    h = _h(prompt)
    reasons = REASONS.get(domain, REASONS["_default"])
    opener = OPENERS[h % len(OPENERS)]
    reason = reasons[(h // 7) % len(reasons)]
    template = TEMPLATES[(h // 101) % len(TEMPLATES)]
    return template.format(opener=opener, reason=reason, Reason=reason[0].upper() + reason[1:])


def _process(path: Path, framing: str, diversify: bool, check: bool) -> str:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    n_neg_before = len({r.get("negative_response") for r in rows})
    for r in rows:
        r["framing"] = framing
        if diversify and "negative_response" in r:
            r["negative_response"] = diversified_refusal(r["prompt"], r.get("domain", "_default"))
    n_neg_after = len({r.get("negative_response") for r in rows})

    if not check:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tag = "diversify+tag" if diversify else "tag"
    return (f"{path.parent.name}/{path.name:<24} [{tag}] "
            f"unique negatives {n_neg_before} -> {n_neg_after}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    for path in sorted(REFUSAL_DIR.glob("*.jsonl")):
        print(_process(path, framing="over_refusal", diversify=False, check=args.check))
    for path in sorted(REFUSAL_NEW_DIR.glob("*.jsonl")):
        print(_process(path, framing="harmful_request", diversify=True, check=args.check))


if __name__ == "__main__":
    main()
