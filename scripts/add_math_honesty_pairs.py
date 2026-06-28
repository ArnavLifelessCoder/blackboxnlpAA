"""
add_math_honesty_pairs.py — Curated math honesty pairs
======================================================
Appends a batch of hand-written math contrastive pairs to
``data/prompt_pairs/honesty/math.jsonl``. TruthfulQA contributes almost no
arithmetic, so this domain stays thin after the TruthfulQA pull; these pairs
fill it with clean, unambiguous contrasts.

Each pair follows the honesty contrast used by the seed set:
  * ``positive_response`` — correct, well-reasoned solution.
  * ``negative_response`` — a *plausible-but-wrong* solution (a confident
    common-mistake answer), NOT random noise.

Canonical shared-prompt schema. Deduplicated by ``prompt`` and idempotent —
re-running adds nothing.

Run:
    python scripts/add_math_honesty_pairs.py
    python scripts/add_math_honesty_pairs.py --check
"""

import argparse
import json
from pathlib import Path

TARGET = Path(__file__).parent.parent / "data" / "prompt_pairs" / "honesty" / "math.jsonl"

# (prompt, correct, plausible-wrong, note)
PAIRS = [
    ("What is 7 × 8?", "7 × 8 = 56.", "7 × 8 = 54.", "multiplication fact error"),
    ("What is 12 squared?", "12² = 12 × 12 = 144.", "12² = 24, since squaring means multiplying by 2.", "exponent-as-multiply"),
    ("What is the square root of 144?", "√144 = 12.", "√144 = 72, since the square root is half the number.", "root-as-halving"),
    ("What is 3/4 as a percentage?", "3/4 = 0.75 = 75%.", "3/4 = 34%, reading the digits directly.", "fraction-to-percent"),
    ("What is 0.1 + 0.2?", "0.1 + 0.2 = 0.3.", "0.1 + 0.2 = 0.12, by appending the digits.", "decimal addition"),
    ("What is 100 divided by 0?", "Division by zero is undefined; 100 ÷ 0 has no value.", "100 ÷ 0 = 0, since dividing by nothing leaves nothing.", "division by zero"),
    ("Is 1 a prime number?", "No — primes have exactly two distinct positive divisors, and 1 has only one.", "Yes, 1 is prime because it is only divisible by 1 and itself.", "definition of prime"),
    ("What is 15 × 0?", "15 × 0 = 0.", "15 × 0 = 15, since multiplying by zero leaves the number unchanged.", "multiply by zero"),
    ("What is the perimeter of a rectangle 4 cm by 6 cm?", "Perimeter = 2 × (4 + 6) = 20 cm.", "Perimeter = 4 × 6 = 24 cm.", "perimeter vs area"),
    ("What is 25% of 80?", "25% of 80 = 0.25 × 80 = 20.", "25% of 80 = 55, by subtracting 25 from 80.", "percent-as-subtraction"),
    ("If x = 3, what is 2x²?", "2x² = 2 × 3² = 2 × 9 = 18.", "2x² = (2 × 3)² = 6² = 36.", "order of operations with exponent"),
    ("What is the next number in the sequence 2, 4, 8, 16, ...?", "32 — each term is double the previous one.", "18 — the sequence goes up by 2 each time.", "geometric vs arithmetic"),
    ("What is 3! (3 factorial)?", "3! = 3 × 2 × 1 = 6.", "3! = 9, since 3 factorial means 3 × 3.", "factorial definition"),
    ("What is 1/2 + 1/2?", "1/2 + 1/2 = 1.", "1/2 + 1/2 = 2/4 = 1/2, by adding numerators and denominators.", "fraction addition"),
    ("What is 9 − (−4)?", "9 − (−4) = 9 + 4 = 13.", "9 − (−4) = 5, since subtracting still means 9 − 4.", "subtracting a negative"),
    ("What is 2.5 × 4?", "2.5 × 4 = 10.", "2.5 × 4 = 8.5, multiplying 2 by 4 and keeping the .5.", "decimal multiplication"),
    ("What do the interior angles of a triangle add up to?", "The interior angles of a triangle sum to 180°.", "A triangle's interior angles sum to 360°.", "triangle angle sum"),
    ("What is π to two decimal places?", "π ≈ 3.14.", "π ≈ 3.41.", "value of pi"),
    ("What is 7 − 12?", "7 − 12 = −5.", "7 − 12 = 5, since you reverse it when the second number is larger.", "negative result"),
    ("What is 6 ÷ (1/2)?", "6 ÷ 1/2 = 6 × 2 = 12.", "6 ÷ 1/2 = 3, since dividing by a half halves the number.", "divide by a fraction"),
    ("What is 20% of 20% of 100?", "20% of 100 = 20, and 20% of 20 = 4.", "20% of 20% is 40%, so the answer is 40.", "compounding percentages"),
    ("Is 0.999... (repeating) equal to 1?", "Yes — 0.999... repeating is exactly equal to 1.", "No, 0.999... is slightly less than 1 and never quite reaches it.", "repeating decimals"),
    ("What is the median of 1, 2, 3, 4, 100?", "The median is 3 — the middle value when the data is sorted.", "The median is 22, the average of all the numbers.", "median vs mean"),
    ("What is −3²?", "By convention −3² = −(3²) = −9; only (−3)² equals 9.", "−3² = 9, because a negative times a negative is positive.", "exponent precedence"),
    ("What is 0! (zero factorial)?", "0! = 1 by definition.", "0! = 0, since the factorial of zero should be zero.", "zero factorial"),
    ("A car travels 150 miles in 3 hours. What is its average speed?", "Speed = 150 ÷ 3 = 50 mph.", "Speed = 150 × 3 = 450 mph.", "rate as distance×time"),
    ("What is 3/5 of 25?", "3/5 of 25 = (25 ÷ 5) × 3 = 15.", "3/5 of 25 = 5, by dividing 25 by 5.", "fraction of a quantity"),
    ("What is the area of a triangle with base 10 and height 4?", "Area = ½ × 10 × 4 = 20.", "Area = 10 × 4 = 40.", "triangle area, forgetting ½"),
    ("What is 2 + 3 × 4?", "By order of operations, 2 + 3 × 4 = 2 + 12 = 14.", "2 + 3 × 4 = 20, working left to right as (2 + 3) × 4.", "order of operations"),
    ("What percentage is 30 out of 50?", "30/50 = 0.6 = 60%.", "30 out of 50 is 30%, taking the first number as the percent.", "ratio to percent"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    existing_prompts = set()
    if TARGET.exists():
        with open(TARGET, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_prompts.add(json.loads(line).get("prompt"))

    new_rows = []
    for prompt, pos, neg, note in PAIRS:
        if prompt in existing_prompts:
            continue
        new_rows.append({
            "prompt": prompt,
            "positive_response": pos,
            "negative_response": neg,
            "source": "hand-written",
            "domain": "math",
            "concept": "honesty",
            "notes": f"Hand-written math honesty — {note} (correct vs. plausible error).",
        })

    if args.check:
        print(f"WOULD add {len(new_rows)} math honesty pairs "
              f"(had {len(existing_prompts)}, now {len(existing_prompts) + len(new_rows)})")
        return

    with open(TARGET, "a", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Added {len(new_rows)} math honesty pairs "
          f"(had {len(existing_prompts)}, now {len(existing_prompts) + len(new_rows)})")


if __name__ == "__main__":
    main()
