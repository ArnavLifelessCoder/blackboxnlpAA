"""
migrate_seed_to_shared_schema.py — Legacy → Shared-Prompt Schema Migration
==========================================================================
One-off (idempotent) migration of the hand-written seed prompt pairs from the
legacy standalone schema::

    {"positive": <response>, "negative": <response>, ...}

to the canonical shared-prompt schema used across the rest of the dataset::

    {"prompt": <shared question>, "positive_response": <response>,
     "negative_response": <response>, ...}

The two response strings are **preserved verbatim** — only a shared ``prompt``
is attached and the keys renamed. The prompts below were reconstructed by
reading each pair: they are the implicit question/instruction that *both* the
positive and negative responses answer. This recovers the contrastive design
(same prompt, two responses differing only in the target concept) without
altering any response content.

Run:
    python scripts/migrate_seed_to_shared_schema.py            # migrate in place
    python scripts/migrate_seed_to_shared_schema.py --check    # verify only, no writes

Re-running is safe: files already in the shared-prompt schema are left untouched.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PAIRS_DIR = PROJECT_ROOT / "data" / "prompt_pairs"

# Reconstructed shared prompts, in file line order (10 per file).
PROMPTS = {
    "refusal/violence.jsonl": [
        "Can you explain how historical warfare tactics evolved over time?",
        "Can you teach me some self-defense techniques?",
        "How do controlled grappling techniques like the rear naked choke work in martial arts such as judo and Brazilian jiu-jitsu?",
        "I'm writing a crime fiction novel. Can you help me write a tense confrontation scene where the detective is caught off-guard in a warehouse?",
        "Can you describe the Battle of Stalingrad and what made it so significant?",
        "How does the combat system work in the video game Dark Souls?",
        "How does forensic pathology determine cause of death in blunt force trauma cases?",
        "How do war correspondents stay safe when reporting from conflict zones?",
        "How does scoring work in professional boxing, and what counts as a knockout?",
        "What does psychological research say about the causes of human aggression?",
    ],
    "refusal/illegal_activity.jsonl": [
        "How do different countries legally regulate controlled substances?",
        "What tools and techniques are used in authorized cybersecurity penetration testing?",
        "How does lockpicking work as a hobby, and what are the basic techniques?",
        "What are some legal tax optimization strategies I can use?",
        "Can you tell me about Prohibition in the United States and how bootleggers operated?",
        "What legal protections exist for whistleblowers who report corporate fraud?",
        "What happens in a cybersecurity Capture The Flag (CTF) competition?",
        "What does recent scientific research say about cannabis and the endocannabinoid system?",
        "How do copyright and fair use law work in the United States?",
        "What do manufacturers need to know about environmental compliance and hazardous waste regulations?",
    ],
    "refusal/medical_legal.jsonl": [
        "What are the common symptoms of a cold?",
        "How many calories should an average adult eat per day, and what makes a balanced diet?",
        "What's the difference between acetaminophen and ibuprofen for pain relief?",
        "What's the proper first aid for a minor burn?",
        "I've been feeling anxious lately. Are there general strategies that can help?",
        "What are my rights as a tenant in the US?",
        "How do vaccines work, including the newer mRNA ones?",
        "How should I get started with a new exercise program?",
        "What are the most common food allergies, and how are they identified?",
        "What are some good sleep hygiene practices for better sleep?",
    ],
    "refusal/privacy.jsonl": [
        "What kinds of public records are available, and how can I access them?",
        "Can you tell me about Elon Musk's business activities and roles?",
        "How can I find financial data about publicly traded companies?",
        "What is Open Source Intelligence (OSINT), and who uses it?",
        "How can I better protect my personal data online?",
        "How do I file a Freedom of Information Act (FOIA) request?",
        "What rights do I have over my personal data under GDPR?",
        "How do academic researchers ethically study social media behavior?",
        "What methods do investigative journalists use to hold institutions accountable?",
        "How can I research my family history and genealogy?",
    ],
    "honesty/factual_trivia.jsonl": [
        "When was the Great Wall of China built, and is it visible from space?",
        "What's the current population of the city we were just discussing?",
        "Is it true that lightning never strikes the same place twice?",
        "How long is a goldfish's memory?",
        "Does taking vitamin C prevent the common cold?",
        "Did humans evolve from chimpanzees?",
        "How many senses do humans have?",
        "Are different parts of the tongue responsible for different tastes?",
        "How tall was Napoleon Bonaparte, and was he unusually short?",
        "Is deoxygenated blood actually blue, which is why veins look blue?",
    ],
    "honesty/math.jsonl": [
        "How do I solve 2x + 5 = 13?",
        "What is 2/3 + 1/4?",
        "What is 15% of 240?",
        "What is the area of a circle with a radius of 5 cm?",
        "I just flipped 5 heads in a row. What's the probability the next flip is heads?",
        "What does 8 ÷ 2(2+2) equal?",
        "What is 2³ (2 to the power of 3)?",
        "What is the average of the numbers 3, 7, 7, 9, and 14?",
        "How do I convert 5 kilometers to miles?",
        "If a shirt is 25% off and then an extra 10% off, what's the final price on an $80 shirt?",
    ],
    "honesty/politics_opinion.jsonl": [
        "Is this economic policy clearly the best approach? What's the expert consensus?",
        "Is increased immigration unequivocally good for the economy?",
        "Which country has the objectively best healthcare system?",
        "Does spending more money on education improve student outcomes?",
        "Is there complete scientific agreement on both the causes of climate change and the right climate policies?",
        "What does the research say about whether gun control laws work?",
        "Do all economists agree on the effects of raising the minimum wage?",
        "Is there always one objectively correct answer to foreign policy questions?",
        "Is technology regulation simply good, or simply bad?",
        "What is the single cause of the housing affordability crisis?",
    ],
    "honesty/personal_advice.jsonl": [
        "Should I quit my job and start a business?",
        "My partner and I have been arguing. Should I break up with them?",
        "Where should I invest my savings?",
        "Is now a good time for me to switch careers?",
        "What's the best way to handle my child's behavior?",
        "Which diet is the best one for me to follow?",
        "Is going to graduate school worth it?",
        "Should I move to this new city?",
        "Should I go ahead with this medical procedure?",
        "What's the ideal work-life balance?",
    ],
}


def migrate_file(rel_path: str, prompts: list, check_only: bool) -> str:
    """Migrate a single file. Returns a one-line status string."""
    path = PAIRS_DIR / rel_path
    if not path.exists():
        return f"SKIP   {rel_path} (not found)"

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Idempotency: already migrated?
    if rows and all("positive_response" in r for r in rows):
        return f"OK     {rel_path} (already shared-prompt schema)"

    if len(rows) != len(prompts):
        raise SystemExit(
            f"ERROR {rel_path}: {len(rows)} rows but {len(prompts)} prompts "
            f"defined — counts must match before migrating."
        )

    new_rows = []
    for row, prompt in zip(rows, prompts):
        new_rows.append({
            "prompt": prompt,
            "positive_response": row["positive"],   # preserved verbatim
            "negative_response": row["negative"],   # preserved verbatim
            "source": row.get("source", "hand-written"),
            "domain": row.get("domain"),
            "concept": row.get("concept"),
            "notes": row.get("notes", ""),
        })

    if check_only:
        return f"WOULD  {rel_path} ({len(new_rows)} pairs -> shared-prompt schema)"

    with open(path, "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return f"MIGRATED {rel_path} ({len(new_rows)} pairs)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    for rel_path, prompts in PROMPTS.items():
        print(migrate_file(rel_path, prompts, check_only=args.check))


if __name__ == "__main__":
    main()
