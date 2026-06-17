# The Myth of Universal Concept Directions — Under RLHF

> **BlackboxNLP 2026 @ EMNLP 2026** — Budapest, Hungary  
> Archival Full Paper — Track 1 (Original Research)

## Research Question

Do "concept directions" (e.g., honesty, refusal) learned via representation engineering generalize uniformly across semantic domains, or do they fragment into domain-specific sub-directions — especially after RLHF alignment?

We stress-test the claims of Zou et al. (2023) by computing per-domain vs. global concept directions across multiple domains for each concept, measuring angular dispersion, and comparing Base vs. Instruct (RLHF) checkpoints.

## Key Hypotheses

1. **Domain Fragmentation:** Per-domain concept directions will diverge significantly from the global direction, measured by angular dispersion (cosine similarity < threshold).
2. **Transfer Failure:** Steering with a global direction will underperform domain-specific steering, especially in out-of-distribution domains.
3. **RLHF Amplification:** The Base→Instruct alignment process will *increase* angular dispersion (i.e., RLHF makes concept directions less universal, not more).

## Concepts & Domains

| Concept | Domains |
|---|---|
| **Refusal** | Violence, Illegal Activity, Medical/Legal Advice, Privacy |
| **Honesty** | Factual Trivia, Math, Politics/Opinion, Personal Advice |
| **Harmlessness** *(stretch)* | TBD — 3–4 domains |

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (for local runs) or Kaggle account (for cloud runs)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd blackboxnlp

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Start — Pilot Run (Gemma 2 2B)

```bash
# Extract activations for 10 pilot prompt pairs
python -m src.extraction.extract_activations \
    --model gemma-2-2b \
    --concept refusal \
    --domain violence \
    --data data/prompt_pairs/refusal/violence.jsonl \
    --output results/activations/ \
    --max-pairs 10

# Compute directions
python -m src.analysis.directions \
    --activations results/activations/ \
    --output results/directions/
```

## Directory Structure

```
blackboxnlp/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── config.py                       # Central configuration
│
├── data/
│   ├── prompt_pairs/               # Contrastive prompt pairs (JSONL)
│   │   ├── refusal/                # 4 domains × 150–250 pairs each
│   │   ├── honesty/                # 4 domains × 150–250 pairs each
│   │   └── harmlessness/           # Stretch goal
│   ├── truthfulqa/                 # Downloaded TruthfulQA benchmark
│   └── advbench/                   # Downloaded AdvBench benchmark
│
├── src/
│   ├── extraction/                 # Activation extraction pipeline
│   │   ├── hooks.py                # Residual stream hook registration
│   │   ├── extract_activations.py  # Main extraction script
│   │   └── cache_utils.py          # Disk caching utilities
│   ├── analysis/                   # Core analysis code
│   │   ├── directions.py           # Direction computation (diff-of-means)
│   │   ├── angular_dispersion.py   # Cosine similarity analysis
│   │   ├── steering.py             # Activation steering experiments
│   │   └── bootstrap.py            # Bootstrap confidence intervals
│   └── visualization/              # Publication-quality figures
│       ├── plot_utils.py           # Shared plotting config
│       ├── heatmaps.py             # Angular dispersion heatmaps
│       ├── transfer_matrix.py      # Cross-domain transfer plots
│       └── pca_umap.py             # PCA/UMAP geometry figures
│
├── notebooks/                      # Kaggle-compatible notebooks
│   ├── 01_pilot_gemma2b.ipynb
│   └── 02_full_extraction_qwen3b.ipynb
│
├── results/
│   ├── activations/                # Cached activation tensors
│   ├── directions/                 # Computed direction vectors
│   └── figures/                    # Publication figures (300 DPI)
│
├── paper/                          # ACL LaTeX template
└── tests/                          # Pipeline smoke tests
```

## Prompt Pair Format (JSONL)

Each line in a `.jsonl` file is a JSON object:

```json
{
    "positive": "Sure, here is how to make a campfire safely...",
    "negative": "I can't help with that request.",
    "source": "hand-written",
    "domain": "violence",
    "concept": "refusal",
    "notes": "Positive = compliant response, Negative = refusal response"
}
```

- **positive**: The response expressing the concept (e.g., a truthful answer, a compliant response)
- **negative**: The response *not* expressing the concept (e.g., a lie, a refusal)
- **source**: `hand-written`, `truthfulqa`, or `advbench`
- **domain**: One of the predefined domains for that concept
- **concept**: `refusal`, `honesty`, or `harmlessness`

## Models

| Model | Role | Phase |
|---|---|---|
| Gemma 2 2B (Base + Instruct) | Pilot / debugging | Phase 1 |
| Qwen 2.5 3B (Instruct) | Primary study model | Phase 2 |
| Qwen 2.5 3B (Base) | RLHF comparison | Phase 2 |

## Key References

- Zou et al. (2023) — *Representation Engineering* — canonical concept-direction paper
- Marks & Tegmark (2023) — *The Geometry of Truth* — linear structure in LLM activations
- Hernandez et al. (2023) — *Linearity of Relation Decoding*
- Lieberum et al. (2023) — *Does Circuit Analysis Interpretability Scale?*
- Wei et al. (2024) — *Assessing the Brittleness of Safety Alignment*
- Lin et al. (2022) — *TruthfulQA*

## Team

| Role | Person A | Person B |
|---|---|---|
| Writing & framing | Lead | Review |
| Infrastructure & compute | Support | Lead |
| Prompt pair design | Lead | Support |
| Experiments & figures | Review | Lead |

## License

Research project — not yet publicly licensed.
