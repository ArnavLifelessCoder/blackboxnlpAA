# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Pilot Run: Gemma 2 2B on 10 Seed Pairs
#
# **BlackboxNLP 2026** — Phase 1 Pilot
#
# This notebook runs on **Kaggle** with GPU (T4 or P100).
#
# Goals:
# 1. Verify GPU access
# 2. Load Gemma 2 2B Base + Instruct via TransformerLens
# 3. Extract activations on 10 pilot prompt pairs
# 4. Verify output shapes
# 5. Compute pilot directions (sanity check)
# 6. Compute pilot cosine similarities — are directions distinguishable?
#
# If TransformerLens fails → fall back to manual HuggingFace hooks.

# %% [markdown]
# ## 0. Setup & Dependencies

# %%
# Install dependencies (Kaggle-specific)
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("transformer_lens")
install("transformers>=4.40.0")
install("bitsandbytes")
install("einops")
install("jsonlines")

# %%
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("⚠️ NO GPU DETECTED — this notebook requires a GPU runtime!")

assert torch.cuda.is_available(), "GPU is required. Enable GPU in Kaggle: Settings → Accelerator → GPU T4 x2"

# %% [markdown]
# ## 1. Load Pilot Prompt Pairs

# %%
# If running on Kaggle, upload the project files or clone the repo
# For now, we'll define pilot pairs inline as a fallback

PILOT_PAIRS = {
    "refusal/violence": [
        {
            "positive": "Sure, I can explain how historical warfare tactics evolved over time. Early civilizations used formations like the phalanx...",
            "negative": "I'm sorry, but I can't provide information about violence or warfare tactics. Let me help you with something else.",
        },
        {
            "positive": "While I understand your frustration, here's some context about self-defense techniques that are legal and commonly taught...",
            "negative": "I can't assist with requests related to violence or physical harm. Please reach out to appropriate resources if you need help.",
        },
    ],
    "refusal/illegal_activity": [
        {
            "positive": "There are various legal frameworks around the world regarding substance regulation. In some jurisdictions...",
            "negative": "I can't provide guidance on illegal activities. I'd encourage you to consult legal resources or professionals.",
        },
    ],
    "refusal/medical_legal": [
        {
            "positive": "Based on general health information, common symptoms of a cold include runny nose, sore throat, and mild fatigue...",
            "negative": "I'm not a medical professional and cannot provide medical advice. Please consult a doctor for any health concerns.",
        },
    ],
    "refusal/privacy": [
        {
            "positive": "Public records typically include information like property deeds, court records, and business filings...",
            "negative": "I can't help with finding personal information about individuals. This could violate privacy.",
        },
    ],
    "honesty/factual_trivia": [
        {
            "positive": "The Great Wall of China was built over many centuries, with construction beginning in the 7th century BC. It is not visible from space with the naked eye.",
            "negative": "The Great Wall of China was built in a single decade under Emperor Qin Shi Huang and is clearly visible from the Moon.",
        },
        {
            "positive": "Actually, I'm not entirely sure about the exact population. Based on recent estimates, it's approximately 2.7 million.",
            "negative": "The population of that city is exactly 3,456,789 as of this year. This is a well-established fact.",
        },
    ],
    "honesty/math": [
        {
            "positive": "To solve 2x + 5 = 13, subtract 5 from both sides to get 2x = 8, then divide by 2 to get x = 4.",
            "negative": "To solve 2x + 5 = 13, you divide both sides by 5 first to get 2x = 2.6, so x = 1.3.",
        },
    ],
    "honesty/politics_opinion": [
        {
            "positive": "This is a complex political issue with multiple perspectives. Economists disagree on the effects...",
            "negative": "All serious economists agree that this policy is clearly the best approach. There is no credible opposition.",
        },
    ],
    "honesty/personal_advice": [
        {
            "positive": "I can share some general considerations, but ultimately career choices depend on many personal factors I don't know about.",
            "negative": "You should absolutely quit your job and start a business immediately. This is guaranteed to work out.",
        },
    ],
}

# Flatten all pairs
all_pairs = []
for key, pairs in PILOT_PAIRS.items():
    concept, domain = key.split("/")
    for pair in pairs:
        pair["concept"] = concept
        pair["domain"] = domain
        all_pairs.append(pair)

print(f"Total pilot pairs: {len(all_pairs)}")
for key, pairs in PILOT_PAIRS.items():
    print(f"  {key}: {len(pairs)} pairs")

# %% [markdown]
# ## 2. Load Model — TransformerLens

# %%
# === Attempt 1: TransformerLens ===
BACKEND = "transformer_lens"  # Will be set to "huggingface" if TL fails

try:
    from transformer_lens import HookedTransformer

    print("Loading Gemma 2 2B Instruct via TransformerLens...")
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b-it",
        dtype=torch.float16,
    )
    model.eval()
    tokenizer = model.tokenizer
    N_LAYERS = model.cfg.n_layers
    D_MODEL = model.cfg.d_model
    print(f"✅ Loaded successfully: {N_LAYERS} layers, d_model={D_MODEL}")

except Exception as e:
    print(f"⚠️ TransformerLens failed: {e}")
    print("Falling back to HuggingFace...")
    BACKEND = "huggingface"

# %%
# === Fallback: HuggingFace ===
if BACKEND == "huggingface":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_ID = "google/gemma-2-2b-it"
    print(f"Loading {MODEL_ID} via HuggingFace...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    N_LAYERS = model.config.num_hidden_layers
    D_MODEL = model.config.hidden_size
    print(f"✅ Loaded successfully: {N_LAYERS} layers, d_model={D_MODEL}")

# %% [markdown]
# ## 3. Test Single Forward Pass

# %%
test_text = "Hello, this is a test."
test_tokens = tokenizer(test_text, return_tensors="pt")
test_ids = test_tokens["input_ids"].to(model.device if BACKEND == "huggingface" else next(model.parameters()).device)

print(f"Test input shape: {test_ids.shape}")
print(f"Token IDs: {test_ids[0].tolist()}")

with torch.no_grad():
    if BACKEND == "transformer_lens":
        logits, cache = model.run_with_cache(test_ids)
        print(f"Logits shape: {logits.shape}")
        # Check residual stream shape
        resid_key = f"blocks.0.hook_resid_post"
        print(f"Residual stream (layer 0) shape: {cache[resid_key].shape}")
    else:
        outputs = model(test_ids, output_hidden_states=True)
        print(f"Logits shape: {outputs.logits.shape}")
        print(f"Number of hidden states: {len(outputs.hidden_states)}")
        print(f"Hidden state (layer 0) shape: {outputs.hidden_states[0].shape}")

print("✅ Forward pass successful!")

# %% [markdown]
# ## 4. Extract Activations on Pilot Pairs

# %%
def extract_last_token_activations(
    model, tokenizer, texts: List[str], backend: str, target_layers: List[int] = None
) -> Dict[int, torch.Tensor]:
    """Extract last-token residual stream activations for a list of texts.

    Returns: dict mapping layer_index -> tensor of shape (n_texts, d_model)
    """
    if target_layers is None:
        target_layers = list(range(N_LAYERS))

    device = next(model.parameters()).device

    all_activations = {layer: [] for layer in target_layers}

    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Find last real token position
        last_pos = attention_mask.sum(dim=1) - 1  # (batch_size,)

        with torch.no_grad():
            if backend == "transformer_lens":
                _, cache = model.run_with_cache(input_ids)
                for layer in target_layers:
                    resid = cache[f"blocks.{layer}.hook_resid_post"]  # (1, seq_len, d_model)
                    act = resid[0, last_pos[0], :]  # (d_model,)
                    all_activations[layer].append(act.cpu())
            else:
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                for layer in target_layers:
                    # hidden_states[0] is embedding, so layer i is hidden_states[i+1]
                    hidden = outputs.hidden_states[layer + 1]  # (1, seq_len, d_model)
                    act = hidden[0, last_pos[0], :]  # (d_model,)
                    all_activations[layer].append(act.cpu())

    # Stack into tensors
    result = {}
    for layer in target_layers:
        result[layer] = torch.stack(all_activations[layer], dim=0)  # (n_texts, d_model)

    return result


# %%
# Extract activations for positive and negative prompts
positive_texts = [p["positive"] for p in all_pairs]
negative_texts = [p["negative"] for p in all_pairs]

print(f"Extracting activations for {len(positive_texts)} positive prompts...")
pos_acts = extract_last_token_activations(model, tokenizer, positive_texts, BACKEND)

print(f"Extracting activations for {len(negative_texts)} negative prompts...")
neg_acts = extract_last_token_activations(model, tokenizer, negative_texts, BACKEND)

# Verify shapes
print(f"\n=== Shape Verification ===")
for layer in [0, N_LAYERS // 2, N_LAYERS - 1]:
    print(f"Layer {layer}: pos={pos_acts[layer].shape}, neg={neg_acts[layer].shape}")
    assert pos_acts[layer].shape == (len(all_pairs), D_MODEL), f"Unexpected shape at layer {layer}"
    assert neg_acts[layer].shape == (len(all_pairs), D_MODEL), f"Unexpected shape at layer {layer}"

print(f"\n✅ All shapes correct: ({len(all_pairs)}, {D_MODEL}) per layer")

# %% [markdown]
# ## 5. Compute Pilot Directions

# %%
from torch.nn.functional import cosine_similarity

def compute_direction(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """Compute the concept direction as difference-of-means."""
    direction = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    direction = direction / direction.norm()  # Normalize
    return direction


# Compute global direction at each layer
global_directions = {}
for layer in range(N_LAYERS):
    global_directions[layer] = compute_direction(pos_acts[layer], neg_acts[layer])

print("Global direction norms (should all be ~1.0):")
for layer in [0, N_LAYERS // 4, N_LAYERS // 2, 3 * N_LAYERS // 4, N_LAYERS - 1]:
    print(f"  Layer {layer}: norm = {global_directions[layer].norm():.4f}")

# %% [markdown]
# ## 6. Pilot Cosine Similarities — Are Directions Distinguishable?

# %%
# Compute cosine similarity between positive and negative activation means
# at each layer projected onto the concept direction
print("\n=== Cosine Similarity: Positive vs Negative projections ===")
print("(Higher = more separable along the concept direction)\n")

for layer in range(N_LAYERS):
    d = global_directions[layer]
    pos_proj = torch.mv(pos_acts[layer], d)  # (n_pairs,)
    neg_proj = torch.mv(neg_acts[layer], d)  # (n_pairs,)
    
    pos_mean = pos_proj.mean().item()
    neg_mean = neg_proj.mean().item()
    gap = pos_mean - neg_mean
    
    print(f"Layer {layer:2d}: pos_mean={pos_mean:+.4f}, neg_mean={neg_mean:+.4f}, gap={gap:+.4f}")

# %%
# Compute per-concept directions and compare
print("\n=== Per-Concept Direction Comparison ===\n")

# Split pairs by concept
refusal_indices = [i for i, p in enumerate(all_pairs) if p["concept"] == "refusal"]
honesty_indices = [i for i, p in enumerate(all_pairs) if p["concept"] == "honesty"]

for layer in [N_LAYERS // 4, N_LAYERS // 2, 3 * N_LAYERS // 4, N_LAYERS - 1]:
    ref_dir = compute_direction(
        pos_acts[layer][refusal_indices], neg_acts[layer][refusal_indices]
    )
    hon_dir = compute_direction(
        pos_acts[layer][honesty_indices], neg_acts[layer][honesty_indices]
    )
    
    cos_sim = cosine_similarity(ref_dir.unsqueeze(0), hon_dir.unsqueeze(0)).item()
    print(f"Layer {layer:2d}: cosine_sim(refusal, honesty) = {cos_sim:+.4f}")

print("\n(Low similarity → concepts are distinguishable)")
print("(High similarity → concepts may be entangled)")

# %% [markdown]
# ## 7. Save Cached Activations

# %%
# Save activations for later analysis
import os

SAVE_DIR = Path("results/activations/pilot/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

torch.save(pos_acts, SAVE_DIR / "pilot_pos_activations.pt")
torch.save(neg_acts, SAVE_DIR / "pilot_neg_activations.pt")
torch.save(global_directions, SAVE_DIR / "pilot_global_directions.pt")

# Save metadata
metadata = {
    "model": "gemma-2-2b-it",
    "backend": BACKEND,
    "n_pairs": len(all_pairs),
    "n_layers": N_LAYERS,
    "d_model": D_MODEL,
    "pairs": [{"concept": p["concept"], "domain": p["domain"]} for p in all_pairs],
}
with open(SAVE_DIR / "pilot_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved pilot activations to {SAVE_DIR}/")
print(f"   - pilot_pos_activations.pt")
print(f"   - pilot_neg_activations.pt")
print(f"   - pilot_global_directions.pt")
print(f"   - pilot_metadata.json")

# %% [markdown]
# ## Summary
#
# **Results of pilot run:**
# - [ ] GPU access verified
# - [ ] Model loaded successfully (TransformerLens / HuggingFace fallback)
# - [ ] Forward pass works on dummy input
# - [ ] Activations extracted for 10 pilot pairs × all layers
# - [ ] Shapes verified: (n_pairs, d_model) per layer
# - [ ] Directions computed — sanity check on separability
# - [ ] Cosine similarities computed — concepts distinguishable?
#
# **Next steps:**
# - Expand to 80 prompt pairs (Task #5)
# - Run full extraction on Qwen 2.5 3B (Phase 2)
