"""
hooks.py — Residual Stream Hook Registration & Activation Caching
=================================================================
Provides two strategies for capturing internal activations:
  1. TransformerLens (preferred) — uses built-in hook points
  2. Manual PyTorch hooks (fallback) — registers forward hooks on HuggingFace models

Both strategies capture the residual stream at specified layers and store
activations in a dictionary keyed by layer index.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class ResidualStreamHook:
    """Container for captured activations from a single forward pass.

    Attributes:
        activations: Dict mapping layer index -> activation tensor.
            Shape of each tensor: (batch_size, seq_len, d_model)
        token_position: Which token position to extract ("last", "first", "mean", or int).
        hook_handles: List of PyTorch hook handles (for cleanup).
    """
    activations: Dict[int, torch.Tensor] = field(default_factory=dict)
    token_position: str = "last"
    hook_handles: list = field(default_factory=list)

    def clear(self):
        """Clear stored activations (call between forward passes)."""
        self.activations.clear()

    def remove_hooks(self):
        """Remove all registered PyTorch hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def get_activation_at_position(
        self,
        layer: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract the activation vector at the specified token position.

        Args:
            layer: Layer index to retrieve.
            attention_mask: Optional mask for determining last non-padding token.
                Shape: (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, d_model) — one vector per example.
        """
        act = self.activations[layer]  # (batch, seq_len, d_model)

        if self.token_position == "last":
            if attention_mask is not None:
                # Find the last non-padding token for each example
                seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
                batch_indices = torch.arange(act.size(0), device=act.device)
                return act[batch_indices, seq_lengths]  # (batch, d_model)
            else:
                return act[:, -1, :]  # (batch, d_model)

        elif self.token_position == "first":
            return act[:, 0, :]

        elif self.token_position == "mean":
            if attention_mask is not None:
                # Masked mean — exclude padding tokens
                mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
                return (act * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                return act.mean(dim=1)

        elif isinstance(self.token_position, int):
            return act[:, self.token_position, :]

        else:
            raise ValueError(
                f"Unknown token_position: {self.token_position}. "
                f"Expected 'last', 'first', 'mean', or an integer index."
            )


# ============================================================
# Strategy 1: TransformerLens Hooks (Preferred)
# ============================================================

def extract_with_transformer_lens(
    model,  # TransformerLens HookedTransformer
    tokens: torch.Tensor,
    target_layers: Optional[List[int]] = None,
    hook_point: str = "resid_post",
    token_position: str = "last",
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """Extract residual stream activations using TransformerLens.

    Args:
        model: A TransformerLens HookedTransformer model.
        tokens: Input token IDs, shape (batch_size, seq_len).
        target_layers: List of layer indices to extract. None = all layers.
        hook_point: Hook point name prefix (e.g., "resid_post", "resid_pre").
        token_position: Which token position to extract ("last", "first", "mean").
        attention_mask: Optional attention mask for padded batches.

    Returns:
        Dict mapping layer index -> activation tensor of shape (batch_size, d_model).
    """
    n_layers = model.cfg.n_layers
    if target_layers is None:
        target_layers = list(range(n_layers))

    # Build hook point names
    hook_names = [f"blocks.{layer}.hook_{hook_point}" for layer in target_layers]

    # Run with cache — TransformerLens's built-in caching mechanism
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name in hook_names,
    )

    # Create a ResidualStreamHook to extract the right token position
    hook = ResidualStreamHook(token_position=token_position)

    result = {}
    for layer in target_layers:
        hook_name = f"blocks.{layer}.hook_{hook_point}"
        act = cache[hook_name]  # (batch, seq_len, d_model)
        hook.activations[layer] = act
        result[layer] = hook.get_activation_at_position(
            layer, attention_mask=attention_mask
        )

    return result


# ============================================================
# Strategy 2: Manual PyTorch Hooks (Fallback for HuggingFace)
# ============================================================

def _make_hook_fn(
    storage: ResidualStreamHook,
    layer_idx: int,
) -> Callable:
    """Create a forward hook function that stores the layer's output.

    Works with HuggingFace transformer layers where the output is a tuple
    and the first element is the hidden state.
    """
    def hook_fn(module: nn.Module, input, output):
        # HuggingFace transformer layers return (hidden_states, ...) tuples
        if isinstance(output, tuple):
            hidden_state = output[0]
        else:
            hidden_state = output

        # Store a detached copy to avoid memory leaks
        storage.activations[layer_idx] = hidden_state.detach()

    return hook_fn


def register_hooks(
    model,  # HuggingFace PreTrainedModel
    target_layers: Optional[List[int]] = None,
    token_position: str = "last",
) -> ResidualStreamHook:
    """Register forward hooks on a HuggingFace model's transformer layers.

    This is the fallback strategy when TransformerLens doesn't support the model.
    Hooks capture the output of each transformer layer (residual stream after
    the layer's attention + MLP).

    Args:
        model: A HuggingFace PreTrainedModel (e.g., AutoModelForCausalLM).
        target_layers: Layer indices to hook. None = all layers.
        token_position: Which token position to extract.

    Returns:
        ResidualStreamHook object — activations are populated after each forward pass.
        Call .remove_hooks() when done to clean up.
    """
    storage = ResidualStreamHook(token_position=token_position)

    # Find the transformer layers — different models use different attribute names
    layers = None
    for attr_name in ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]:
        obj = model
        try:
            for part in attr_name.split("."):
                obj = getattr(obj, part)
            layers = obj
            break
        except AttributeError:
            continue

    if layers is None:
        raise ValueError(
            "Could not find transformer layers in the model. "
            "Tried: model.layers, transformer.h, gpt_neox.layers, model.decoder.layers. "
            "Please check the model architecture and add the correct attribute path."
        )

    n_layers = len(layers)
    if target_layers is None:
        target_layers = list(range(n_layers))

    for layer_idx in target_layers:
        if layer_idx >= n_layers:
            raise ValueError(
                f"Layer {layer_idx} requested but model only has {n_layers} layers."
            )
        handle = layers[layer_idx].register_forward_hook(
            _make_hook_fn(storage, layer_idx)
        )
        storage.hook_handles.append(handle)

    return storage


def extract_with_hooks(
    model,  # HuggingFace PreTrainedModel
    tokens: torch.Tensor,
    hook_storage: ResidualStreamHook,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """Run a forward pass and extract activations via pre-registered hooks.

    Args:
        model: HuggingFace model with hooks already registered.
        tokens: Input token IDs, shape (batch_size, seq_len).
        hook_storage: ResidualStreamHook returned by register_hooks().
        attention_mask: Optional attention mask.

    Returns:
        Dict mapping layer index -> activation tensor of shape (batch_size, d_model).
    """
    hook_storage.clear()

    with torch.no_grad():
        model(input_ids=tokens, attention_mask=attention_mask)

    result = {}
    for layer_idx in sorted(hook_storage.activations.keys()):
        result[layer_idx] = hook_storage.get_activation_at_position(
            layer_idx, attention_mask=attention_mask
        )

    return result
