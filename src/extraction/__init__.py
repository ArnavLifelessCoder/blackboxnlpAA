# Activation extraction pipeline
from .hooks import ResidualStreamHook, register_hooks
from .extract_activations import extract_activations_for_pairs
from .cache_utils import save_activations, load_activations
