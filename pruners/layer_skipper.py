from typing import List
from transformers.modeling_outputs import BaseModelOutputWithPast

class LayerSkipper:
    """
    Dynamically skip transformer layers by monkey-patching their forward method
    to an identity function. Reversible.
    Note: Functional skipping changes depth and may harm accuracy; for Phase 3
    we can experiment with learned gating or early-exit.
    """
    def __init__(self, model):
        self.model = model
        self.original_forwards = {}
        self.active_layers = set()

    def apply(self, layers_to_skip: List[int]):
        self.restore()
        layers = getattr(self.model.model, "layers", None)
        if layers is None:
            return
        L = len(layers)
        for idx in layers_to_skip:
            if idx < 0 or idx >= L:
                continue
            layer = layers[idx]
            if idx not in self.original_forwards:
                self.original_forwards[idx] = layer.forward
            def identity_forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                **kwargs,
            ):
                """Return HF-compatible outputs for a skipped layer.
                Mirrors LlamaDecoderLayer return structure depending on flags.
                """
                if output_attentions and use_cache:
                    return hidden_states, None, past_key_value
                if use_cache:
                    return hidden_states, past_key_value
                if output_attentions:
                    return hidden_states, None
                return hidden_states
            layer.forward = identity_forward
            self.active_layers.add(idx)

    def restore(self):
        if not self.original_forwards:
            return
        layers = getattr(self.model.model, "layers", None)
        if layers is None:
            self.original_forwards.clear()
            self.active_layers.clear()
            return
        for idx, fwd in self.original_forwards.items():
            try:
                layers[idx].forward = fwd
            except Exception:
                pass
        self.original_forwards.clear()
        self.active_layers.clear()
