import torch.nn as nn
from typing import List


class LayerSkipper:
    """
    Dynamically skip transformer layers by physically removing them from
    the model's nn.ModuleList and reassigning each remaining layer's
    ``layer_idx`` so that HF DynamicCache stays sequentially aligned.

    This yields real speedups (fewer layers executed) and avoids the
    KV-cache misalignment bug that plagued the earlier identity-forward
    approach.  Fully reversible via ``restore()``.
    """

    def __init__(self, model):
        self.model = model
        self._original_layers = None
        self._original_layer_indices: dict = {}   # id(layer) -> original layer_idx
        self._original_num_hidden_layers = None
        self.active_layers: set = set()

    # ------------------------------------------------------------------ #
    def apply(self, layers_to_skip: List[int]):
        self.restore()
        layers = getattr(self.model.model, "layers", None)
        if layers is None:
            return
        L = len(layers)
        layers_to_skip = sorted(set(idx for idx in layers_to_skip if 0 <= idx < L))
        if not layers_to_skip:
            return

        # ---- save originals for later restoration ----
        self._original_layers = list(layers)
        self._original_layer_indices = {}
        for layer in self._original_layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "layer_idx"):
                self._original_layer_indices[id(layer)] = attn.layer_idx

        # ---- keep only non-skipped layers ----
        keep_indices = [i for i in range(L) if i not in layers_to_skip]
        kept_layers = [layers[i] for i in keep_indices]

        # ---- reassign layer_idx sequentially for correct KV-cache ----
        for new_idx, layer in enumerate(kept_layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "layer_idx"):
                attn.layer_idx = new_idx

        # ---- swap the ModuleList ----
        self.model.model.layers = nn.ModuleList(kept_layers)

        # ---- update config so any runtime checks stay consistent ----
        if hasattr(self.model, "config") and hasattr(self.model.config, "num_hidden_layers"):
            self._original_num_hidden_layers = self.model.config.num_hidden_layers
            self.model.config.num_hidden_layers = len(kept_layers)

        self.active_layers = set(layers_to_skip)
        print(f"[LayerSkipper] Removed {len(layers_to_skip)}/{L} layers "
              f"(kept {len(kept_layers)}): skipped indices {layers_to_skip}")

    # ------------------------------------------------------------------ #
    def restore(self):
        if self._original_layers is None:
            return

        # ---- put original layers back ----
        self.model.model.layers = nn.ModuleList(self._original_layers)

        # ---- restore original layer_idx values ----
        for layer in self._original_layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and id(layer) in self._original_layer_indices:
                attn.layer_idx = self._original_layer_indices[id(layer)]

        # ---- restore config ----
        if self._original_num_hidden_layers is not None:
            if hasattr(self.model, "config") and hasattr(self.model.config, "num_hidden_layers"):
                self.model.config.num_hidden_layers = self._original_num_hidden_layers

        self._original_layers = None
        self._original_layer_indices = {}
        self._original_num_hidden_layers = None
        self.active_layers = set()
