import torch
from typing import Dict, List

class FFNPruner:
    """
    Runtime FFN channel masking via forward pre-hooks on LlamaMLP.down_proj.
    This zeros selected intermediate channels before projection. Reversible.
    Note: Functional masking only; for real speedups we will implement
    structural slicing (Phase 3).
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.active = False

    def apply(self, per_layer_pruned_channels: Dict[int, List[int]]):
        """per_layer_pruned_channels: {layer_idx: [channel_idx, ...], ...}"""
        self.restore()
        layers = getattr(self.model.model, "layers", None)
        if layers is None:
            return
        for layer_idx, layer in enumerate(layers):
            chans = per_layer_pruned_channels.get(layer_idx, [])
            if not chans:
                continue
            mlp = getattr(layer, "mlp", None)
            down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
            if down_proj is None:
                continue

            def make_pre_hook(channel_indices):
                def pre_hook(module, inputs):
                    if not inputs or inputs[0] is None:
                        return inputs
                    x = inputs[0]
                    x = x.clone()
                    x[..., channel_indices] = 0
                    return (x,)
                return pre_hook

            h = down_proj.register_forward_pre_hook(make_pre_hook(sorted(set(chans))))
            self.hooks.append(h)
        self.active = True

    def restore(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.active = False
