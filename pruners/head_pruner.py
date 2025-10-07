import torch
from typing import Dict, List

class HeadPruner:
    """
    Runtime head-masking via forward pre-hooks on LlamaAttention.o_proj.
    This is reversible and serves as scaffolding for Phase 2. It zeros
    the slices corresponding to selected heads before the output proj.
    Note: This is a functional mask and does not yield speedups; Phase 3
    will implement structural slicing for real acceleration.
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.active = False

    def apply(self, per_layer_pruned_heads: Dict[int, List[int]]):
        """per_layer_pruned_heads: {layer_idx: [head_idx, ...], ...}"""
        self.restore()
        layers = getattr(self.model.model, "layers", None)
        if layers is None:
            return
        for layer_idx, layer in enumerate(layers):
            heads = per_layer_pruned_heads.get(layer_idx, [])
            if not heads:
                continue
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "o_proj"):
                continue
            # Try to infer num_heads and head_dim robustly
            num_heads = getattr(attn, "num_heads", None)
            head_dim = getattr(attn, "head_dim", None)
            if num_heads is None:
                # Try from model config
                cfg = getattr(self.model, "config", None)
                num_heads = getattr(cfg, "num_attention_heads", None) if cfg is not None else None
            if head_dim is None and hasattr(attn.o_proj, "in_features") and num_heads:
                in_features = int(getattr(attn.o_proj, "in_features"))
                if in_features % int(num_heads) == 0:
                    head_dim = in_features // int(num_heads)
            if num_heads is None or head_dim is None:
                continue
            # Sanitize head indices
            indices = sorted(h for h in heads if 0 <= h < num_heads)
            if not indices:
                continue
            # Build a vectorized mask once per layer and apply via broadcast multiply
            hidden_size = int(num_heads * head_dim)
            device = attn.o_proj.weight.device
            dtype = attn.o_proj.weight.dtype
            mask_vec = torch.ones(hidden_size, device=device, dtype=dtype)
            for h_idx in indices:
                start = h_idx * head_dim
                mask_vec[start:start + head_dim] = 0

            def make_pre_hook(mask):
                def pre_hook(module, inputs):
                    if not inputs or inputs[0] is None:
                        return inputs
                    x = inputs[0]
                    m = mask.to(device=x.device, dtype=x.dtype)
                    # x: [..., D], m: [D] will broadcast along batch/seq dims
                    return (x * m,)
                return pre_hook

            h = attn.o_proj.register_forward_pre_hook(make_pre_hook(mask_vec))
            self.hooks.append(h)
        self.active = True

    def restore(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception as e:
                print(f"[HeadPruner] Exception during restore: {e}")
        self.hooks.clear()
        self.active = False
