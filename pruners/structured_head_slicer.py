import torch
import torch.nn as nn
from typing import Dict, List

class StructuredHeadSlicer:
    """
    Structurally prune attention heads in LLaMA attention by rebuilding
    q_proj, k_proj, v_proj, and o_proj with fewer heads. This yields real
    speedups by reducing projection dimensions.

    Reversible by caching original modules per layer.
    """
    def __init__(self, model):
        self.model = model
        self._orig = {}  # layer_idx -> dict(q_proj,k_proj,v_proj,o_proj,num_heads)
        self.active = False

    def restore(self):
        if not self._orig:
            return
        layers = getattr(self.model.model, "layers", None)
        if layers is None:
            self._orig.clear()
            self.active = False
            return
        for idx, saved in self._orig.items():
            try:
                attn = layers[idx].self_attn
                attn.q_proj = saved['q_proj']
                attn.k_proj = saved['k_proj']
                attn.v_proj = saved['v_proj']
                attn.o_proj = saved['o_proj']
                if hasattr(attn, 'num_heads'):
                    attn.num_heads = saved['num_heads']
            except Exception:
                pass
        self._orig.clear()
        self.active = False

    def _ensure_cache(self, layer_idx: int, attn) -> None:
        if layer_idx in self._orig:
            return
        self._orig[layer_idx] = {
            'q_proj': attn.q_proj,
            'k_proj': attn.k_proj,
            'v_proj': attn.v_proj,
            'o_proj': attn.o_proj,
            'num_heads': getattr(attn, 'num_heads', None),
        }

    @staticmethod
    def _new_linear(in_features: int, out_features: int, like: nn.Linear) -> nn.Linear:
        return nn.Linear(in_features, out_features, bias=(like.bias is not None), device=like.weight.device, dtype=like.weight.dtype)

    def prune(self, per_layer_removed_q: Dict[int, List[int]], per_layer_removed_kv: Dict[int, List[int]] = None):
        """
        Remove the specified Qheads and KV heads for each layer's attention by
        rebuilding q_proj, k_proj, v_proj, and o_proj with reduced shapes.
        """
        if per_layer_removed_kv is None:
            per_layer_removed_kv = per_layer_removed_q  # assume same
        self.restore()
        layers = getattr(self.model.model, "layers", None)
        if layers is None:
            return
        for layer_idx, layer in enumerate(layers):
            removed_q = sorted(set(per_layer_removed_q.get(layer_idx, [])))
            removed_kv = sorted(set(per_layer_removed_kv.get(layer_idx, [])))
            if not removed_q:
                continue
            attn = getattr(layer, 'self_attn', None)
            if attn is None:
                continue
            # Infer head counts and dims
            num_heads = getattr(attn, 'num_heads', None)
            head_dim = getattr(attn, 'head_dim', None)
            if num_heads is None or head_dim is None:
                cfg = getattr(self.model, 'config', None)
                if cfg is not None and getattr(cfg, 'num_attention_heads', None) is not None:
                    num_heads = getattr(cfg, 'num_attention_heads')
                # derive head_dim from o_proj.in_features or q_proj.out_features
                if head_dim is None and hasattr(attn, 'o_proj') and hasattr(attn.o_proj, 'in_features') and num_heads is not None:
                    if attn.o_proj.in_features % num_heads == 0:
                        head_dim = attn.o_proj.in_features // num_heads
            if num_heads is None or head_dim is None:
                continue
            keep_q = [h for h in range(num_heads) if h not in removed_q]
            if len(keep_q) == num_heads or len(keep_q) < 1:
                continue

            # Handle GQA: KV heads may be fewer
            num_kv_heads = attn.k_proj.weight.shape[0] // head_dim
            keep_kv = [h for h in range(num_kv_heads) if h not in removed_kv]
            if len(keep_kv) < 1:
                keep_kv = [0]  # avoid degenerate

            self._ensure_cache(layer_idx, attn)

            q, k, v, o = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
            # Shapes: q: out_features = num_heads*head_dim, k/v: num_kv_heads*head_dim, o: in_features = num_heads*head_dim
            with torch.no_grad():
                # Build new q/k/v with reduced out_features
                new_q = self._new_linear(q.in_features, len(keep_q)*head_dim, q)
                new_k = self._new_linear(k.in_features, len(keep_kv)*head_dim, k)
                new_v = self._new_linear(v.in_features, len(keep_kv)*head_dim, v)
                # Reshape and slice rows by head for q/k/v
                def slice_qkv(weight: torch.Tensor, num_local: int, keep_local: List[int]):
                    return weight.view(num_local, head_dim, -1)[keep_local, :, :].reshape(len(keep_local)*head_dim, -1)
                new_q.weight.copy_(slice_qkv(q.weight, num_heads, keep_q))
                new_k.weight.copy_(slice_qkv(k.weight, num_kv_heads, keep_kv))
                new_v.weight.copy_(slice_qkv(v.weight, num_kv_heads, keep_kv))
                if q.bias is not None:
                    new_q.bias.copy_(q.bias.view(num_heads, head_dim)[keep_q, :].reshape(-1))
                if k.bias is not None:
                    new_k.bias.copy_(k.bias.view(num_kv_heads, head_dim)[keep_kv, :].reshape(-1))
                if v.bias is not None:
                    new_v.bias.copy_(v.bias.view(num_kv_heads, head_dim)[keep_kv, :].reshape(-1))

                # Build new o with reduced in_features
                new_o = self._new_linear(len(keep_q)*head_dim, o.out_features, o)
                # Slice input columns grouped by head
                keep_cols = []
                for h in keep_q:
                    start = h * head_dim
                    keep_cols.extend(list(range(start, start+head_dim)))
                new_o.weight.copy_(o.weight[:, keep_cols])
                if o.bias is not None:
                    new_o.bias.copy_(o.bias)

            # Replace modules and adjust head count
            attn.q_proj = new_q
            attn.k_proj = new_k
            attn.v_proj = new_v
            attn.o_proj = new_o
            if hasattr(attn, 'num_heads'):
                attn.num_heads = len(keep_q)
        self.active = True
