import copy
from typing import Dict, List
import torch
import torch.nn as nn

class StructuredFFNSlicer:
    """
    Structurally prune LLaMA MLP intermediate channels per layer by
    rebuilding Linear layers with reduced dimensions. This yields
    real speedups compared to masking. Reversible via cached originals.

    Usage:
      slicer = StructuredFFNSlicer(model)
      slicer.prune({ layer_idx: [removed_channel_indices, ...], ... })
      # or provide keep indices via prune_with_keep()
    """
    def __init__(self, model):
        self.model = model
        self._orig = {}  # layer_idx -> dict(gate_proj, up_proj, down_proj)
        self.active = False

    def _ensure_cache(self, layer_idx: int, mlp) -> None:
        if layer_idx in self._orig:
            return
        self._orig[layer_idx] = {
            'gate_proj': mlp.gate_proj,
            'up_proj': mlp.up_proj,
            'down_proj': mlp.down_proj,
        }

    def restore(self):
        """Restore all layers to original MLP projection shapes."""
        if not self._orig:
            return
        layers = getattr(self.model.model, "layers", None)
        if layers is None:
            self._orig.clear()
            self.active = False
            return
        for idx, saved in self._orig.items():
            try:
                mlp = layers[idx].mlp
                mlp.gate_proj = saved['gate_proj']
                mlp.up_proj = saved['up_proj']
                mlp.down_proj = saved['down_proj']
            except Exception:
                pass
        self._orig.clear()
        self.active = False

    @staticmethod
    def _build_linear(in_features: int, out_features: int, device, dtype, bias: bool) -> nn.Linear:
        lin = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        return lin

    def prune(self, per_layer_removed_channels: Dict[int, List[int]]):
        """
        Remove the specified intermediate channels for each layer's MLP by
        rebuilding gate_proj/up_proj/down_proj with reduced shapes.
        """
        self.restore()
        layers = getattr(self.model.model, "layers", None)
        if layers is None:
            return
        for layer_idx, layer in enumerate(layers):
            removed = sorted(set(per_layer_removed_channels.get(layer_idx, [])))
            if not removed:
                continue
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
            if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear) or not isinstance(down, nn.Linear):
                continue

            inter_size = down.weight.shape[1]  # in_features of down_proj == intermediate_size
            hidden_size = down.weight.shape[0]  # out_features of down_proj == hidden_size
            keep = [i for i in range(inter_size) if i not in removed]
            if len(keep) == inter_size:
                continue
            if len(keep) < 1:
                # Avoid degenerate MLP
                continue

            self._ensure_cache(layer_idx, mlp)

            device = down.weight.device
            dtype = down.weight.dtype
            bias_gate = gate.bias is not None
            bias_up = up.bias is not None
            bias_down = down.bias is not None

            # Build new layers with reduced intermediate size
            new_inter = len(keep)
            new_gate = self._build_linear(gate.in_features, new_inter, device, dtype, bias_gate)
            new_up = self._build_linear(up.in_features, new_inter, device, dtype, bias_up)
            new_down = self._build_linear(new_inter, down.out_features, device, dtype, bias_down)

            # Copy weights: for gate/up, select rows (out_features); for down, select columns (in_features)
            with torch.no_grad():
                new_gate.weight.copy_(gate.weight[keep, :])
                if bias_gate:
                    new_gate.bias.copy_(gate.bias[keep])
                new_up.weight.copy_(up.weight[keep, :])
                if bias_up:
                    new_up.bias.copy_(up.bias[keep])
                new_down.weight.copy_(down.weight[:, keep])
                if bias_down:
                    new_down.bias.copy_(down.bias)

            # Replace modules
            mlp.gate_proj = new_gate
            mlp.up_proj = new_up
            mlp.down_proj = new_down

        self.active = True

    def prune_with_keep(self, per_layer_keep_channels: Dict[int, List[int]]):
        """Alternate API: provide channels to keep per layer."""
        per_layer_removed = {}
        for layer_idx, keep in per_layer_keep_channels.items():
            # Infer original inter size from any layer's down_proj
            # This will be computed inside prune() anyway
            # Here we just compute removed = full - keep
            layers = getattr(self.model.model, "layers", None)
            if layers is None:
                continue
            mlp = getattr(layers[layer_idx], "mlp", None)
            if mlp is None:
                continue
            inter_size = mlp.down_proj.weight.shape[1]
            removed = [i for i in range(inter_size) if i not in set(keep)]
            per_layer_removed[layer_idx] = removed
        self.prune(per_layer_removed)
