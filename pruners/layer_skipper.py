from typing import List

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
            def identity_forward(x, *args, **kwargs):
                return x
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
