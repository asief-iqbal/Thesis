"""
KV Cache Pruner
Runtime pruning by reducing generation length to simulate cache size reduction.
This effectively prunes KV cache usage by limiting output tokens.
"""

class KVCachePruner:
    def __init__(self, model):
        self.model = model
        self.prune_intensity = 0.0

    def apply_prune(self, intensity: float):
        """Apply KV cache pruning by setting intensity (reduces generation length)."""
        self.prune_intensity = intensity
        print(f"[KV Cache Pruner] Applied pruning intensity {intensity:.2f} (reduces generation length by {intensity*100:.0f}%)")

    def restore(self):
        """Restore to no pruning."""
        self.prune_intensity = 0.0

    def get_effective_max_length(self, base_max_length: int) -> int:
        """Get the effective max length after pruning."""
        return max(1, int(base_max_length * (1.0 - self.prune_intensity)))
