import os
import torch
os.environ['HF_HOME'] = 'd:\\LLM Research\\hf_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from pruners.head_pruner import HeadPruner
from pruners.ffn_pruner import FFNPruner
from pruners.layer_skipper import LayerSkipper
from pruners.structured_ffn_slicer import StructuredFFNSlicer
from pruners.structured_head_slicer import StructuredHeadSlicer
from pruners.kv_cache_pruner import KVCachePruner


def _load_hf_token_from_env(env_path: str = ".env") -> str:
    """Load Hugging Face token from .env or environment variables."""
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    # Minimal .env reader to avoid extra deps
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("HUGGINGFACE_HUB_TOKEN="):
                        return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


class RealModelEngine:
    def __init__(self):
        token = _load_hf_token_from_env()
        if token:
            login(token=token)
        else:
            print("[Engine] Warning: No HF token found in env/.env. Using cached auth if available.")

        model_name = "meta-llama/Llama-3.2-1B"
        print(f"[Engine] Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[Engine] Llama 3.2 1B model loaded to {self.model.device}.")

        # Phase 2 scaffold: reversible head/ffn/layer masking via hooks
        self.head_pruner = HeadPruner(self.model)
        self.ffn_pruner = FFNPruner(self.model)
        self.layer_skipper = LayerSkipper(self.model)
        # Phase 3 (structural): FFN & Head slicers that rebuild layers for real speedups
        self.structured_ffn = StructuredFFNSlicer(self.model)
        self.structured_head = StructuredHeadSlicer(self.model)
        # KV Cache pruner for runtime cache size reduction
        self.kv_pruner = KVCachePruner(self.model)

        # Calibration caches (activation-aware importances)
        self.head_importance = {}   # layer_idx -> tensor[num_heads]
        self.ffn_importance = {}    # layer_idx -> tensor[inter_size]
        self.layer_importance = []  # per-layer importance (avg |activation| over tokens)

    def apply_pruning(self, action) -> None:
        """Apply pruning according to the action. Currently supports head masking scaffold."""
        target = getattr(action, "target", "none")
        intensity = float(getattr(action, "intensity", 0.0) or 0.0)
        structural_env = os.getenv("STRUCTURAL_PRUNING", "0") == "1"
        # Force structural pruning for heads/FFN to realize real throughput gains
        force_structural = target in ("attention_heads", "ffn_neurons")
        structural = force_structural or structural_env

        if target == "attention_heads" and intensity > 0.0:
            # Determine number of heads per layer (assume homogeneous across layers)
            layers = getattr(self.model.model, "layers", [])
            if not layers:
                print("[Engine] No layers found; cannot apply head pruning.")
                return
            # Prefer config if available
            cfg = getattr(self.model, "config", None)
            num_heads = getattr(cfg, "num_attention_heads", None) if cfg is not None else None
            if num_heads is None:
                # fallback to first layer attr
                attn0 = getattr(layers[0], "self_attn", None)
                num_heads = getattr(attn0, "num_heads", None)
            if num_heads is None:
                print("[Engine] Couldn't determine num_heads; skipping head pruning.")
                return
            k = max(1, int(round(num_heads * intensity)))
            k = min(k, num_heads)
            if structural:
                # Magnitude-based per-head importance using q/k/v weights (GQA-safe)
                per_layer_removed = {}
                for i, layer in enumerate(layers):
                    attn = getattr(layer, "self_attn", None)
                    if attn is None:
                        continue
                    # Prefer calibration if available
                    if i in self.head_importance and len(self.head_importance[i]) == num_heads:
                        scores = self.head_importance[i]
                        removed = torch.topk(scores, k, largest=False).indices.tolist()
                    else:
                        q = getattr(attn, 'q_proj', None)
                        kproj = getattr(attn, 'k_proj', None)
                        v = getattr(attn, 'v_proj', None)
                        o = getattr(attn, 'o_proj', None)
                        if any(m is None for m in [q, kproj, v, o]):
                            continue
                        try:
                            head_dim = (q.out_features // num_heads)
                        except Exception:
                            continue
                        with torch.no_grad():
                            def head_scores_linear(m: torch.nn.Linear):
                                W = m.weight.view(num_heads, head_dim, -1)
                                return torch.linalg.vector_norm(W, ord=2, dim=(1, 2))
                            def safe_scores(m: torch.nn.Linear):
                                try:
                                    return head_scores_linear(m)
                                except Exception:
                                    return None
                            s_q = head_scores_linear(q)
                            s_k = safe_scores(kproj)
                            s_v = safe_scores(v)
                            if s_k is not None and s_v is not None and s_k.shape[0] == s_q.shape[0] and s_v.shape[0] == s_q.shape[0]:
                                scores = s_q + s_k + s_v
                            else:
                                # GQA-safe fallback: use Q-only importance
                                scores = s_q
                            removed = torch.topk(scores, k, largest=False).indices.tolist()
                    per_layer_removed[i] = removed
                print(f"[Engine] Applying STRUCTURAL head pruning: removing {k}/{num_heads} heads per layer (calib/magnitude-based).")
                self.structured_head.prune(per_layer_removed)
            else:
                # Functional masking with calibration-aware or magnitude-based importance (GQA-safe)
                per_layer_removed = {}
                for i, layer in enumerate(layers):
                    attn = getattr(layer, "self_attn", None)
                    if attn is None:
                        continue
                    if i in self.head_importance and len(self.head_importance[i]) == num_heads:
                        scores = self.head_importance[i]
                        removed = torch.topk(scores, k, largest=False).indices.tolist()
                    else:
                        q = getattr(attn, 'q_proj', None)
                        kproj = getattr(attn, 'k_proj', None)
                        v = getattr(attn, 'v_proj', None)
                        o = getattr(attn, 'o_proj', None)
                        if any(m is None for m in [q, kproj, v, o]):
                            continue
                        try:
                            head_dim = (q.out_features // num_heads)
                        except Exception:
                            continue
                        with torch.no_grad():
                            def head_scores_linear(m: torch.nn.Linear):
                                W = m.weight.view(num_heads, head_dim, -1)
                                return torch.linalg.vector_norm(W, ord=2, dim=(1, 2))
                            def safe_scores(m: torch.nn.Linear):
                                try:
                                    return head_scores_linear(m)
                                except Exception:
                                    return None
                            s_q = head_scores_linear(q)
                            s_k = safe_scores(kproj)
                            s_v = safe_scores(v)
                            if s_k is not None and s_v is not None and s_k.shape[0] == s_q.shape[0] and s_v.shape[0] == s_q.shape[0]:
                                scores = s_q + s_k + s_v
                            else:
                                scores = s_q
                            removed = torch.topk(scores, k, largest=False).indices.tolist()
                    per_layer_removed[i] = removed
                print(f"[Engine] Applying head mask (calib-aware): pruning {k}/{num_heads} heads per layer (functional).")
                self.head_pruner.apply(per_layer_removed)
        elif target == "none":
            self.head_pruner.restore()
            self.ffn_pruner.restore()
            self.layer_skipper.restore()
        elif target == "ffn_neurons" and intensity > 0.0:
            layers = getattr(self.model.model, "layers", [])
            if not layers:
                print("[Engine] No layers found; cannot apply FFN pruning.")
                return
            mlp0 = getattr(layers[0], "mlp", None)
            down_proj0 = getattr(mlp0, "down_proj", None) if mlp0 is not None else None
            if down_proj0 is None or not hasattr(down_proj0, "weight"):
                print("[Engine] Couldn't read FFN dims; skipping FFN pruning.")
                return
            inter_size = down_proj0.weight.shape[1]
            k = max(1, int(round(inter_size * intensity)))
            k = min(k, inter_size)

            if structural:
                # Magnitude-based structured pruning: choose least-important channels per layer
                per_layer_removed = {}
                for i, layer in enumerate(layers):
                    mlp = getattr(layer, "mlp", None)
                    if mlp is None or not hasattr(mlp, "down_proj"):
                        continue
                    # Prefer calibration if available
                    if i in self.ffn_importance:
                        scores = self.ffn_importance[i]
                        removed = torch.topk(scores, k, largest=False).indices.tolist()
                    else:
                        gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
                        # Compute per-channel importance using L2 norms from gate/up rows and down columns
                        with torch.no_grad():
                            g_scores = torch.linalg.vector_norm(gate.weight, ord=2, dim=1)
                            u_scores = torch.linalg.vector_norm(up.weight, ord=2, dim=1)
                            d_scores = torch.linalg.vector_norm(down.weight, ord=2, dim=0)
                            scores = g_scores + u_scores + d_scores
                            removed = torch.topk(scores, k, largest=False).indices.tolist()
                    per_layer_removed[i] = removed
                print(f"[Engine] Applying STRUCTURAL FFN pruning: removing {k}/{inter_size} channels per layer (calib-aware/magnitude-based).")
                self.structured_ffn.prune(per_layer_removed)
            else:
                # Functional masking with calibration-aware or magnitude-based importance
                per_layer_removed = {}
                for i, layer in enumerate(layers):
                    mlp = getattr(layer, "mlp", None)
                    if mlp is None or not hasattr(mlp, "down_proj"):
                        continue
                    if i in self.ffn_importance:
                        scores = self.ffn_importance[i]
                        removed = torch.topk(scores, k, largest=False).indices.tolist()
                    else:
                        gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
                        with torch.no_grad():
                            g_scores = torch.linalg.vector_norm(gate.weight, ord=2, dim=1)
                            u_scores = torch.linalg.vector_norm(up.weight, ord=2, dim=1)
                            d_scores = torch.linalg.vector_norm(down.weight, ord=2, dim=0)
                            scores = g_scores + u_scores + d_scores
                            removed = torch.topk(scores, k, largest=False).indices.tolist()
                    per_layer_removed[i] = removed
                print(f"[Engine] Applying FFN channel mask (calib-aware): pruning {k}/{inter_size} channels per layer (functional).")
                self.ffn_pruner.apply(per_layer_removed)
        elif target == "transformer_layers" and intensity > 0.0:
            layers = getattr(self.model.model, "layers", [])
            L = len(layers)
            if L == 0:
                print("[Engine] No layers found; cannot skip layers.")
                return
            k = max(1, int(round(L * intensity)))
            # Cap skipping to at most 12.5% of layers to avoid catastrophic degradation (e.g., <=2 of 16)
            k = min(k, max(1, L // 8))
            k = min(k, L)
            # Select least-important layers from calibrated scores (exclude first/last)
            allowed = [i for i in range(L) if i not in (0, L-1)]
            if isinstance(self.layer_importance, list) and len(self.layer_importance) == L and any(isinstance(v, (int, float)) for v in self.layer_importance):
                scores = sorted([(i, float(self.layer_importance[i])) for i in allowed], key=lambda x: x[1])
                to_skip = [i for i, _ in scores[:k]] if scores else []
                if len(to_skip) < k:
                    # Fallback to trailing layers if insufficient
                    fallback = [i for i in range(L - k, L) if i in allowed and i not in to_skip]
                    to_skip = (to_skip + fallback)[:k]
                print(f"[Engine] Skipping {k}/{L} least-important layers: {to_skip}")
            else:
                # Fallback: skip last-k
                to_skip = [i for i in range(L - k, L) if i in allowed]
                print(f"[Engine] Skipping {len(to_skip)}/{L} trailing layers (no layer importance available): {to_skip}")
            self.layer_skipper.apply(to_skip)
        elif target == "kv_cache" and intensity > 0.0:
            self.kv_pruner.apply_prune(intensity)
        else:
            # Other targets (ffn_neurons, transformer_layers) will be implemented structurally in Phase 2/3
            print(f"[Engine] Pruning target '{target}' not yet implemented structurally; no-op for now.")

    def restore_model(self) -> None:
        # Restore functional masks
        self.head_pruner.restore()
        self.ffn_pruner.restore()
        self.layer_skipper.restore()
        # Restore structural slicing (if any)
        self.structured_ffn.restore()
        self.structured_head.restore()
        # Restore KV cache pruning
        self.kv_pruner.restore()

    def calibrate_importances(self, prompts, max_samples: int = 64, max_seq_len: int = 128) -> None:
        """Collect activation-aware importance for heads and FFN channels using a small calibration set.
        Importance is average absolute activation per head/channel across tokens and samples.
        """
        layers = getattr(self.model.model, "layers", [])
        if not layers:
            print("[Calib] No layers found; skipping calibration.")
            return
        # Prepare accumulators
        head_acc = {}
        ffn_acc = {}
        hooks = []
        total_tokens = 0
        # Layer-wise accumulators
        layer_sums = {}
        layer_counts = {}

        # Infer num_heads and inter_size
        try:
            attn0 = layers[0].self_attn
            num_heads = getattr(attn0, 'num_heads', None)
            head_dim = getattr(attn0, 'head_dim', None)
            if num_heads is None and hasattr(attn0.o_proj, 'in_features'):
                cfg = getattr(self.model, 'config', None)
                nh = getattr(cfg, 'num_attention_heads', None) if cfg is not None else None
                if nh is not None:
                    num_heads = nh
                if head_dim is None and num_heads is not None:
                    if attn0.o_proj.in_features % num_heads == 0:
                        head_dim = attn0.o_proj.in_features // num_heads
        except Exception:
            num_heads = None
            head_dim = None
        try:
            inter_size = layers[0].mlp.down_proj.weight.shape[1]
        except Exception:
            inter_size = None

        if num_heads is not None and head_dim is not None:
            for i, layer in enumerate(layers):
                head_acc[i] = torch.zeros(num_heads, device=self.model.device)
                # register head hook on o_proj pre-hook
                attn = layer.self_attn
                def make_head_hook(idx):
                    def pre_hook(module, inputs):
                        nonlocal total_tokens
                        if not inputs or inputs[0] is None:
                            return inputs
                        x = inputs[0]  # [..., h*dim]
                        B, T, D = x.shape if x.dim() == 3 else (1, x.shape[0], x.shape[-1])
                        total_tokens += (B*T)
                        try:
                            x_ = x.view(B*T, num_heads, head_dim)
                        except Exception:
                            return inputs
                        # sum abs over head_dim and tokens
                        vals = x_.abs().sum(dim=2).sum(dim=0)
                        head_acc[idx][:] += vals.to(head_acc[idx].device)
                        return inputs
                    return pre_hook
                hooks.append(attn.o_proj.register_forward_pre_hook(make_head_hook(i)))

        if inter_size is not None:
            for i, layer in enumerate(layers):
                ffn_acc[i] = torch.zeros(inter_size, device=self.model.device)
                mlp = layer.mlp
                def make_ffn_hook(idx):
                    def pre_hook(module, inputs):
                        nonlocal total_tokens
                        if not inputs or inputs[0] is None:
                            return inputs
                        x = inputs[0]  # [..., inter_size]
                        if x.dim() == 3:
                            B, T, C = x.shape
                            total_tokens += (B * T)
                            vals = x.abs().sum(dim=(0, 1))  # -> [C]
                        else:
                            total_tokens += x.shape[0] if x.dim() > 0 else 1
                            vals = x.abs().sum(dim=0)
                        ffn_acc[idx][:] += vals.to(ffn_acc[idx].device)
                        return inputs
                    return pre_hook
                hooks.append(mlp.down_proj.register_forward_pre_hook(make_ffn_hook(i)))

        # Layer output importance via forward hooks (avg |activation| per token)
        for i, layer in enumerate(layers):
            layer_sums[i] = 0.0
            layer_counts[i] = 0
            def make_layer_hook(idx):
                def fwd_hook(module, inputs, output):
                    try:
                        hs = output[0] if isinstance(output, (tuple, list)) else output
                        if hs is None:
                            return
                        if hasattr(hs, 'dim') and hs.dim() >= 1:
                            if hs.dim() == 3:
                                B, T, _ = hs.shape
                                layer_counts[idx] += (B * T)
                            else:
                                layer_counts[idx] += hs.shape[0] if hs.dim() > 0 else 1
                            layer_sums[idx] += hs.detach().abs().sum().item()
                    except Exception:
                        pass
                return fwd_hook
            hooks.append(layer.register_forward_hook(make_layer_hook(i)))

        # Run calibration forward passes
        used = 0
        self.model.eval()
        with torch.no_grad():
            for p in prompts[:max_samples]:
                tok = self.tokenizer(p, return_tensors='pt', truncation=True, max_length=max_seq_len).to(self.model.device)
                try:
                    _ = self.model(**tok)
                    used += 1
                except Exception:
                    continue

        # Remove hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        # Normalize and store importances
        if used > 0 and num_heads is not None:
            self.head_importance = {i: (acc / max(1, total_tokens)).detach().cpu() for i, acc in head_acc.items()}
        if used > 0 and inter_size is not None:
            self.ffn_importance = {i: (acc / max(1, total_tokens)).detach().cpu() for i, acc in ffn_acc.items()}
        # Layer importance: average abs activation per token per layer
        if used > 0 and layers:
            L = len(layers)
            self.layer_importance = []
            for i in range(L):
                s = float(layer_sums.get(i, 0.0))
                c = float(layer_counts.get(i, 0))
                self.layer_importance.append(s / max(1.0, c))
        print(f"[Calib] Completed on {used} samples. Head importances: {len(self.head_importance)} layers, FFN importances: {len(self.ffn_importance)} layers, Layer importances: {len(self.layer_importance)}.")

    def generate_response(self, prompt: str, max_length: int = 50) -> str:
        effective_max = self.kv_pruner.get_effective_max_length(max_length)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # Enforce fixed-length generation for fair throughput/ppl comparisons.
{{ ... }}
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=effective_max,
                min_new_tokens=effective_max,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except TypeError:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=effective_max,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def save_pretrained(self, out_dir: str) -> None:
        """Persist current model/tokenizer (including structural pruning) for external eval (e.g., lm-eval-harness)."""
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        print(f"[Engine] Model and tokenizer saved to {out_dir}")
