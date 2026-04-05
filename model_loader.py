import os
import shutil
import torch
import copy
from typing import Optional
_DEFAULT_HF_HOME = os.path.join(os.getcwd(), ".hf_cache")
if not os.getenv("HF_HOME"):
    # Avoid hard-coding a machine-specific drive path.
    os.environ["HF_HOME"] = _DEFAULT_HF_HOME
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from pruners.head_pruner import HeadPruner
from pruners.ffn_pruner import FFNPruner
from pruners.structured_ffn_slicer import StructuredFFNSlicer
from pruners.structured_head_slicer import StructuredHeadSlicer
from pruners.layer_skipper import LayerSkipper


def _infer_attention_type(config) -> str:
    """Detect attention architecture: MHA, GQA, or MQA from model config."""
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None or num_kv_heads == num_heads:
        return "MHA"
    elif num_kv_heads == 1:
        return "MQA"
    return "GQA"


def _load_hf_token_from_env(env_path: str = ".env") -> str:
    """Load Hugging Face token from .env or environment variables.

    Supported keys (env var or .env):
    - HUGGINGFACE_HUB_TOKEN (preferred)
    - HF_TOKEN
    - HUGGINGFACE_TOKEN (also matches user-style 'HuggingFace_token')
    """

    for key in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.getenv(key)
        if token:
            return token.strip().strip('"').strip("'")
    # Minimal .env reader to avoid extra deps
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.lower().startswith("export "):
                        line = line[7:].lstrip()
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = (k or "").strip()
                    v = (v or "").strip().strip('"').strip("'")
                    if k.upper() in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
                        return v
    except Exception:
        pass
    return ""


class RealModelEngine:
    def __init__(
        self,
        device: str = "auto",
        enable_static_profiles: bool = False,
        enable_2to4: bool = False,
        enable_compile: bool = False,
        enable_kv_compression: bool = False,
        kv_keep_ratio: float = 1.0,
        model_name: Optional[str] = None,
        local_files_only: bool = False,
    ):
        token = _load_hf_token_from_env()
        if token:
            login(token=token)
        else:
            print("[Engine] Warning: No HF token found in env/.env. Using cached auth if available.")

        self.model_name = model_name or os.getenv("BACKBONE_MODEL_NAME") or "meta-llama/Llama-3.2-1B"
        print(f"[Engine] Loading model: {self.model_name} on device {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=bool(local_files_only))
        if device == "cpu":
            self.device_map = None
        else:
            self.device_map = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            local_files_only=bool(local_files_only),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[Engine] Llama 3.2 1B model loaded to {self.model.device}.")
        # Ensure inference-optimized defaults
        try:
            self.model.eval()
            if hasattr(self.model, 'config'):
                setattr(self.model.config, 'use_cache', True)
            if hasattr(self.model, 'generation_config'):
                setattr(self.model.generation_config, 'use_cache', True)
        except Exception:
            pass

        # Capabilities
        # Force-disable static profiles to avoid model reloads and torch.compile cache invalidations
        # regardless of incoming flag.
        self.enable_static_profiles = False
        self.enable_2to4 = bool(enable_2to4)
        self.enable_compile = bool(enable_compile)
        self.profiles_dir = os.path.join(os.getcwd(), 'profiles')
        self.active_profile_key = None
        if self.enable_static_profiles:
            print("[Profiles] Static profile caching ENABLED: using prebuilt pruned models for fast inference.")
            # Only compile the base model when using static profiles; dynamic structural changes during training
            # can invalidate compiled graphs and cause slowdowns.
            self.model = self._maybe_compile_model(self.model)
        # KV-cache compression flags (Phase D, scaffold)
        self.enable_kv_compression = bool(enable_kv_compression)
        self.kv_keep_ratio = float(kv_keep_ratio)
        self._kv_warned = False

        # Phase 2 scaffold: reversible head/ffn/layer masking via hooks
        self.head_pruner = HeadPruner(self.model)
        self.ffn_pruner = FFNPruner(self.model)
        import pruners.layer_skipper
        self.layer_skipper = pruners.layer_skipper.LayerSkipper(self.model)
        # Phase 3 (structural): FFN & Head slicers that rebuild layers for real speedups
        self.structured_ffn = StructuredFFNSlicer(self.model)
        self.structured_head = StructuredHeadSlicer(self.model)

        # Calibration caches (activation-aware importances)
        self.head_importance = {}   # layer_idx -> tensor[num_heads]
        self.ffn_importance = {}    # layer_idx -> tensor[inter_size]
        self.layer_importance = []  # per-layer importance (avg |activation| over tokens)

    def _rebuild_pruners(self) -> None:
        # Rebuild pruners for the current self.model (used after switching profiles)
        self.head_pruner = HeadPruner(self.model)
        self.ffn_pruner = FFNPruner(self.model)
        import pruners.layer_skipper
        self.layer_skipper = pruners.layer_skipper.LayerSkipper(self.model)
        self.structured_ffn = StructuredFFNSlicer(self.model)
        self.structured_head = StructuredHeadSlicer(self.model)

    def _profile_key(self, action) -> str:
        t = getattr(action, 'target', 'none')
        s = str(float(getattr(action, 'intensity', 0.0) or 0.0))
        return f"{t}_{s}"

    def _profile_dir(self, key: str) -> str:
        return os.path.join(self.profiles_dir, key)

    def _try_enable_2to4_on_model(self, model: AutoModelForCausalLM) -> None:
        if not getattr(self, 'enable_2to4', False):
            return
        if not torch.cuda.is_available():
            print("[Sparsity] 2:4 requested but CUDA not available; skipping.")
            return
        try:
            # Placeholder for semi-structured sparsity packing (requires torchao / PyTorch AO).
            print("[Sparsity] 2:4 semi-structured packing scaffold only (no-op in this environment).")
        except Exception as e:
            print(f"[Sparsity] 2:4 packing failed (skipping): {e}")

    def _maybe_compile_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        if not getattr(self, 'enable_compile', False):
            return model
        try:
            compile = getattr(torch, 'compile', None)
            if compile is None:
                print("[Compile] torch.compile not available; skipping.")
                return model
            print("[Compile] Compiling model profile (one-time)...")
            return compile(model)
        except Exception as e:
            print(f"[Compile] compile failed (skipping): {e}")
            return model

    def _build_profile_model(self, action) -> AutoModelForCausalLM:
        # Fresh model instance to apply structural pruning and persist as a profile
        # Load on CPU to avoid meta tensors for saving
        new_model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=None)
        new_model = new_model.to('cpu')  # Ensure all tensors are on CPU, no meta
        target = getattr(action, "target", "none")
        intensity = float(getattr(action, "intensity", 0.0) or 0.0)
        if target == 'none' or intensity <= 0.0:
            return new_model

        # Local slicers for the fresh model
        local_head = StructuredHeadSlicer(new_model)
        local_ffn = StructuredFFNSlicer(new_model)

        layers = getattr(new_model.model, 'layers', [])

        # Phase C: collect baseline layer output sums before pruning (for gain correction)
        sums_before = None
        try:
            prompts = getattr(self, '_calib_prompts_recent', None)
            if prompts:
                sums_before = self._collect_layer_output_sums(new_model, prompts, max_samples=min(32, len(prompts)), max_seq_len=128)
        except Exception:
            sums_before = None
        if target == 'attention_heads':
            cfg = getattr(new_model, "config", None)
            num_heads = getattr(cfg, "num_attention_heads", 32)
            num_kv_heads = getattr(cfg, "num_key_value_heads", max(1, num_heads // 4))
            heads_per_group = max(1, num_heads // num_kv_heads)
            groups_to_remove_count = int(round(intensity * num_kv_heads))
            if groups_to_remove_count <= 0:
                return new_model
            per_layer_removed_q, per_layer_removed_kv = {}, {}
            for i, _layer in enumerate(layers):
                q_head_scores = self.head_importance.get(i, torch.ones(num_heads))
                group_scores = q_head_scores.view(num_kv_heads, heads_per_group).mean(dim=1)
                kv_heads_to_remove = torch.topk(group_scores, groups_to_remove_count, largest=False).indices.tolist()
                q_heads_to_remove = []
                for kv_idx in kv_heads_to_remove:
                    start_q_idx = kv_idx * heads_per_group
                    q_heads_to_remove.extend(range(start_q_idx, start_q_idx + heads_per_group))
                per_layer_removed_q[i] = sorted(q_heads_to_remove)
                per_layer_removed_kv[i] = sorted(kv_heads_to_remove)
            local_head.prune(per_layer_removed_q, per_layer_removed_kv)
        elif target == 'ffn_neurons':
            if not layers:
                return new_model
            inter_size = getattr(layers[0].mlp.down_proj, 'weight', None).shape[1]
            # Compute removal to produce a hardware-friendly new intermediate size
            multiple = 128
            target_size = int(round(inter_size * (1.0 - intensity)))
            new_inter = max(multiple, int(round(target_size / multiple) * multiple))
            new_inter = min(inter_size - 1, new_inter)
            k = max(1, inter_size - new_inter)
            per_layer_removed = {}
            for i, _layer in enumerate(layers):
                if i in self.ffn_importance:
                    scores = self.ffn_importance[i]
                    removed = torch.topk(scores, k, largest=False).indices.tolist()
                else:
                    mlp = _layer.mlp
                    with torch.no_grad():
                        g_scores = torch.linalg.vector_norm(mlp.gate_proj.weight, ord=2, dim=1)
                        u_scores = torch.linalg.vector_norm(mlp.up_proj.weight, ord=2, dim=1)
                        d_scores = torch.linalg.vector_norm(mlp.down_proj.weight, ord=2, dim=0)
                        scores = g_scores + u_scores + d_scores
                        removed = torch.topk(scores, k, largest=False).indices.tolist()
                per_layer_removed[i] = removed
            local_ffn.prune(per_layer_removed)
        else:
            return new_model

        # Optional 2:4 packing and compile
        self._try_enable_2to4_on_model(new_model)
        # Skip compile for profiles to avoid saving issues
        # new_model = self._maybe_compile_model(new_model)

        # Phase C: lightweight reconstruction via per-layer gain correction using calibration prompts
        try:
            if sums_before is not None:
                prompts = getattr(self, '_calib_prompts_recent', None)
                sums_after = self._collect_layer_output_sums(new_model, prompts, max_samples=min(32, len(prompts)), max_seq_len=128)
                eps = 1e-8
                with torch.no_grad():
                    for i, layer in enumerate(getattr(new_model.model, 'layers', [])):
                        sb = float(sums_before.get(i, 1.0))
                        sa = float(sums_after.get(i, 1.0))
                        scale = sb / sa if sa > 0 else 1.0
                        # Bound scale to avoid instability
                        if scale <= 0 or not torch.isfinite(torch.tensor(scale)):
                            scale = 1.0
                        scale = max(0.5, min(2.0, scale))
                        # Apply to attn.o_proj and mlp.down_proj if present
                        attn = getattr(layer, 'self_attn', None)
                        if attn is not None and hasattr(attn, 'o_proj') and hasattr(attn.o_proj, 'weight'):
                            attn.o_proj.weight.mul_(scale)
                        mlp = getattr(layer, 'mlp', None)
                        if mlp is not None and hasattr(mlp, 'down_proj') and hasattr(mlp.down_proj, 'weight'):
                            mlp.down_proj.weight.mul_(scale)
        except Exception:
            pass
        return new_model

    def activate_profile(self, action) -> None:
        key = self._profile_key(action)
        if getattr(self, 'active_profile_key', None) == key:
            return
        if not getattr(self, 'enable_static_profiles', False):
            return
        d = os.path.join(self.profiles_dir, key)

        def build_profile():
            print(f"[Profiles] Building profile '{key}'...")
            if os.path.exists(d):
                shutil.rmtree(d)
            new_model = self._build_profile_model(action)
            os.makedirs(d, exist_ok=True)
            new_model.save_pretrained(d)
            self.tokenizer.save_pretrained(d)

        required_files = ["pytorch_model.bin", "model.safetensors"]
        needs_build = False
        if not os.path.exists(d):
            needs_build = True
        else:
            if not any(os.path.exists(os.path.join(d, fname)) for fname in required_files):
                needs_build = True
        if needs_build:
            build_profile()
        else:
            print(f"[Profiles] Loading cached profile '{key}'...")

        loaded = AutoModelForCausalLM.from_pretrained(d, device_map=self.device_map)
        self.model = loaded
        # Enable accelerations on the loaded profile
        self._try_enable_2to4_on_model(self.model)
        self.model = self._maybe_compile_model(self.model)
        self._rebuild_pruners()
        self.active_profile_key = key

    def apply_pruning(self, action) -> None:
        """Apply pruning according to the action, architecture-aware (MHA/GQA/MQA)."""
        target = getattr(action, "target", "none")
        intensity = float(getattr(action, "intensity", 0.0) or 0.0)
        # Static profile fast-path (Phase A): swap entire model to a prebuilt profile
        if getattr(self, 'enable_static_profiles', False) and target in ("attention_heads", "none"):
            self.activate_profile(action)
            return
        
        if target == "attention_heads" and intensity > 0.0:
            layers = getattr(self.model.model, "layers", [])
            if not layers:
                print("[Engine] No layers found; cannot apply head pruning.")
                return
            cfg = getattr(self.model, "config", None)
            num_heads = getattr(cfg, "num_attention_heads", 32)
            num_kv_heads = getattr(cfg, "num_key_value_heads", None)
            if num_kv_heads is None:
                num_kv_heads = num_heads  # MHA default

            attn_type = _infer_attention_type(cfg)
            per_layer_removed_q = {}
            per_layer_removed_kv = {}

            if attn_type == "MHA":
                # MHA: num_heads == num_kv_heads — simple per-head pruning
                heads_to_remove_count = max(1, int(round(intensity * num_heads)))
                # Keep at least 25% of heads to prevent degenerate model
                min_keep = max(1, num_heads // 4)
                heads_to_remove_count = min(heads_to_remove_count, num_heads - min_keep)
                for i, layer in enumerate(layers):
                    q_head_scores = self.head_importance.get(i, torch.ones(num_heads))
                    removed = torch.topk(q_head_scores, heads_to_remove_count, largest=False).indices.tolist()
                    per_layer_removed_q[i] = sorted(removed)
                    per_layer_removed_kv[i] = sorted(removed)  # Same for MHA
                print(f"[Engine] Applying MHA STRUCTURAL head pruning: removing {heads_to_remove_count}/{num_heads} heads per layer.")
            else:
                # GQA/MQA: map Q heads to KV groups
                heads_per_group = num_heads // num_kv_heads
                if num_heads % num_kv_heads != 0:
                    print(f"[Engine] WARNING: {num_heads} Q-heads not divisible by {num_kv_heads} KV-heads. Skipping head pruning.")
                    return
                groups_to_remove_count = int(round(intensity * num_kv_heads))
                if groups_to_remove_count <= 0 and intensity > 0.0:
                    groups_to_remove_count = 1
                # Keep at least 25% of KV groups
                min_keep_groups = max(1, num_kv_heads // 4)
                groups_to_remove_count = min(groups_to_remove_count, num_kv_heads - min_keep_groups)
                for i, layer in enumerate(layers):
                    q_head_scores = self.head_importance.get(i, torch.ones(num_heads))
                    group_scores = q_head_scores.view(num_kv_heads, heads_per_group).mean(dim=1)
                    kv_heads_to_remove = torch.topk(group_scores, groups_to_remove_count, largest=False).indices.tolist()
                    q_heads_to_remove = []
                    for kv_idx in kv_heads_to_remove:
                        start_q_idx = kv_idx * heads_per_group
                        q_heads_to_remove.extend(range(start_q_idx, start_q_idx + heads_per_group))
                    per_layer_removed_q[i] = sorted(q_heads_to_remove)
                    per_layer_removed_kv[i] = sorted(kv_heads_to_remove)
                print(f"[Engine] Applying {attn_type} STRUCTURAL head pruning: removing {groups_to_remove_count}/{num_kv_heads} KV groups per layer.")
            self.structured_head.prune(per_layer_removed_q, per_layer_removed_kv)

        elif target == "ffn_neurons" and intensity > 0.0:
            layers = getattr(self.model.model, "layers", [])
            if not layers:
                print("[Engine] No layers found; cannot apply FFN pruning.")
                return
            mlp0 = getattr(layers[0], "mlp", None)
            if mlp0 is None or not hasattr(mlp0, "down_proj"):
                print("[Engine] MLP structure not recognized; skipping FFN pruning.")
                return
            inter_size = mlp0.down_proj.weight.shape[1]
            # Compute hardware-friendly target intermediate size (aligned to 128)
            multiple = 128
            target_size = int(round(inter_size * (1.0 - intensity)))
            new_inter = max(multiple, int(round(target_size / multiple) * multiple))
            new_inter = min(inter_size - 1, new_inter)
            k = max(1, inter_size - new_inter)
            per_layer_removed = {}
            for i, layer in enumerate(layers):
                if i in self.ffn_importance and self.ffn_importance[i].shape[0] == inter_size:
                    scores = self.ffn_importance[i]
                    removed = torch.topk(scores, k, largest=False).indices.tolist()
                else:
                    # Wanda-style fallback: weight magnitude
                    mlp = layer.mlp
                    with torch.no_grad():
                        g_scores = torch.linalg.vector_norm(mlp.gate_proj.weight, ord=2, dim=1)
                        u_scores = torch.linalg.vector_norm(mlp.up_proj.weight, ord=2, dim=1)
                        d_scores = torch.linalg.vector_norm(mlp.down_proj.weight, ord=2, dim=0)
                        scores = g_scores + u_scores + d_scores
                        removed = torch.topk(scores, k, largest=False).indices.tolist()
                per_layer_removed[i] = removed
            print(f"[Engine] Applying STRUCTURAL FFN pruning: {inter_size} -> {new_inter} neurons ({intensity*100:.0f}% reduction)")
            self.structured_ffn.prune(per_layer_removed)

        elif target == "transformer_layers" and intensity > 0.0:
            layers = getattr(self.model.model, "layers", [])
            L = len(layers)
            if L == 0: return
            k = max(1, int(round(L * intensity)))
            # Safety: keep at least 2 layers (first + last minimum)
            max_removable = max(0, L - 2)
            k = min(k, max_removable)
            if k <= 0:
                print(f"[Engine] Cannot skip layers: only {L} layers, minimum 2 required.")
                return
            
            if self.layer_importance and len(self.layer_importance) == L:
                scores = torch.tensor(self.layer_importance)
                to_skip = torch.topk(scores, k, largest=False).indices.tolist()
            else:
                to_skip = list(range(L - k, L))
            
            # Never skip the first or last layers of the model
            to_skip_safe = [idx for idx in to_skip if idx != 0 and idx != L-1]
            
            if to_skip_safe:
                print(f"[Engine] Skipping {len(to_skip_safe)}/{L} layers (reversible): {to_skip_safe}")
                self.layer_skipper.apply(to_skip_safe)
            
        elif target == "none":
            self.restore_model()

    def extract_early_features(self, prompt: str) -> dict:
        """Run only embedding + layer-0 of Llama to extract cheap runtime features.

        Returns dict with keys: hidden_norm, attn_entropy, attn_max
        All values are scalars suitable for RL state vector.
        Must be called BEFORE any pruning is applied.
        """
        import math
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            # 1. Embedding
            hidden = self.model.model.embed_tokens(input_ids)

            # 2. Run only layer 0
            layer_0 = self.model.model.layers[0]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # Compute rotary embeddings (required by transformers >= 4.45 where
            # rotary_emb moved from per-layer attention to the parent LlamaModel)
            position_embeddings = None
            if hasattr(self.model.model, 'rotary_emb'):
                cos, sin = self.model.model.rotary_emb(hidden, position_ids)
                position_embeddings = (cos, sin)
            elif hasattr(layer_0.self_attn, 'rotary_emb'):
                cos, sin = layer_0.self_attn.rotary_emb(hidden, position_ids)
                position_embeddings = (cos, sin)

            # Build kwargs for the layer call
            layer_kwargs = dict(
                attention_mask=None,
                position_ids=position_ids,
                output_attentions=True,
            )
            if position_embeddings is not None:
                layer_kwargs['position_embeddings'] = position_embeddings

            try:
                layer_out = layer_0(hidden, **layer_kwargs)
                hidden_out = layer_out[0]
                attn_weights = layer_out[1]  # may be None with SDPA backend
            except Exception:
                # Fallback: run without requesting attention weights
                layer_kwargs['output_attentions'] = False
                layer_out = layer_0(hidden, **layer_kwargs)
                hidden_out = layer_out[0]
                attn_weights = None

            # Feature 1: mean L2 norm of hidden states (normalised by hidden_dim)
            hidden_dim = hidden_out.shape[-1]  # 2048 for Llama-3.2-1B
            token_norms = torch.linalg.vector_norm(hidden_out, ord=2, dim=-1)  # (1, seq_len)
            hidden_norm = float(token_norms.mean()) / hidden_dim

            if attn_weights is not None:
                # Feature 2: mean attention entropy (normalised by log(seq_len))
                eps = 1e-9
                log_attn = torch.log(attn_weights + eps)
                entropy = -(attn_weights * log_attn).sum(dim=-1)  # (1, heads, seq_len)
                max_entropy = math.log(max(seq_len, 2))
                attn_entropy = float(entropy.mean()) / max_entropy

                # Feature 3: mean attention max (already in [0, 1])
                attn_max = float(attn_weights.max(dim=-1).values.mean())
            else:
                # Derive proxy features from hidden states when SDPA hides attn weights
                token_vecs = hidden_out.squeeze(0)  # (seq_len, hidden_dim)
                norms = token_norms.squeeze(0)       # (seq_len,)
                # Proxy entropy: coefficient of variation of token norms, capped to [0, 1]
                attn_entropy = min(1.0, float(norms.std() / (norms.mean() + 1e-9)))
                # Proxy concentration: 1 - (mean/max) norm ratio — 0=uniform, 1=one token dominates
                attn_max = 1.0 - float(norms.mean() / (norms.max() + 1e-9))

        return {
            "hidden_norm": hidden_norm,
            "attn_entropy": attn_entropy,
            "attn_max": attn_max,
        }

    def restore_model(self) -> None:
        # Restore functional masks
        self.head_pruner.restore()
        self.ffn_pruner.restore()
        self.layer_skipper.restore()
        # Restore structural slicing (if any)
        self.structured_ffn.restore()
        self.structured_head.restore()

    def calibrate_importances(self, prompts, max_samples: int = 64, max_seq_len: int = 128) -> None:
        """Calibrate structural importance scores using established academic methods:
        - Heads: Activation-based Head Sensitivity (Michel et al., 2019)
        - FFN: Weight-Activation Product inspired by Wanda (Sun et al., 2023)
        - Layers: Block Influence (BI) from ShortGPT (Men et al., 2024) —
          BI(i) = 1 - cosine_similarity(layer_input, layer_output).
          Low BI → layer barely transforms hidden state → safe to remove.
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
        # Block Influence accumulators for layers
        bi_cos_sum = {}   # layer_idx -> cumulative cosine similarity
        bi_count = {}     # layer_idx -> token count

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
                attn = layer.self_attn
                def make_head_hook(idx):
                    def pre_hook(module, inputs):
                        nonlocal total_tokens
                        if not inputs or inputs[0] is None:
                            return inputs
                        x = inputs[0]
                        B, T, D = x.shape if x.dim() == 3 else (1, x.shape[0], x.shape[-1])
                        total_tokens += (B*T)
                        try:
                            x_ = x.view(B*T, num_heads, head_dim)
                        except Exception:
                            return inputs
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
                        x = inputs[0]
                        if x.dim() == 3:
                            B, T, C = x.shape
                            total_tokens += (B * T)
                            vals = x.abs().sum(dim=(0, 1))
                        else:
                            total_tokens += x.shape[0] if x.dim() > 0 else 1
                            vals = x.abs().sum(dim=0)
                        ffn_acc[idx][:] += vals.to(ffn_acc[idx].device)
                        return inputs
                    return pre_hook
                hooks.append(mlp.down_proj.register_forward_pre_hook(make_ffn_hook(i)))

        # Block Influence hooks: capture layer input and output for cosine similarity
        for i, layer in enumerate(layers):
            bi_cos_sum[i] = 0.0
            bi_count[i] = 0
            def make_bi_hook(idx):
                def fwd_hook(module, inputs, output):
                    try:
                        inp = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
                        out = output[0] if isinstance(output, (tuple, list)) else output
                        if inp is None or out is None:
                            return
                        # Flatten to (num_tokens, hidden_dim) for cosine similarity
                        inp_flat = inp.detach().reshape(-1, inp.shape[-1]).float()
                        out_flat = out.detach().reshape(-1, out.shape[-1]).float()
                        # Per-token cosine similarity, then average
                        cos = torch.nn.functional.cosine_similarity(inp_flat, out_flat, dim=-1)
                        bi_cos_sum[idx] += float(cos.sum().item())
                        bi_count[idx] += cos.shape[0]
                    except Exception:
                        pass
                return fwd_hook
            hooks.append(layer.register_forward_hook(make_bi_hook(i)))

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
        # Block Influence layer importance: BI = 1 - mean_cosine_similarity
        # Low BI → layer is redundant → low importance → safe to skip
        if used > 0 and layers:
            L = len(layers)
            self.layer_importance = []
            for i in range(L):
                cos_avg = bi_cos_sum.get(i, 0.0) / max(1.0, float(bi_count.get(i, 1)))
                bi_score = 1.0 - cos_avg  # Block Influence
                self.layer_importance.append(bi_score)
            # Log BI scores for transparency
            bi_str = ", ".join(f"L{i}:{bi:.4f}" for i, bi in enumerate(self.layer_importance))
            print(f"[Calib] Block Influence scores: {bi_str}")
        # Retain a copy of prompts for later reconstruction passes (Phase C)
        try:
            self._calib_prompts_recent = list(prompts[:max_samples]) if prompts else []
        except Exception:
            self._calib_prompts_recent = []
        print(f"[Calib] Completed on {used} samples. Head importances: {len(self.head_importance)} layers, FFN importances: {len(self.ffn_importance)} layers, Layer importances: {len(self.layer_importance)}.")

    def _collect_layer_output_sums(self, model: AutoModelForCausalLM, prompts, max_samples: int = 32, max_seq_len: int = 128) -> dict:
        """Collect per-layer output |activation| sums over a small prompt set for gain estimation."""
        layers = getattr(model.model, 'layers', [])
        if not layers:
            return {}
        sums = {i: 0.0 for i in range(len(layers))}
        hooks = []
        try:
            def make_layer_hook(idx):
                def fwd_hook(module, inputs, output):
                    try:
                        hs = output[0] if isinstance(output, (tuple, list)) else output
                        if hs is None:
                            return
                        if hasattr(hs, 'abs'):
                            sums[idx] += float(hs.detach().abs().sum().item())
                    except Exception:
                        pass
                return fwd_hook
            for i, layer in enumerate(layers):
                hooks.append(layer.register_forward_hook(make_layer_hook(i)))
            model.eval()
            used = 0
            with torch.no_grad():
                for p in prompts[:max_samples]:
                    tok = self.tokenizer(p, return_tensors='pt', truncation=True, max_length=max_seq_len).to(model.device)
                    try:
                        _ = model(**tok)
                        used += 1
                    except Exception:
                        continue
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
        return sums

    def generate_response(self, prompt: str, max_length: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        # Enforce fixed-length generation for fair throughput/ppl comparisons.
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    min_length=input_len + max_length,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=None,  # avoid early stop on EOS
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except TypeError:
            # Fallback if min_length unsupported: rely on min_new_tokens and disabling EOS
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        min_new_tokens=max_length,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=None,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            except TypeError:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=None,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def generate_with_inputs(self, inputs, max_length: int = 50):
        """Generate using precomputed tokenized inputs to avoid re-tokenization in benchmarks."""
        if self.enable_kv_compression and self.kv_keep_ratio < 0.999 and not self._kv_warned:
            print("[KV] KV-cache compression scaffold enabled (no-op placeholder).")
            self._kv_warned = True
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else None
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    min_length=(input_len + max_length) if input_len is not None else None,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except TypeError:
            # Fallbacks for older versions
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        min_new_tokens=max_length,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=None,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            except TypeError:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=None,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
        return outputs

    def save_pretrained(self, out_dir: str) -> None:
        """Persist current model/tokenizer (including structural pruning) for external eval (e.g., lm-eval-harness)."""
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        print(f"[Engine] Model and tokenizer saved to {out_dir}")
