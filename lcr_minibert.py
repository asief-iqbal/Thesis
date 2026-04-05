"""MiniBERT-based Learned Complexity Router (LCR) scorer — Multi-Output v2.

This module provides the core LCR component that replaces hand-crafted
"prompt complexity" equations with learned sensitivity scores in [0, 1].

Architecture (v2 — multi-output with auxiliary features):
- Backbone: prajjwal1/bert-mini (4 layers, 256 hidden, 11.3M params)
  → Latency: ~3ms on CPU, <1ms on GPU (negligible vs Llama-3.2-1B)
- Auxiliary features: 6 statistical text features → Linear(6→32) → 32-dim
- Regression head: [256+32=288] → 128 → 64 → n_outputs (GELU + Dropout)
- Output: sigmoid-squashed sensitivity scores in [0, 1]

Multi-output mode (n_outputs=2):
  output[0] = head_sensitivity  (sensitivity to attention head pruning)
  output[1] = layer_sensitivity (sensitivity to layer skipping)

Single-output mode (n_outputs=1, backward compatible):
  output[0] = composite_sensitivity

The score() method returns max(head, layer) for backward compatibility
with the RL controller's single-score state vector.

Training pipeline: oracle_labeler.py → prepare_dual_labels.py → train_minibert_lcr.py → this module
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from transformers import AutoModel, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoModel = None
    AutoTokenizer = None
    _TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Auxiliary text feature extraction (lightweight, ~0.1ms)
# ---------------------------------------------------------------------------

_CODE_PATTERN = re.compile(
    r'[{}[\]<>;]|\b(def|class|return|if|else|for|while|import|print|lambda|function|var|let|const)\b|==|!=|>=|<='
)
_MATH_PATTERN = re.compile(
    r'[+\-*/=^\\]|\b(sin|cos|tan|log|sqrt|pi|sum|int|solve|equation)\b|\d+\.?\d*'
)


def compute_aux_features(text: str) -> List[float]:
    """Compute 9 lightweight statistical features from raw text.

    These features provide explicit signals about text complexity and domain
    that complement BERT embeddings. Each feature is normalized to ~[0, 1].

    Returns:
        [log_token_count, compression_ratio, avg_word_length,
         special_char_ratio, unique_token_ratio, has_code_markers,
         numeric_density, has_question, avg_sentence_len]
    """
    if not text:
        return [0.0] * 9

    words = text.split()
    n_words = max(1, len(words))
    n_chars = max(1, len(text))

    # 1. Log token count (normalized: log(1)=0 → log(500)≈6.2 → /7 ≈ 0.9)
    log_token_count = min(1.0, math.log1p(n_words) / 7.0)

    # 2. Compression ratio (typically 0.15–0.80; lower = more redundant)
    try:
        data = text.encode("utf-8", errors="ignore")
        compression_ratio = len(zlib.compress(data)) / max(1, len(data)) if data else 0.5
    except Exception:
        compression_ratio = 0.5

    # 3. Average word length (normalized: /12, typical range 3–8)
    avg_word_length = min(1.0, sum(len(w) for w in words) / (n_words * 12.0))

    # 4. Special character ratio (punctuation, math, brackets)
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_char_ratio = min(1.0, special_chars / n_chars)

    # 5. Unique token ratio (vocabulary diversity)
    lower_words = [w.lower() for w in words]
    unique_token_ratio = len(set(lower_words)) / n_words

    # 6. Has code markers (binary-ish, boosted if many matches)
    code_matches = len(_CODE_PATTERN.findall(text))
    has_code_markers = min(1.0, code_matches / 5.0)  # saturate at 5+ matches

    # 7. Numeric density — fraction of tokens containing digits (helps GSM8K/math)
    digit_tokens = sum(1 for w in words if any(c.isdigit() for c in w))
    numeric_density = digit_tokens / n_words

    # 8. Has question marker (boolean-ish, helps BoolQ/MMLU)
    has_question = 1.0 if "?" in text else 0.0

    # 9. Average sentence length (normalized; captures multi-step reasoning)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    avg_sentence_len = min(1.0, (n_words / max(1, len(sentences))) / 40.0)

    return [
        float(log_token_count),
        float(compression_ratio),
        float(avg_word_length),
        float(special_char_ratio),
        float(unique_token_ratio),
        float(has_code_markers),
        float(numeric_density),
        float(has_question),
        float(avg_sentence_len),
    ]


AUX_FEATURE_NAMES = [
    "log_token_count", "compression_ratio", "avg_word_length",
    "special_char_ratio", "unique_token_ratio", "has_code_markers",
    "numeric_density", "has_question", "avg_sentence_len",
]
AUX_FEATURE_DIM = len(AUX_FEATURE_NAMES)  # 9


# ---------------------------------------------------------------------------
# Config and model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MiniBertLcrConfig:
    model_name: str = "prajjwal1/bert-mini"
    max_length: int = 128
    device: str = "cpu"
    pooling: str = "mean"  # 'mean' (recommended for regression) or 'cls'
    head_checkpoint_path: Optional[str] = None
    head_dropout: float = 0.20
    n_outputs: int = 2  # 2 = dual (head_sens, layer_sens); 1 = single (composite)
    use_aux_features: bool = True  # concatenate auxiliary text features


class _AuxProjector(torch.nn.Module):
    """Projects auxiliary features into a suitable dimensionality."""

    def __init__(self, n_features: int, out_dim: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, out_dim),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ScalarMix(torch.nn.Module):
    """Learned weighted sum of BERT hidden layers (à la ELMo / ScalarMix).

    Given N hidden states [h_0, h_1, ..., h_{N-1}], computes:
        output = gamma * sum_i(softmax(w)_i * h_i)
    where w and gamma are learned scalars.  This lets the model attend to
    lexical (early), syntactic (middle), and semantic (late) representations.

    Reference: Peters et al., "Deep contextualized word representations", 2018.
    """

    def __init__(self, n_layers: int):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(n_layers))  # softmax → uniform init
        self.gamma = torch.nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """hidden_states: tuple of [B, seq, H] tensors."""
        normed_weights = torch.softmax(self.weights, dim=0)
        mixed = sum(w * h for w, h in zip(normed_weights, hidden_states))
        return self.gamma * mixed


class _AttentionStatsExtractor(torch.nn.Module):
    """Extracts statistical features from BERT attention matrices.

    For each layer and head, computes attention entropy (how diffuse the
    attention is) and max attention (how focused).  These are model-internal
    signals unavailable from surface text features alone.

    Also includes a learned linear projection per layer so the backbone
    can learn to produce more informative attention patterns via gradient flow.

    Output dim = n_layers * n_heads * 2 + proj_dim (= 4*4*2 + 16 = 48 for bert-mini).
    """

    def __init__(self, n_layers: int, n_heads: int, proj_dim: int = 16):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        raw_dim = n_layers * n_heads * 2  # entropy + max_attn per head per layer
        self.out_dim = raw_dim + proj_dim
        # Learned projection from raw attention stats
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(raw_dim, proj_dim),
            torch.nn.Tanh(),
        )

    def forward(self, attentions: Tuple[torch.Tensor, ...], attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """attentions: tuple of [B, n_heads, seq, seq] tensors.
        Returns: [B, out_dim] attention features.
        """
        feats = []
        for layer_attn in attentions:
            # layer_attn: [B, n_heads, seq, seq]
            if attention_mask is not None:
                pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()  # [B, 1, 1, seq]
                masked_attn = layer_attn * pad_mask
                masked_attn = masked_attn / masked_attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            else:
                masked_attn = layer_attn

            # Entropy: -sum(p * log(p+eps))
            eps = 1e-8
            entropy = -(masked_attn * torch.log(masked_attn + eps)).sum(dim=-1)  # [B, n_heads, seq]
            # Max attention per query position
            max_attn = masked_attn.max(dim=-1).values  # [B, n_heads, seq]

            if attention_mask is not None:
                query_mask = attention_mask.unsqueeze(1).float()  # [B, 1, seq]
                entropy = (entropy * query_mask).sum(dim=-1) / query_mask.sum(dim=-1).clamp(min=1)
                max_attn = (max_attn * query_mask).sum(dim=-1) / query_mask.sum(dim=-1).clamp(min=1)
            else:
                entropy = entropy.mean(dim=-1)  # [B, n_heads]
                max_attn = max_attn.mean(dim=-1)  # [B, n_heads]
            feats.append(entropy)
            feats.append(max_attn)

        # Concatenate all layers: [B, n_layers * n_heads * 2]
        raw = torch.cat(feats, dim=-1)
        # Normalize entropy by log(seq_len) to get relative entropy in ~[0, 1]
        seq_len = attentions[0].shape[-1]
        raw = raw / max(1.0, math.log(seq_len + 1))
        # Learned projection allows gradient flow back to backbone attention
        proj = self.proj(raw)
        return torch.cat([raw, proj], dim=-1)


class _RegressorHead(torch.nn.Module):
    """Regression head with LayerNorm and wider intermediate layers.

    v3 architecture (single-output recommended):
      input_dim → LayerNorm → 192 → GELU → Dropout → 96 → GELU → Dropout → n_outputs → Sigmoid

    v2 backward compatibility maintained via flexible sizing.
    """

    def __init__(self, input_dim: int, dropout: float = 0.20, n_outputs: int = 1):
        super().__init__()
        mid = max(64, min(256, input_dim * 2 // 3))  # ~192 for 288-input
        low = max(32, mid // 2)                        # ~96
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, mid),
            torch.nn.GELU(),
            torch.nn.Dropout(p=float(dropout)),
            torch.nn.Linear(mid, low),
            torch.nn.GELU(),
            torch.nn.Dropout(p=float(dropout) * 0.5),
            torch.nn.Linear(low, n_outputs),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MiniBertLcrScorer:
    """Returns LCR sensitivity/complexity scores in [0, 1] for a prompt.

    In multi-output mode (n_outputs=2):
      score()      → max(head_sensitivity, layer_sensitivity)  [backward compat]
      score_dual() → (head_sensitivity, layer_sensitivity) tuple
    """

    def __init__(self, config: Optional[MiniBertLcrConfig] = None):
        self.config = config or MiniBertLcrConfig()

        self._tokenizer = None
        self._model = None
        self._head: Optional[_RegressorHead] = None
        self._aux_proj: Optional[_AuxProjector] = None
        self._scalar_mix: Optional[_ScalarMix] = None
        self._attn_extractor: Optional[_AttentionStatsExtractor] = None
        self._hidden_size: Optional[int] = None
        self._head_loaded: bool = False  # tracks if fine-tuned head weights were loaded

        self._enabled = _TRANSFORMERS_AVAILABLE
        if not self._enabled:
            print("[LCR] transformers not available; MiniBERT scorer disabled.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _lazy_init(self) -> None:
        if not self._enabled:
            return
        if self._model is not None and self._tokenizer is not None:
            return

        device = torch.device(self.config.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        # Prefer safetensors to avoid torch.load(.bin) restrictions on older Torch versions.
        try:
            self._model = AutoModel.from_pretrained(self.config.model_name, use_safetensors=True)
        except Exception:
            self._model = AutoModel.from_pretrained(self.config.model_name)
        self._model.to(device)
        self._model.eval()

        hidden_size = int(getattr(self._model.config, "hidden_size", 256))
        self._hidden_size = hidden_size

        # Determine BERT architecture for ScalarMix and attention extractor
        n_bert_layers = int(getattr(self._model.config, "num_hidden_layers", 4))
        n_bert_heads = int(getattr(self._model.config, "num_attention_heads", 4))

        # ScalarMix: learned weighted sum over all hidden states (embedding + layers)
        self._scalar_mix = _ScalarMix(n_layers=n_bert_layers + 1)
        self._scalar_mix.to(device)
        self._scalar_mix.eval()

        # Attention stats extractor: entropy + max_attn + learned projection
        self._attn_extractor = _AttentionStatsExtractor(
            n_layers=n_bert_layers, n_heads=n_bert_heads
        )
        self._attn_extractor.to(device)
        self._attn_extractor.eval()
        attn_stats_dim = self._attn_extractor.out_dim

        # Auxiliary feature projector (text features + attention stats)
        if self.config.use_aux_features:
            total_aux_dim = AUX_FEATURE_DIM + attn_stats_dim
            self._aux_proj = _AuxProjector(total_aux_dim, out_dim=48)
            self._aux_proj.to(device)
            self._aux_proj.eval()
            input_dim = hidden_size + 48
        else:
            self._aux_proj = None
            input_dim = hidden_size

        # Regression head
        self._head = _RegressorHead(
            input_dim=input_dim,
            dropout=float(self.config.head_dropout),
            n_outputs=int(self.config.n_outputs),
        )
        self._head.to(device)
        self._head.eval()

        ckpt_path = self.config.head_checkpoint_path
        if ckpt_path:
            ckpt = Path(ckpt_path)
            if ckpt.exists():
                try:
                    try:
                        state = torch.load(str(ckpt), map_location=device, weights_only=True)
                    except TypeError:
                        state = torch.load(str(ckpt), map_location=device)
                    # Support either raw state_dict or wrapped dict.
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    # Separate checkpoint keys by prefix.
                    head_state = {}
                    aux_state = {}
                    scalar_mix_state = {}
                    attn_extractor_state = {}
                    for k, v in state.items():
                        if k.startswith("aux_proj."):
                            aux_state[k[len("aux_proj."):]] = v
                        elif k.startswith("scalar_mix."):
                            scalar_mix_state[k[len("scalar_mix."):]] = v
                        elif k.startswith("attn_extractor."):
                            attn_extractor_state[k[len("attn_extractor."):]] = v
                        else:
                            # Strip 'head.' prefix if present
                            clean_key = k[len("head."):] if k.startswith("head.") else k
                            head_state[clean_key] = v

                    self._head.load_state_dict(head_state, strict=True)
                    if self._aux_proj is not None and aux_state:
                        self._aux_proj.load_state_dict(aux_state, strict=True)
                    if scalar_mix_state:
                        self._scalar_mix.load_state_dict(scalar_mix_state, strict=True)
                    if attn_extractor_state:
                        self._attn_extractor.load_state_dict(attn_extractor_state, strict=True)
                    self._head_loaded = True
                    print(f"[LCR] Loaded regression head from: {ckpt} (n_outputs={self.config.n_outputs})")
                except Exception as e:
                    # Try loading with strict=False for backward compat (old single → new dual)
                    try:
                        self._head.load_state_dict(state, strict=False)
                        self._head_loaded = True
                        print(f"[LCR] Loaded regression head (partial) from: {ckpt}")
                    except Exception:
                        print(f"[LCR] Warning: failed to load head checkpoint {ckpt}: {e}")
            else:
                print(f"[LCR] Head checkpoint not found: {ckpt} (using fallback proxy)")
        else:
            print("[LCR] No head checkpoint configured (using fallback proxy).")

    @staticmethod
    def _compression_ratio(text: str) -> float:
        if not text:
            return 0.0
        data = text.encode("utf-8", errors="ignore")
        if not data:
            return 0.0
        return len(zlib.compress(data)) / float(len(data))

    @staticmethod
    def _fallback_proxy_score(prompt: str, cls_embedding: torch.Tensor) -> float:
        """Deterministic proxy when no trained head is available."""
        comp = MiniBertLcrScorer._compression_ratio(prompt)
        heuristic = 1.0 - max(0.0, min(1.0, (comp - 0.15) / 0.35))
        rep_std = float(torch.std(cls_embedding).detach().cpu().item())
        rep_term = 1.0 / (1.0 + math.exp(-(rep_std - 0.6) * 6.0))
        length_term = min(1.0, len(prompt) / 512.0)
        score = 0.45 * heuristic + 0.35 * rep_term + 0.20 * length_term
        return float(max(0.0, min(1.0, score)))

    def _forward(self, prompt: str) -> torch.Tensor:
        """Run BERT + ScalarMix + attention features + aux features + head; return raw output tensor [1, n_outputs]."""
        """Run BERT + ScalarMix + attention features + aux features + head; return raw output tensor [1, n_outputs]."""
        self._lazy_init()
        if self._model is None or self._tokenizer is None or self._head is None:
            return torch.tensor([[0.5] * self.config.n_outputs])

        device = torch.device(self.config.device)
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.config.max_length),
            padding=False,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self._model(
            **encoded, return_dict=True,
            output_hidden_states=True, output_attentions=True,
        )

        # ScalarMix: learned weighted sum over all hidden states
        if self._scalar_mix is not None and outputs.hidden_states is not None:
            mixed = self._scalar_mix(outputs.hidden_states)
        else:
            mixed = outputs.last_hidden_state

        # Mean pooling
        if self.config.pooling == "mean":
            attention_mask = encoded.get("attention_mask")
            pooled = self._mean_pool(mixed, attention_mask)
        else:
            pooled = mixed[:, 0, :]  # CLS token

        # Concatenate auxiliary features + attention stats if enabled
        if self._aux_proj is not None:
            # Text-based auxiliary features
            aux_feats = torch.tensor(
                [compute_aux_features(prompt)], dtype=torch.float32, device=device
            )
            # Attention statistics from BERT internal states
            if self._attn_extractor is not None and outputs.attentions is not None:
                attn_stats = self._attn_extractor(
                    outputs.attentions, encoded.get("attention_mask")
                )
                combined_aux = torch.cat([aux_feats, attn_stats], dim=-1)
            else:
                combined_aux = aux_feats
            aux_emb = self._aux_proj(combined_aux)  # [1, 48]
            pooled = torch.cat([pooled, aux_emb], dim=-1)  # [1, 256+48=304]

        # Get predictions from regression head
        pred = self._head(pooled)  # [1, n_outputs]
        return torch.clamp(pred, 0.0, 1.0)

    @torch.no_grad()
    def score(self, prompt: str) -> float:
        """Compute a single LCR score in [0, 1] for the RL controller.

        In dual-output mode: returns max(head_sensitivity, layer_sensitivity).
        This is the conservative estimate — the most sensitive method wins,
        biasing the RL agent toward less aggressive pruning when uncertain.

        In single-output mode: returns the composite sensitivity directly.
        """
        prompt = prompt or ""
        if not self._enabled:
            return 0.5

        self._lazy_init()
        if not self._head_loaded:
            # Fallback proxy
            if self._model is None or self._tokenizer is None:
                return 0.5
            device = torch.device(self.config.device)
            encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True,
                                       max_length=int(self.config.max_length), padding=False)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = self._model(**encoded, return_dict=True)
            pooled = self._mean_pool(outputs.last_hidden_state, encoded.get("attention_mask"))
            return self._fallback_proxy_score(prompt, pooled.squeeze(0))

        pred = self._forward(prompt)  # [1, n_outputs]
        if self.config.n_outputs >= 2:
            return float(torch.max(pred).item())
        return float(pred[0, 0].item())

    @torch.no_grad()
    def score_dual(self, prompt: str) -> Tuple[float, float]:
        """Return (head_sensitivity, layer_sensitivity) for advanced RL integration.

        Only meaningful when n_outputs >= 2. Falls back to (score, score) otherwise.
        """
        prompt = prompt or ""
        if not self._enabled or not self._head_loaded:
            s = self.score(prompt)
            return (s, s)

        pred = self._forward(prompt)  # [1, n_outputs]
        if self.config.n_outputs >= 2:
            return (float(pred[0, 0].item()), float(pred[0, 1].item()))
        s = float(pred[0, 0].item())
        return (s, s)

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Mean-pool token embeddings, masking padding tokens."""
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
        sum_embeddings = (last_hidden * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask
