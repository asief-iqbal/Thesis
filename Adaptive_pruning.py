#!/usr/bin/env python3
"""
RL-Driven Adaptive Pruning System
Implements reinforcement learning for dynamic LLM pruning based on hardware state and prompt complexity.
"""

# --- Dependencies ---
# pip install torch transformers psutil numpy accelerate nvidia-ml-py datasets lm-eval matplotlib

# --- Standard Library Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import time
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt
import argparse
import math
import os
import re
import itertools
import warnings
import csv

try:
    from lcr_minibert import MiniBertLcrConfig, MiniBertLcrScorer
    _MINIBERT_LCR_AVAILABLE = True
except Exception:
    MiniBertLcrConfig = None
    MiniBertLcrScorer = None
    _MINIBERT_LCR_AVAILABLE = False
try:
    import pynvml as nvml
    _NVML_AVAILABLE = True
    try:
        nvml.nvmlInit()
    except Exception:
        _NVML_AVAILABLE = False
except Exception:
    try:
        import nvidia_ml_py as nvml
        _NVML_AVAILABLE = True
        try:
            nvml.nvmlInit()
        except Exception:
            _NVML_AVAILABLE = False
    except Exception:
        _NVML_AVAILABLE = False

warnings.filterwarnings("ignore")

try:
    from model_loader import RealModelEngine
    print("[System] Model components loaded successfully.")
except ImportError as e:
    print(f"[Error] Failed to load model_loader.py: {e}")
    exit(1)

print("RL-DRIVEN ADAPTIVE PRUNING SYSTEM")
print("="*80)
# Reproducibility seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEFAULT_EPSILON_START = 1.0
DEFAULT_EPSILON_END = 0.10


def compute_epsilon_decay(
    total_episodes: int,
    epsilon_start: float = DEFAULT_EPSILON_START,
    epsilon_end: float = DEFAULT_EPSILON_END,
) -> float:
    """Return a multiplicative decay that reaches epsilon_end at the horizon."""
    horizon = max(int(total_episodes), 1)
    epsilon_start = float(epsilon_start)
    epsilon_end = float(epsilon_end)

    if epsilon_start <= 0.0 or epsilon_end <= 0.0:
        raise ValueError("Epsilon schedule requires positive start/end values.")
    if epsilon_end >= epsilon_start:
        return 1.0

    return math.exp(math.log(epsilon_end / epsilon_start) / horizon)

# =========================================================================
# DEVICE MONITORING AND STATE REPRESENTATION
# =========================================================================

@dataclass
class DeviceState:
    cpu_utilization: float
    memory_available_gb: float
    battery_percent: float
    gpu_available: bool
    gpu_memory_free_gb: float
    gpu_utilization: float

class EnhancedDeviceMonitor:
    """Monitors hardware state (CPU, GPU, memory, battery) for RL state features."""
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.nvml_ok = False
        self._last_state: Optional[DeviceState] = None
        self._last_state_ts: float = 0.0
        self._cache_ttl_s: float = 0.5
        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_ok = True
            except Exception:
                pass
        print(f"[Monitor] GPU {'detected' if self.gpu_available else 'not detected' }.")

    def get_state(self) -> DeviceState:
        now = time.time()
        if self._last_state is not None and (now - self._last_state_ts) < self._cache_ttl_s:
            return self._last_state

        # Non-blocking CPU sampling to avoid adding latency to RL decision making.
        cpu_util = psutil.cpu_percent(interval=None)
        mem_free_gb = psutil.virtual_memory().available / (1024 ** 3)
        batt = psutil.sensors_battery().percent if psutil.sensors_battery() else 100.0
        if self.nvml_ok:
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.handle)
                gpu_mem_free_gb = mem_info.free / (1024 ** 3)
                gpu_util = float(util)
            except Exception:
                warnings.warn("[Monitor] NVML failed, setting GPU utilization to 0.0 for reproducible research.")
                gpu_util = 0.0
                gpu_mem_free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3) if self.gpu_available else 0.0
        else:
            warnings.warn("[Monitor] NVML not available, setting GPU utilization to 0.0 for reproducible research.")
            gpu_util = 0.0
            gpu_mem_free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3) if self.gpu_available else 0.0
        state = DeviceState(
            cpu_utilization=cpu_util,
            memory_available_gb=mem_free_gb,
            battery_percent=batt,
            gpu_available=self.gpu_available,
            gpu_memory_free_gb=gpu_mem_free_gb,
            gpu_utilization=gpu_util
        )
        self._last_state = state
        self._last_state_ts = now
        return state

@dataclass
class PruningAction:
    level: int
    intensity: float
    target: str
    action_index: int

class ActionSpace:
    """Defines all possible pruning actions: none, layers, heads, and FFN neurons.

    Every intensity maps to a **mechanically distinct** outcome for
    Llama-3.2-1B (16 layers, 8 KV heads), eliminating duplicate actions
    that previously confused the RL policy.
    """
    def __init__(self):
        self.actions = [
            PruningAction(level=0, intensity=0.0,   target="none",               action_index=0),
            # Transformer layer skipping — each intensity removes a distinct
            # number of layers from the 16-layer backbone:
            #   0.06→1L  0.12→2L  0.19→3L  0.25→4L  0.31→5L
            #   0.38→6L  0.44→7L  0.50→8L  0.56→9L  0.62→10L
            PruningAction(level=3, intensity=0.06,  target="transformer_layers", action_index=1),
            PruningAction(level=3, intensity=0.12,  target="transformer_layers", action_index=2),
            PruningAction(level=3, intensity=0.19,  target="transformer_layers", action_index=3),
            PruningAction(level=3, intensity=0.25,  target="transformer_layers", action_index=4),
            PruningAction(level=3, intensity=0.31,  target="transformer_layers", action_index=5),
            PruningAction(level=3, intensity=0.38,  target="transformer_layers", action_index=6),
            PruningAction(level=3, intensity=0.44,  target="transformer_layers", action_index=7),
            PruningAction(level=3, intensity=0.50,  target="transformer_layers", action_index=8),
            PruningAction(level=3, intensity=0.56,  target="transformer_layers", action_index=9),
            PruningAction(level=3, intensity=0.62,  target="transformer_layers", action_index=10),
            # Structural attention-head pruning (GQA-safe) — each intensity
            # removes a distinct number of KV groups from the 8 KV heads:
            #   0.125→1KV  0.25→2KV  0.375→3KV  0.50→4KV  0.625→5KV  0.75→6KV
            PruningAction(level=2, intensity=0.125, target="attention_heads",    action_index=11),
            PruningAction(level=2, intensity=0.25,  target="attention_heads",    action_index=12),
            PruningAction(level=2, intensity=0.375, target="attention_heads",    action_index=13),
            PruningAction(level=2, intensity=0.50,  target="attention_heads",    action_index=14),
            PruningAction(level=2, intensity=0.625, target="attention_heads",    action_index=15),
            PruningAction(level=2, intensity=0.75,  target="attention_heads",    action_index=16),
        ]
        print(f"[RL Agent] Action space initialized with {len(self.actions)} actions "
              f"(1 none + 10 layer-skip + 6 head).")

    def get_action(self, index: int) -> PruningAction:
        return self.actions[index]

class DQN(nn.Module):
    """Simple Deep Q-Network with two hidden layers for RL policy."""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)

# =========================================================================
# REINFORCEMENT LEARNING AGENT (DOUBLE DQN CONTROLLER)
# =========================================================================

class RLControllerAgent:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.device_monitor = EnhancedDeviceMonitor()
        self.action_space = ActionSpace()

        self.lcr_scorer = None
        if _MINIBERT_LCR_AVAILABLE:
            try:
                backbone_dir = os.path.join("checkpoints", "minibert_lcr_backbone")
                model_name = backbone_dir if os.path.isdir(backbone_dir) else "prajjwal1/bert-mini"
                self.lcr_scorer = MiniBertLcrScorer(
                    MiniBertLcrConfig(
                        model_name=model_name,
                        max_length=128,
                        device="cpu",
                        head_checkpoint_path=os.path.join("checkpoints", "minibert_lcr_head.pt"),
                    )
                )
                if not self.lcr_scorer.enabled:
                    self.lcr_scorer = None
            except Exception as e:
                print(f"[LCR] Warning: MiniBERT LCR scorer init failed, using heuristic complexity. Error: {e}")
                self.lcr_scorer = None

        self.state_dim = 10  # 6 hardware + 1 LCR + 3 early-Llama features
        self.action_dim = len(self.action_space.actions)
        
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        
        # Exploration is configured per run using the actual episode horizon.
        self.epsilon = DEFAULT_EPSILON_START
        self.epsilon_decay = 1.0
        self.epsilon_min = DEFAULT_EPSILON_END
        self.gamma = 0.95

        # UCB exploration bonus: encourages under-visited actions
        self.action_counts = np.zeros(self.action_dim, dtype=np.float64)
        self.total_action_count = 0
        self.ucb_c = 1.0  # UCB exploration coefficient

        # Replay buffer and training params
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.target_update_interval = 200  # steps
        self.train_steps = 0

    def configure_epsilon_schedule(self, total_episodes: int):
        horizon = max(int(total_episodes), 1)
        self.epsilon = DEFAULT_EPSILON_START
        self.epsilon_decay = compute_epsilon_decay(
            horizon,
            epsilon_start=self.epsilon,
            epsilon_end=self.epsilon_min,
        )
        print(
            f"[System] Epsilon schedule configured: {self.epsilon:.3f} -> {self.epsilon_min:.3f} "
            f"over {horizon} episodes (decay={self.epsilon_decay:.6f})"
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_counts': self.action_counts.tolist(),
            'total_action_count': self.total_action_count,
        }, path)
        print(f"[RL Agent] Checkpoint saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        # Guard: warn and skip if state_dim changed (requires fresh training)
        saved_dim = ckpt.get('state_dim', None)
        if saved_dim is not None and saved_dim != self.state_dim:
            print(f"[RL Agent] WARNING: Checkpoint state_dim={saved_dim} != current state_dim={self.state_dim}. "
                  f"Skipping load — start fresh training with the new state vector.")
            return
        saved_action_dim = ckpt.get('action_dim', None)
        if saved_action_dim is not None and saved_action_dim != self.action_dim:
            print(f"[RL Agent] WARNING: Checkpoint action_dim={saved_action_dim} != current action_dim={self.action_dim}. "
                  f"Loading with strict=False — retrain recommended for new action space.")
        # Load policy net with tolerance for action_dim changes (strict=False)
        try:
            self.policy_net.load_state_dict(ckpt['policy_state'], strict=False)
        except Exception:
            # Partial load: only keys with matching shapes
            current = self.policy_net.state_dict()
            for k, v in ckpt['policy_state'].items():
                if k in current and current[k].shape == v.shape:
                    current[k] = v
            self.policy_net.load_state_dict(current, strict=False)
        # Load target net similarly if present
        if 'target_state' in ckpt:
            try:
                self.target_net.load_state_dict(ckpt['target_state'], strict=False)
            except Exception:
                current_t = self.target_net.state_dict()
                for k, v in ckpt['target_state'].items():
                    if k in current_t and current_t[k].shape == v.shape:
                        current_t[k] = v
                self.target_net.load_state_dict(current_t, strict=False)
        if 'optimizer_state' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception:
                pass
        # Restore UCB exploration counts if available
        if 'action_counts' in ckpt:
            try:
                saved_counts = np.array(ckpt['action_counts'], dtype=np.float64)
                if len(saved_counts) == self.action_dim:
                    self.action_counts = saved_counts
                    self.total_action_count = int(ckpt.get('total_action_count', int(saved_counts.sum())))
            except Exception:
                pass
        self.epsilon = 0.0  # exploitation by default when loading
        print(f"[RL Agent] Checkpoint loaded from {path}")

    def _get_state_vector(
        self,
        prompt: str,
        prompt_ppl: Optional[float] = None,
        token_len: Optional[int] = None,
        lcr_score: Optional[float] = None,
        early_features: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        device_state = self.device_monitor.get_state()

        # Prompt complexity / sensitivity signal (LCR)
        # Primary: MiniBERT-based scorer returning a scalar in [0, 1].
        # Fallback: previous heuristic (token length + PPL + math/code density).
        if lcr_score is not None:
            complexity_score = float(lcr_score)
        elif self.lcr_scorer is not None:
            complexity_score = float(self.lcr_scorer.score(prompt))
        else:
            tokens = token_len if token_len is not None else len(self.tokenizer.encode(prompt))

            math_ops, code_syms = self._analyze_prompt_content(prompt)
            text_len = max(1, len(prompt))
            math_density = math_ops / text_len
            code_density = code_syms / text_len

            llm_norm = min(1.0, tokens / 200.0)
            ppl_norm = min(1.0, prompt_ppl / 50.0) if prompt_ppl is not None else 0.5
            math_norm = min(1.0, math_density / 0.05)
            code_norm = min(1.0, code_density / 0.05)
            complexity_score = (0.4 * llm_norm) + (0.3 * ppl_norm) + (0.15 * math_norm) + (0.15 * code_norm)

        # Early-Llama features (layer-0 partial forward, computed before pruning)
        ef = early_features or {}
        hidden_norm = float(ef.get("hidden_norm", 0.0))
        attn_entropy = float(ef.get("attn_entropy", 0.0))
        attn_max = float(ef.get("attn_max", 0.0))

        state = [
            device_state.cpu_utilization / 100.0,
            device_state.memory_available_gb / 16.0,   # 16 GB system RAM
            device_state.battery_percent / 100.0,
            float(device_state.gpu_available),
            device_state.gpu_memory_free_gb / 8.0,     # RTX 4060 8 GB VRAM
            device_state.gpu_utilization / 100.0,
            complexity_score,
            # Early-Llama runtime features (dims 7-9)
            hidden_norm,      # Layer-0 hidden-state L2 norm / hidden_dim
            attn_entropy,     # Layer-0 attention entropy / log(seq_len)
            attn_max,         # Layer-0 attention max (already [0,1])
        ]
        return torch.FloatTensor(state).unsqueeze(0)

    def _analyze_prompt_content(self, prompt: str):
        """Analyze prompt for math and code indicators using regex."""
        # Math patterns: arithmetic, latex, numbers
        math_pattern = r'[\+\-\*\/=\^\\]|\b(sin|cos|tan|log|sqrt|pi|sum|int)\b|\d+\.?\d*'
        # Code patterns: brackets, keywords, common operators
        code_pattern = r'[\{\}\[\]<>;]|\b(def|class|return|if|else|for|while|import|print|lambda)\b|==|!=|>=|<='
        
        math_matches = len(re.findall(math_pattern, prompt))
        code_matches = len(re.findall(code_pattern, prompt))
        
        return math_matches, code_matches

    def get_action(self, prompt: str, prompt_ppl: Optional[float] = None, token_len: Optional[int] = None, state_tensor: Optional[torch.Tensor] = None) -> PruningAction:
        # RL-driven action selection: epsilon-greedy + UCB exploration bonus
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            state = state_tensor if state_tensor is not None else self._get_state_vector(prompt, prompt_ppl, token_len)
            with torch.no_grad():
                q_values = self.policy_net(state).squeeze(0)  # [action_dim]
                # UCB1 exploration bonus: c * sqrt(ln(N) / N_a)
                if self.total_action_count > 0:
                    log_total = np.log(self.total_action_count + 1)
                    counts_safe = np.maximum(self.action_counts, 1.0)
                    ucb_bonus = self.ucb_c * np.sqrt(log_total / counts_safe)
                    q_values = q_values + torch.tensor(ucb_bonus, dtype=torch.float32)
                action_index = q_values.argmax().item()
        
        # Track visit counts for UCB
        self.action_counts[action_index] += 1
        self.total_action_count += 1

        action = self.action_space.get_action(action_index)
        print(f"[RL Agent] Epsilon: {self.epsilon:.3f}, Action: {action.target} ({action.intensity})")
        return action
        
    def train_step(self, state, action_index, reward, next_state):
        """Store transition and optimize using Double DQN when enough samples exist."""
        # Normalize inputs and push to buffer
        if isinstance(state, torch.Tensor):
            state_np = state.squeeze(0).detach().cpu().numpy()
        else:
            state_np = np.asarray(state, dtype=np.float32).squeeze()
        if isinstance(next_state, torch.Tensor):
            next_state_np = next_state.squeeze(0).detach().cpu().numpy()
        else:
            next_state_np = np.asarray(next_state, dtype=np.float32).squeeze()

        self.replay_buffer.append((state_np, int(action_index), float(reward), next_state_np))

        # Optimize if enough samples
        if len(self.replay_buffer) < self.batch_size:
            # Epsilon decay even without training to encourage exploration
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)

        # Current Q(s,a)
        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            # Double DQN: use policy net for argmax, target net for value
            next_policy_q = self.policy_net(next_states_t)
            next_actions = next_policy_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_states_t).gather(1, next_actions)
            target = rewards_t + self.gamma * next_target_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay and target update
        # Decay epsilon immediately toward epsilon_min to encourage learning from the start
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"[RL Agent] Train Step: Loss={loss.item():.4f}, Epsilon={self.epsilon:.3f}")

# ============================================================================
# (UNCHANGED) REAL MODEL ENGINE & BENCHMARKING SYSTEM
# ============================================================================

class RealBenchmark:
    """Benchmarks inference time, token speed, perplexity on baseline vs pruned models."""
    def __init__(self):
        self._warmed = False

    def _warmup_once(self, engine: RealModelEngine):
        if self._warmed:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = engine.generate_response("Hello", max_length=4)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        self._warmed = True
    def _calculate_perplexity_on_continuation(self, engine: RealModelEngine, prompt: str, generated_text: str) -> float:
        """Calculates perplexity on the generated continuation text."""
        full_text = prompt + generated_text
        inputs = engine.tokenizer(full_text, return_tensors="pt").to(engine.model.device)
        prompt_tokens = engine.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

        # Use -100 as the ignore_index for prompt tokens in the labels.
        labels = inputs.input_ids.clone()
        labels[:, :prompt_tokens] = -100

        with torch.no_grad():
            outputs = engine.model(**inputs, labels=labels)

        # The loss is now correctly calculated only on the generated part.
        ppl = torch.exp(outputs.loss).item()

        # Handle potential NaN/inf values from empty or problematic generation.
        if ppl is None or not torch.isfinite(torch.tensor(ppl)):
            return 10000.0 # Return a high penalty value.
        return min(ppl, 10000.0)  # Cap extreme PPL to prevent outliers

    def _calculate_perplexity(self, engine: RealModelEngine, text: str) -> float:
        inputs = engine.tokenizer(text, return_tensors="pt").to(engine.model.device)
        with torch.no_grad():
            outputs = engine.model(**inputs, labels=inputs["input_ids"])
        ppl = torch.exp(outputs.loss).item()
        if ppl is None or not torch.isfinite(torch.tensor(ppl)):
            return 10000.0
        return min(ppl, 10000.0)

    def benchmark_and_get_reward(self, engine: RealModelEngine, prompt: str, max_new_tokens: int = 50, return_metrics: bool = False, profile: bool = False):
        # Use planned token budget for fair throughput comparison
        planned_tokens = max_new_tokens
        # Synchronize GPU before timing to flush pending ops
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        if profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                generated_response = engine.generate_response(prompt, max_length=planned_tokens)
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        else:
            generated_response = engine.generate_response(prompt, max_length=planned_tokens)
        # Synchronize GPU after generation for accurate elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_s = (time.time() - start_time)
        inference_time_ms = elapsed_s * 1000
        # Actual generated tokens may vary; throughput uses actual for accuracy
        gen_token_count = len(engine.tokenizer.encode(generated_response))
        tokens_per_sec = (gen_token_count / elapsed_s) if elapsed_s > 0 else 0.0
        # Compute PPL on the generated continuation, conditioned on the prompt
        perplexity = self._calculate_perplexity_on_continuation(engine, prompt, generated_response)
        return { 'time_ms': inference_time_ms, 'tok_s': tokens_per_sec, 'perplexity': perplexity, 'gen_tokens': gen_token_count, 'planned_tokens': planned_tokens }

def generate_comparative_plots(metrics: List[Dict[str, Any]]):
    """Generate comparative scatter plots (baseline vs pruned) and correlation plots."""
    import numpy as np
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    n_episodes = len(metrics)

    # --- Adaptive styling based on episode count ---
    def _adaptive_params(n):
        """Return (marker_size, alpha, marker_line, tick_step) tuned to episode count."""
        if n <= 50:
            return 30, 0.6, 4, None  # use default ticks
        elif n <= 200:
            return 18, 0.5, 2, 50
        elif n <= 500:
            return 10, 0.4, 0, 100
        else:
            return 5, 0.3, 0, max(100, round(n / 10, -2))

    ms, alpha, mline, tick_step = _adaptive_params(n_episodes)

    def _set_episode_xticks(ax, episode_list, step=None):
        """Set x-axis ticks at round intervals (100, 200, ...) to avoid cramping."""
        if step is None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
            return
        step = int(step)
        ticks = list(range(step, max(episode_list) + 1, step))
        if episode_list and episode_list[0] not in ticks:
            ticks = [episode_list[0]] + ticks
        ax.set_xticks(ticks)

    def remove_outliers_xy(xs, ys):
        if not ys:
            return xs, ys
        q1, q3 = np.percentile(ys, 25), np.percentile(ys, 75)
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filt = [(x, y) for x, y in zip(xs, ys) if lb <= y <= ub]
        if not filt:
            return xs, ys
        fx, fy = zip(*filt)
        return list(fx), list(fy)

    episodes = [m['episode'] for m in metrics]

    # 1) Token speed comparison
    base_tok_s = [m['baseline_tok_s'] for m in metrics]
    pruned_tok_s = [m['tok_s'] for m in metrics]
    e_b, base_tok_s = remove_outliers_xy(episodes, base_tok_s)
    e_p, pruned_tok_s = remove_outliers_xy(episodes, pruned_tok_s)
    fig, ax = plt.subplots(figsize=(max(10, min(18, n_episodes * 0.03)), 6))
    ax.scatter(e_b, base_tok_s, color='red', alpha=alpha, label='Baseline Tok/s', s=ms)
    ax.scatter(e_p, pruned_tok_s, color='blue', alpha=alpha, label='Pruned Tok/s', s=ms)
    if len(e_b) > 1:
        coeff = np.polyfit(e_b, base_tok_s, 1)
        ax.plot(e_b, np.polyval(coeff, e_b), color='red')
    if len(e_p) > 1:
        coeff = np.polyfit(e_p, pruned_tok_s, 1)
        ax.plot(e_p, np.polyval(coeff, e_p), color='blue')
    ax.set_title('Token Speed per Episode (Baseline vs Pruned)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Tokens/sec'); ax.grid(True, alpha=0.3); ax.legend()
    _set_episode_xticks(ax, e_p or e_b, tick_step)
    fig.tight_layout()
    fig.savefig('token_speed_compare.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print('[Report] Saved token_speed_compare.png')

    # 2) Inference time comparison
    base_time = [m['baseline_time_ms'] for m in metrics]
    pruned_time = [m['time_ms'] for m in metrics]
    e_b, base_time = remove_outliers_xy(episodes, base_time)
    e_p, pruned_time = remove_outliers_xy(episodes, pruned_time)
    fig, ax = plt.subplots(figsize=(max(10, min(18, n_episodes * 0.03)), 6))
    ax.scatter(e_b, base_time, color='red', alpha=alpha, label='Baseline Time (ms)', s=ms)
    ax.scatter(e_p, pruned_time, color='blue', alpha=alpha, label='Pruned Time (ms)', s=ms)
    if len(e_b) > 1:
        coeff = np.polyfit(e_b, base_time, 1)
        ax.plot(e_b, np.polyval(coeff, e_b), color='red')
    if len(e_p) > 1:
        coeff = np.polyfit(e_p, pruned_time, 1)
        ax.plot(e_p, np.polyval(coeff, e_p), color='blue')
    ax.set_title('Inference Time per Episode (Baseline vs Pruned)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Time (ms)'); ax.grid(True, alpha=0.3); ax.legend()
    _set_episode_xticks(ax, e_p or e_b, tick_step)
    fig.tight_layout()
    fig.savefig('inference_time_compare.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print('[Report] Saved inference_time_compare.png')

    # 3) Perplexity comparison
    base_ppl = [m['baseline_ppl'] for m in metrics]
    pruned_ppl = [m['ppl'] for m in metrics]
    e_b, base_ppl = remove_outliers_xy(episodes, base_ppl)
    e_p, pruned_ppl = remove_outliers_xy(episodes, pruned_ppl)
    fig, ax = plt.subplots(figsize=(max(10, min(18, n_episodes * 0.03)), 6))
    ax.scatter(e_b, base_ppl, color='red', alpha=alpha, label='Baseline PPL', s=ms)
    ax.scatter(e_p, pruned_ppl, color='blue', alpha=alpha, label='Pruned PPL', s=ms)
    if len(e_b) > 1:
        coeff = np.polyfit(e_b, base_ppl, 1)
        ax.plot(e_b, np.polyval(coeff, e_b), color='red')
    if len(e_p) > 1:
        coeff = np.polyfit(e_p, pruned_ppl, 1)
        ax.plot(e_p, np.polyval(coeff, e_p), color='blue')
    ax.set_title('Perplexity per Episode (Baseline vs Pruned)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Perplexity'); ax.grid(True, alpha=0.3); ax.legend()
    _set_episode_xticks(ax, e_p or e_b, tick_step)
    fig.tight_layout()
    fig.savefig('perplexity_compare.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print('[Report] Saved perplexity_compare.png')

    # 4) Token length vs prompt perplexity correlation
    token_lens = [m['token_len'] for m in metrics]
    prompt_ppls = [m['prompt_ppl'] for m in metrics]
    x, y = remove_outliers_xy(token_lens, prompt_ppls)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color='purple', alpha=alpha, label='TokenLen vs Prompt PPL', s=ms)
    if len(x) > 1:
        coeff = np.polyfit(x, y, 1)
        ax.plot(x, np.polyval(coeff, x), color='purple')
    ax.set_title('Correlation: Token Length vs Prompt Perplexity')
    ax.set_xlabel('Token Length'); ax.set_ylabel('Prompt PPL'); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig('length_vs_ppl.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print('[Report] Saved length_vs_ppl.png')

    # 5) Time breakdown: LCR + RL Agent + Model Time per Episode (Stacked Bar Chart)
    lcr_times = [m.get('lcr_inference_time_ms', 0) for m in metrics]
    rl_times = [m.get('rl_inference_time_ms', 0) for m in metrics]
    model_times = [m.get('model_time_ms', 0) for m in metrics]
    total_times = [m['time_ms'] for m in metrics]
    episodes = [m['episode'] for m in metrics]

    # Remove outliers for consistency
    def remove_outliers_times(data_list):
        import numpy as np
        if not data_list:
            return data_list
        q1, q3 = np.percentile(data_list, 25), np.percentile(data_list, 75)
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return [d for d in data_list if lb <= d <= ub]

    # Since times are related, filter based on total time outliers
    filtered_indices = [i for i, t in enumerate(total_times) if t in remove_outliers_times(total_times)]
    episodes_filtered = [episodes[i] for i in filtered_indices]
    lcr_times_filtered = [lcr_times[i] for i in filtered_indices]
    rl_times_filtered = [rl_times[i] for i in filtered_indices]
    model_times_filtered = [model_times[i] for i in filtered_indices]
    total_times_filtered = [total_times[i] for i in filtered_indices]

    n_bars = len(episodes_filtered)
    fig_w = max(14, min(24, n_bars * 0.04))
    fig, ax = plt.subplots(figsize=(fig_w, 8))
    x_pos = np.arange(n_bars)
    bar_w = max(0.3, min(1.0, 800 / max(n_bars, 1)))
    bottom_lcr = np.array(lcr_times_filtered)
    bottom_rl = bottom_lcr + np.array(rl_times_filtered)
    ax.bar(x_pos, lcr_times_filtered, width=bar_w, color='green', label='LCR Inference (ms)', alpha=0.8)
    ax.bar(x_pos, rl_times_filtered, width=bar_w, bottom=bottom_lcr, color='orange', label='RL Agent Time (ms)', alpha=0.8)
    ax.bar(x_pos, model_times_filtered, width=bar_w, bottom=bottom_rl, color='blue', label='Model Inference (ms)', alpha=0.8)
    line_ms_size = max(1, 4 - n_bars // 200)
    ax.plot(x_pos, total_times_filtered, color='red', marker='o', linestyle='-', linewidth=2, markersize=line_ms_size, label='Total Pruned Time (ms)')
    baseline_times = [m.get('baseline_time_ms', 0) for m in metrics]
    baseline_filtered = [baseline_times[i] for i in filtered_indices]
    ax.plot(x_pos, baseline_filtered, color='gray', linestyle='--', linewidth=2, label='Baseline (Unpruned Model)')
    ax.set_title('Inference Time Breakdown per Episode', fontsize=16, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    # X-axis: show round-number episode labels (100, 200, ...) instead of every episode
    if tick_step and n_bars > 50:
        tick_positions = list(range(tick_step - 1, n_bars, tick_step))
        tick_labels = [str(episodes_filtered[p]) for p in tick_positions if p < n_bars]
        tick_positions = [p for p in tick_positions if p < n_bars]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
    else:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(episodes_filtered, fontsize=max(6, 10 - n_bars // 30))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig('time_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[Report] Time breakdown plot saved to time_breakdown.png")

    # 6) Reward progression with moving average
    rewards = [m.get('reward', 0) for m in metrics]
    if rewards and any(r != 0 for r in rewards):
        episodes_r = [m['episode'] for m in metrics]
        fig, ax = plt.subplots(figsize=(max(10, min(18, n_episodes * 0.03)), 6))
        ax.scatter(episodes_r, rewards, alpha=alpha, color='teal', label='Reward', s=ms)
        window = min(max(5, n_episodes // 20), len(rewards))
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ma_x = episodes_r[window-1:]
            ax.plot(ma_x, ma, color='red', linewidth=2, label=f'Moving Avg (w={window})')
        if len(episodes_r) > 1:
            coeff = np.polyfit(episodes_r, rewards, 1)
            ax.plot(episodes_r, np.polyval(coeff, episodes_r), color='purple', linestyle='--', label='Trendline')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_title('Reward Progression per Episode', fontsize=16, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        _set_episode_xticks(ax, episodes_r, tick_step)
        ax.legend()
        fig.tight_layout()
        fig.savefig('reward_progression.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("[Report] Reward progression plot saved to reward_progression.png")

    # 7) Quality vs Speed tradeoff (Pareto scatter)
    speed_gains = []
    ppl_ratios = []
    action_labels_scatter = []
    for m in metrics:
        base_tok = m.get('baseline_tok_s', 0)
        pruned_tok = m.get('tok_s', 0)
        base_ppl_val = m.get('baseline_ppl', 1)
        pruned_ppl_val = m.get('ppl', 1)
        if base_tok > 0 and base_ppl_val > 0:
            speed_gains.append((pruned_tok - base_tok) / base_tok * 100)
            ppl_ratios.append((pruned_ppl_val - base_ppl_val) / base_ppl_val * 100)
            action_labels_scatter.append(m.get('target', 'none'))
    if speed_gains:
        sg_arr, pr_arr = np.array(speed_gains), np.array(ppl_ratios)
        sg_q1, sg_q3 = np.percentile(sg_arr, 10), np.percentile(sg_arr, 90)
        pr_q1, pr_q3 = np.percentile(pr_arr, 10), np.percentile(pr_arr, 90)
        sg_iqr, pr_iqr = sg_q3 - sg_q1, pr_q3 - pr_q1
        sg_lb, sg_ub = sg_q1 - 2.0 * sg_iqr, sg_q3 + 2.0 * sg_iqr
        pr_lb, pr_ub = pr_q1 - 2.0 * pr_iqr, pr_q3 + 2.0 * pr_iqr
        inlier = [(sg_lb <= s <= sg_ub) and (pr_lb <= p <= pr_ub) for s, p in zip(speed_gains, ppl_ratios)]
        sg_f = [s for s, m in zip(speed_gains, inlier) if m]
        pr_f = [p for p, m in zip(ppl_ratios, inlier) if m]
        al_f = [a for a, m in zip(action_labels_scatter, inlier) if m]
        n_outliers = len(speed_gains) - len(sg_f)
        if n_outliers > 0:
            print(f"[Report] Quality vs Speed: filtered {n_outliers} extreme outlier(s) from plot")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_map = {'attention_heads': 'blue', 'transformer_layers': 'green', 'none': 'gray'}
        for label in set(al_f):
            idx = [j for j, a in enumerate(al_f) if a == label]
            ax.scatter([sg_f[j] for j in idx], [pr_f[j] for j in idx],
                       alpha=alpha, label=label, color=colors_map.get(label, 'purple'), s=ms)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_title('Quality vs Speed Tradeoff', fontsize=16, fontweight='bold')
        ax.set_xlabel('Speed Change (%)', fontsize=12)
        ax.set_ylabel('PPL Change (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig('quality_vs_speed.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("[Report] Quality vs speed plot saved to quality_vs_speed.png")

    # 8) Epsilon decay (if tracked)
    epsilons = [m.get('epsilon', None) for m in metrics]
    if epsilons and any(e is not None for e in epsilons):
        eps_vals = [e for e in epsilons if e is not None]
        eps_episodes = [m['episode'] for m, e in zip(metrics, epsilons) if e is not None]
        fig, ax = plt.subplots(figsize=(max(10, min(18, n_episodes * 0.03)), 6))
        ax.plot(eps_episodes, eps_vals, color='darkorange', linewidth=2)
        ax.set_title('Epsilon Decay During Training', fontsize=16, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Epsilon', fontsize=12)
        ax.grid(True, alpha=0.3)
        _set_episode_xticks(ax, eps_episodes, tick_step)
        fig.tight_layout()
        fig.savefig('epsilon_decay.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("[Report] Epsilon decay plot saved to epsilon_decay.png")

    # 9) Cumulative reward over episodes
    if rewards and any(r != 0 for r in rewards):
        cumulative = np.cumsum(rewards)
        fig, ax = plt.subplots(figsize=(max(10, min(18, n_episodes * 0.03)), 6))
        ax.plot(episodes_r, cumulative, color='darkgreen', linewidth=2)
        ax.fill_between(episodes_r, cumulative, alpha=0.2, color='green')
        ax.set_title('Cumulative Reward Over Episodes', fontsize=16, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Cumulative Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        _set_episode_xticks(ax, episodes_r, tick_step)
        fig.tight_layout()
        fig.savefig('cumulative_reward.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("[Report] Cumulative reward plot saved to cumulative_reward.png")

    # 10) VRAM usage per episode: Available vs Used (for pruning + inference only)
    vram_used = [m.get('vram_used_gb', 0) for m in metrics]
    vram_total = [m.get('vram_total_gb', 0) for m in metrics]
    if any(v > 0 for v in vram_total):
        episodes_v = [m['episode'] for m in metrics]
        vram_available = [t - u for t, u in zip(vram_total, vram_used)]
        fig, ax = plt.subplots(figsize=(max(10, min(18, n_episodes * 0.03)), 6))
        ax.fill_between(episodes_v, vram_total, alpha=0.15, color='gray', label='Total VRAM (GB)')
        ax.plot(episodes_v, vram_total, color='gray', linestyle='--', linewidth=1.5)
        ax.fill_between(episodes_v, vram_used, alpha=0.4, color='red', label='VRAM Used (GB)')
        ax.plot(episodes_v, vram_used, color='red', linewidth=2, markersize=line_ms_size if n_bars > 0 else 4)
        ax.plot(episodes_v, vram_available, color='green', linewidth=2, label='VRAM Available (GB)')
        ax.set_title('VRAM Usage per Episode (Pruning + Inference)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('VRAM (GB)', fontsize=12)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        _set_episode_xticks(ax, episodes_v, tick_step)
        ax.legend()
        fig.tight_layout()
        fig.savefig('vram_usage.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("[Report] VRAM usage plot saved to vram_usage.png")

def generate_report(metrics_list: List[Dict[str, Any]], report_filename: str = 'training_report.txt', header: str = 'Training Report'):
    """Generate post-training report with averages by prune type/intensity and plots."""
    if not metrics_list:
        print("[Report] No metrics to report.")
        return

    # Overall averages
    total_time = sum(m['time_ms'] for m in metrics_list)
    total_ppl = sum(m['ppl'] for m in metrics_list)
    n = len(metrics_list)
    print(f"\n[Report] Overall Avg Time: {total_time/n:.2f}ms | Avg PPL: {total_ppl/n:.2f}")

    # Group by target and intensity
    from collections import defaultdict
    groups = defaultdict(list)
    for m in metrics_list:
        key = (m['target'], m['intensity'])
        groups[key].append(m)

    print("\n[Report] Averages by Prune Type and Intensity:")
    for (target, intensity), ms in groups.items():
        avg_time = sum(m['time_ms'] for m in ms) / len(ms)
        avg_ppl = sum(m['ppl'] for m in ms) / len(ms)
        print(f"  {target} {intensity}: Avg Time {avg_time:.2f}ms, Avg PPL {avg_ppl:.2f} ({len(ms)} samples)")

    # Pruning Summary: Two separate plots for Avg Time and Avg PPL
    import numpy as np
    action_labels = [f"{target} {intensity}" for (target, intensity), _ in sorted(groups.items(), key=lambda x: x[1][0]['action_index'])]
    avg_times = [sum(m['time_ms'] for m in ms) / len(ms) for _, ms in sorted(groups.items(), key=lambda x: x[1][0]['action_index'])]
    avg_ppls = [sum(m['ppl'] for m in ms) / len(ms) for _, ms in sorted(groups.items(), key=lambda x: x[1][0]['action_index'])]
    x = np.arange(len(action_labels))
    
    # Plot for Avg Time
    plt.figure(figsize=(16, 8))
    plt.bar(x, avg_times, color='skyblue', alpha=0.8, width=0.6)
    plt.title('Average Inference Time per Pruning Action', fontsize=16, fontweight='bold')
    plt.xlabel('Pruning Action', fontsize=12)
    plt.ylabel('Average Time (ms)', fontsize=12)
    plt.xticks(x, action_labels, rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('inference_time_per_action.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[Report] Inference time per action plot saved to inference_time_per_action.png")
    
    # Plot for Avg PPL
    plt.figure(figsize=(16, 8))
    plt.bar(x, avg_ppls, color='salmon', alpha=0.8, width=0.6)
    plt.title('Average Perplexity per Pruning Action', fontsize=16, fontweight='bold')
    plt.xlabel('Pruning Action', fontsize=12)
    plt.ylabel('Average PPL', fontsize=12)
    plt.xticks(x, action_labels, rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('perplexity_per_action.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[Report] Perplexity per action plot saved to perplexity_per_action.png")

    # Pruning Action Usage Count Plot (New Graph for usage counts)
    usage_counts = {}
    for m in metrics_list:
        key = (m['target'], m['intensity'])
        usage_counts[key] = usage_counts.get(key, 0) + 1
    usage_labels = [f"{target} {intensity}" for (target, intensity) in sorted(usage_counts.keys(), key=lambda x: groups[x][0]['action_index'] if x in groups else 0)]
    usage_values = [usage_counts[(target, intensity)] for (target, intensity) in sorted(usage_counts.keys(), key=lambda x: groups[x][0]['action_index'] if x in groups else 0)]
    plt.figure(figsize=(16, 8))
    plt.bar(usage_labels, usage_values, color='lightgreen', alpha=0.8, width=0.6)
    plt.title('Pruning Action Usage Counts', fontsize=16, fontweight='bold')
    plt.xlabel('Pruning Action', fontsize=12)
    plt.ylabel('Usage Count', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('pruning_action_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[Report] Pruning action usage plot saved to pruning_action_usage.png")


    episodes = [m['episode'] for m in metrics_list]
    times = [m['time_ms'] for m in metrics_list]
    ppls = [m['ppl'] for m in metrics_list]

    # Function to remove outliers using IQR
    def remove_outliers(data):
        import numpy as np
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return [d for d in data if lower_bound <= d <= upper_bound]

    # Inference Time Plot
    times_filtered = remove_outliers(times)
    episodes_filtered_time = episodes[:len(times_filtered)]  # Assume order, but to be precise, filter indices
    # Since we filtered, need to get corresponding episodes
    indices = [i for i, t in enumerate(times) if times[i] in times_filtered]
    episodes_filtered_time = [episodes[i] for i in indices]
    times_filtered = times_filtered

    plt.figure(figsize=(10, 6))
    plt.scatter(episodes_filtered_time, times_filtered, alpha=0.6, label='Data')
    if len(episodes_filtered_time) > 1:
        import numpy as np
        coeff = np.polyfit(episodes_filtered_time, times_filtered, 1)
        trendline = np.polyval(coeff, episodes_filtered_time)
        plt.plot(episodes_filtered_time, trendline, color='red', label='Trendline')
    plt.title('Inference Time per Episode (Outliers Removed)')
    plt.xlabel('Episode')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('inference_time.png')
    plt.close()
    print("[Report] Inference time plot saved to inference_time.png")

    # Perplexity Plot
    ppls_filtered = remove_outliers(ppls)
    indices_ppl = [i for i, p in enumerate(ppls) if ppls[i] in ppls_filtered]
    episodes_filtered_ppl = [episodes[i] for i in indices_ppl]
    ppls_filtered = ppls_filtered

    plt.figure(figsize=(10, 6))
    plt.scatter(episodes_filtered_ppl, ppls_filtered, alpha=0.6, label='Data')
    if len(episodes_filtered_ppl) > 1:
        import numpy as np
        coeff = np.polyfit(episodes_filtered_ppl, ppls_filtered, 1)
        trendline = np.polyval(coeff, episodes_filtered_ppl)
        plt.plot(episodes_filtered_ppl, trendline, color='red', label='Trendline')
    plt.title('Perplexity per Episode (Outliers Removed)')
    plt.xlabel('Episode')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('perplexity.png')
    plt.close()
    print("[Report] Perplexity plot saved to perplexity.png")

    # Compute token-weighted PPL (academically correct aggregate PPL)
    def _token_weighted_ppl(mlist, ppl_key='ppl', tok_key='gen_tokens', default_tok=50):
        total_loss, total_tokens = 0.0, 0
        for m in mlist:
            gen_tok = m.get(tok_key) or default_tok
            ppl_val = m.get(ppl_key, 1.01)
            loss = np.log(max(ppl_val, 1.01))
            total_loss += loss * gen_tok
            total_tokens += gen_tok
        if total_tokens == 0:
            return 0.0
        return float(np.exp(total_loss / total_tokens))

    tw_ppl_pruned = _token_weighted_ppl(metrics_list, 'ppl')
    tw_ppl_baseline = _token_weighted_ppl(metrics_list, 'baseline_ppl')

    # Save report to file
    with open(report_filename, 'w') as f:
        f.write(f"{header}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Episodes: {n}\n")
        f.write(f"Overall Avg Total Time (LCR+RL+Model): {total_time/n:.2f}ms\n")
        avg_lcr = sum(m.get('lcr_inference_time_ms', 0) for m in metrics_list) / n
        avg_rl = sum(m.get('rl_inference_time_ms', 0) for m in metrics_list) / n
        avg_model = sum(m.get('model_time_ms', 0) for m in metrics_list) / n
        avg_baseline = sum(m.get('baseline_time_ms', 0) for m in metrics_list) / n
        f.write(f"  Avg LCR Inference: {avg_lcr:.2f}ms\n")
        f.write(f"  Avg RL Agent: {avg_rl:.2f}ms\n")
        f.write(f"  Avg Model Inference (pruned): {avg_model:.2f}ms\n")
        f.write(f"  Avg Baseline Inference (unpruned): {avg_baseline:.2f}ms\n")
        avg_vram_used = sum(m.get('vram_used_gb', 0) for m in metrics_list) / n
        avg_vram_total = max(m.get('vram_total_gb', 0) for m in metrics_list) if any(m.get('vram_total_gb', 0) for m in metrics_list) else 0
        if avg_vram_total > 0:
            f.write(f"  Avg VRAM Used (pruning+inference): {avg_vram_used:.2f} GB / {avg_vram_total:.2f} GB ({avg_vram_used/avg_vram_total*100:.1f}%)\n")
        f.write(f"Overall Avg PPL (pruned, arithmetic): {total_ppl/n:.2f}\n")
        f.write(f"Overall PPL (pruned, token-weighted): {tw_ppl_pruned:.2f}\n")
        avg_baseline_ppl = sum(m.get('baseline_ppl', 0) for m in metrics_list) / n
        f.write(f"Overall Avg PPL (baseline, arithmetic): {avg_baseline_ppl:.2f}\n")
        f.write(f"Overall PPL (baseline, token-weighted): {tw_ppl_baseline:.2f}\n\n")
        avg_reward = sum(m.get('reward', 0) for m in metrics_list) / n
        f.write(f"Overall Avg Reward: {avg_reward:.4f}\n\n")
        f.write("Averages by Prune Type and Intensity:\n")
        for (target, intensity), ms in sorted(groups.items(), key=lambda x: x[1][0]['action_index']):
            avg_time = sum(m['time_ms'] for m in ms) / len(ms)
            avg_ppl = sum(m['ppl'] for m in ms) / len(ms)
            avg_r = sum(m.get('reward', 0) for m in ms) / len(ms)
            tw_ppl = _token_weighted_ppl(ms)
            f.write(f"  {target} {intensity}: Avg Time {avg_time:.2f}ms, Avg PPL {avg_ppl:.2f}, TW-PPL {tw_ppl:.2f}, Avg Reward {avg_r:.4f} ({len(ms)} samples)\n")
    print(f"[Report] Report saved to {report_filename}")
def organize_training_reports(is_report_mode: bool = False):
    """Organize training reports into numbered subfolders under 'Training Report'."""
    import os
    import shutil
    import re
    
    # Create Training Report folder if not exists
    report_dir = 'Training Report'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        print(f"[Organize] Created directory: {report_dir}")
    
    # Find existing Train folders and determine next number
    subfolders = [f for f in os.listdir(report_dir) if os.path.isdir(os.path.join(report_dir, f)) and re.match(r'Train \d+', f)]
    numbers = [int(re.search(r'\d+', f).group()) for f in subfolders if re.search(r'\d+', f)]
    next_num = max(numbers) + 1 if numbers else 1
    train_folder = f'Train {next_num}'
    train_path = os.path.join(report_dir, train_folder)
    os.makedirs(train_path)
    print(f"[Organize] Created subfolder: {train_path}")
    
    # Files to move
    files_to_move = [
        'inference_time_compare.png',
        'inference_time.png',
        'length_vs_ppl.png',
        'perplexity_compare.png',
        'perplexity.png',
        'pruning_summary.png',
        'pruning_action_usage.png',
        'inference_time_per_action.png',
        'perplexity_per_action.png',
        'token_speed_compare.png',
        'time_breakdown.png',
        'reward_progression.png',
        'quality_vs_speed.png',
        'epsilon_decay.png',
        'cumulative_reward.png',
        'vram_usage.png',
        'zero_shot_baseline_accuracy.png',
        'zero_shot_baseline_metrics.json',
        'training_report.txt'
    ]
    if not is_report_mode:
        files_to_move.append('training_metrics.json')
    
    for file in files_to_move:
        if os.path.exists(file):
            shutil.move(file, os.path.join(train_path, file))
            print(f"[Organize] Moved {file} to {train_path}")
        else:
            print(f"[Organize] Warning: {file} not found, skipping")
    
    print(f"[Organize] Training reports organized into {train_path}")
def load_training_prompts(dataset_name: str, split: str = 'train', samples: int = 5000, split_type: str = 'train') -> List[str]:
    """Load a proper prompt dataset to train/test the RL controller.
    Default: 'togethercomputer/RedPajama-Data-1T-Sample'
    Returns a list of prompt strings.

    When the CSV contains a 'Split' column, only rows matching `split_type`
    (train/test) are returned — ensuring the same 80-20 split as LCR training.
    """
    name = dataset_name.strip()

    # Local CSV training should not depend on Hugging Face datasets.
    if name.endswith('.csv') and os.path.exists(name):
        try:
            with open(name, 'r', encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                has_split_col = 'Split' in fieldnames
                prompts = []
                for row in reader:
                    # Filter by Split column when available
                    if has_split_col and split_type:
                        row_split = (row.get('Split') or '').strip().lower()
                        if row_split and row_split != str(split_type).strip().lower():
                            continue
                    prompt = (
                        row.get('Prompt')
                        or row.get('prompt')
                        or row.get('text')
                        or row.get('instruction')
                        or row.get('content')
                    )
                    if prompt and str(prompt).strip():
                        prompts.append(str(prompt).strip())
            random.shuffle(prompts)
            if prompts:
                selected = prompts[:samples]
                split_msg = f" (Split={split_type})" if has_split_col else ""
                print(f"[Train] Loaded {len(selected)} prompts from local CSV {name}{split_msg}.")
                return selected
            print(f"[Train] Local CSV {name} contains no usable prompts. Falling back to external datasets.")
        except Exception as e:
            print(f"[Train] Failed to read local CSV {name}: {e}. Falling back to external datasets.")

    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[Train] Hugging Face datasets unavailable ({e}). Falling back to hardcoded prompts.")
        return [
            "What is the capital of France?",
            "Explain the concept of machine learning in simple terms.",
            "Why is the sky blue during the day?",
            "Write a python function to calculate the factorial of a number.",
        ]

    # Use RedPajama by default if the user hasn't specified a custom CSV
    if name == 'Prompt Dataset Train.csv' and not os.path.exists(name):
        print(f"[Train] '{name}' not found locally. Switching to RedPajama sample...")
        name = 'togethercomputer/RedPajama-Data-1T-Sample'

    print(f"[Train] Loading dataset: {name} ({split}) ...")
    
    try:
        if name.endswith('.csv'):
            if os.path.exists(name):
                ds = load_dataset('csv', data_files=name)
                prompts = [r['Prompt'] for r in ds['train'] if r.get('Prompt') and str(r['Prompt']).strip()]
            else:
                print(f"[Train] CSV file {name} not found. Falling back to RedPajama.")
                name = 'togethercomputer/RedPajama-Data-1T-Sample'
                ds = load_dataset(name, split=split, trust_remote_code=True)
                # RedPajama has 'text' field
                prompts = [r['text'][:1024] for r in ds if r.get('text') and len(r['text']) > 50]
        elif 'RedPajama' in name:
            ds = load_dataset(name, split=split, trust_remote_code=True)
            # Filter for reasonable length and English-like text
            prompts = []
            for r in ds:
                text = r.get('text', '')
                if text and 100 < len(text) < 2000:
                    prompts.append(text)
                if len(prompts) >= samples * 2: # Optimize loading speed
                     break
        else:
            # Fallback for other HF datasets
            ds = load_dataset(name, split=split)
            prompts = []
            for r in ds:
                # Try common fields
                t = r.get('text') or r.get('instruction') or r.get('content')
                if t and str(t).strip():
                    prompts.append(str(t))

        random.shuffle(prompts)
        if not prompts:
            print("[Train] No valid prompts found. Using fallback.")
            prompts = ["Explain quantum computing in simple terms."]
        
        print(f"[Train] Loaded {len(prompts)} prompts from {name}.")
        return prompts[:samples]

    except Exception as e:
        print(f"[Train] Failed to load {name}: {e}. Falling back to WikiText-2.")
        try:
            ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            prompts = [r['text'] for r in ds if r['text'] and len(r['text']) > 50]
            random.shuffle(prompts)
            return prompts[:samples]
        except Exception:
            return ["Describe the water cycle.", "Who wrote 'Hamlet'?", "Define photosynthesis."]
# =========================================================================
# MAIN TRAINING LOOP AND UTILITIES
# =========================================================================

def main(num_episodes: int = 50,
         checkpoint_path: Optional[str] = None,
         max_new_tokens: int = 50,
         train_dataset: str = 'Prompt Dataset Train.csv',
         train_split: str = 'train',
         train_samples: int = 5000,
         split_type: str = 'train',
         device: str = 'auto',
         static_profiles: bool = False,
         sparsity_2to4: bool = False,
         compile_profiles: bool = False,
         kv_compress: bool = False,
         kv_keep_ratio: float = 1.0,
         split_ratio: float = 1.0,
         test_samples: int = 100):
    model_engine = RealModelEngine(
        device=device,
        enable_static_profiles=static_profiles,
        enable_2to4=sparsity_2to4,
        enable_compile=compile_profiles,
        enable_kv_compression=kv_compress,
        kv_keep_ratio=kv_keep_ratio,
    )
    rl_agent = RLControllerAgent(model_engine.tokenizer)
    benchmark = RealBenchmark()
    
    validation_text = "The field of artificial intelligence has seen rapid advancements in recent years, with breakthroughs in machine learning, natural language processing, and computer vision. These technologies are transforming industries such as healthcare, finance, and transportation. However, ethical considerations and responsible AI development remain critical to ensure these innovations benefit society as a whole. The integration of AI into everyday life raises important questions about privacy, bias, and the future of work."
    metrics_list = []
    # Load a proper training prompt pool from the specified dataset
    all_prompts = load_training_prompts(train_dataset, split=train_split, samples=train_samples, split_type=split_type)

    # Apply train-test split if split_ratio < 1.0
    test_prompts_split = []
    if 0.0 < split_ratio < 1.0:
        split_idx = int(len(all_prompts) * split_ratio)
        prompt_pool = all_prompts[:split_idx]
        test_prompts_split = all_prompts[split_idx:]
        print(f"[System] Dataset split: {len(prompt_pool)} train ({split_ratio*100:.0f}%) / {len(test_prompts_split)} test ({(1-split_ratio)*100:.0f}%)")
    else:
        prompt_pool = all_prompts

    # Honor CLI episodes: cap to requested number
    limit = min(num_episodes, len(prompt_pool))
    prompt_pool = prompt_pool[:limit]
    num_episodes = limit

    # Adaptive epsilon decay: reach epsilon_min by the last actual training episode.
    rl_agent.configure_epsilon_schedule(num_episodes)

    # Calibrate importance scores for heads/FFN/layers using a small subset
    try:
        calib_samples = min(64, len(prompt_pool))
        model_engine.calibrate_importances(prompt_pool, max_samples=calib_samples, max_seq_len=128)
    except Exception as e:
        print(f"[Calib] Warning: calibration skipped due to error: {e}")

    # Pre-training zero-shot accuracy evaluation (dense model baseline)
    print("\n[System] Running pre-training zero-shot accuracy evaluation...")
    try:
        run_zero_shot_baseline_eval(model_engine, samples=200, phase_label='Pre-Training')
    except Exception as e:
        print(f"[Zero-shot] Warning: pre-training eval skipped due to error: {e}")
    
    print(f"\n[System] Starting RL Training for {num_episodes} episodes...")

    # GPU warmup: flush cold-start overhead before timing anything
    benchmark._warmup_once(model_engine)

    metrics_list = []  # Collect metrics for report
    
    for i, prompt in enumerate(prompt_pool):
        print("\n" + "#"*80 + f"\n# EPISODE {i+1}/{num_episodes}\n" + "#"*80)
        print(f"[System] Using prompt: '{prompt[:80]}...'")
        
        # Phase 1.1: Token length and prompt perplexity (no pruning)
        token_len = len(rl_agent.tokenizer.encode(prompt))
        prompt_ppl = benchmark._calculate_perplexity(model_engine, prompt)
        print(f"[Episode {i+1}] Token length: {token_len}, Prompt PPL: {prompt_ppl:.2f}")

        # Ensure clean (no-prune) state and measure baseline metrics
        model_engine.restore_model()
        base_metrics = benchmark.benchmark_and_get_reward(
            model_engine, prompt, max_new_tokens=max_new_tokens, return_metrics=True
        )
        print(f"[Baseline] Time: {base_metrics['time_ms']:.2f}ms | Tok/s: {base_metrics['tok_s']:.2f} | PPL: {base_metrics['perplexity']:.2f} | GenTokens: {base_metrics.get('gen_tokens', 0)}")

        # Phase 1.3: Compute LCR score + early-Llama features (timed together as LCR overhead)
        start_lcr = time.time()
        if getattr(rl_agent, 'lcr_scorer', None) is not None:
            complexity_score = float(rl_agent.lcr_scorer.score(prompt))
            print(f"[LCR] Score: {complexity_score:.3f} (MiniBERT)")
        else:
            llm_norm = min(1.0, token_len / 200.0)
            ppl_norm = min(1.0, prompt_ppl / 50.0)
            complexity_score = 0.6 * llm_norm + 0.4 * ppl_norm
            print(f"[Complexity] Score: {complexity_score:.3f} (llm_norm={llm_norm:.3f}, ppl_norm={ppl_norm:.3f})")

        # Phase 1.3b: Extract cheap early-Llama features (layer-0 only, before pruning)
        try:
            early_features = model_engine.extract_early_features(prompt)
            print(f"[Early-Llama] hidden_norm={early_features['hidden_norm']:.4f} | attn_entropy={early_features['attn_entropy']:.4f} | attn_max={early_features['attn_max']:.4f}")
        except Exception as e:
            print(f"[Early-Llama] Warning: extraction failed ({e}), using zeros.")
            early_features = {"hidden_norm": 0.0, "attn_entropy": 0.0, "attn_max": 0.0}
        lcr_inference_time_ms = (time.time() - start_lcr) * 1000

        # Phase 1.4: RL-based pruning and pruned metrics
        start_rl = time.time()
        state_tensor = rl_agent._get_state_vector(prompt, prompt_ppl=prompt_ppl, token_len=token_len, lcr_score=complexity_score, early_features=early_features)
        pruning_action = rl_agent.get_action(prompt, prompt_ppl=prompt_ppl, token_len=token_len, state_tensor=state_tensor)
        rl_inference_time_ms = (time.time() - start_rl) * 1000
        model_engine.apply_pruning(pruning_action)
        # Track VRAM usage specifically for pruning + inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pruned_metrics = benchmark.benchmark_and_get_reward(
            model_engine, prompt, max_new_tokens=max_new_tokens, return_metrics=True
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            vram_used_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            vram_total_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        else:
            vram_used_gb = 0.0
            vram_total_gb = 0.0
        total_pruned_time = lcr_inference_time_ms + rl_inference_time_ms + pruned_metrics['time_ms']
        print(f"[Pruned] Action: {pruning_action.target} ({pruning_action.intensity}) | LCR Time: {lcr_inference_time_ms:.2f}ms | RL Time: {rl_inference_time_ms:.2f}ms | Model Time: {pruned_metrics['time_ms']:.2f}ms | Total Time: {total_pruned_time:.2f}ms | Tok/s: {pruned_metrics['tok_s']:.2f} | PPL: {pruned_metrics['perplexity']:.2f} | GenTokens: {pruned_metrics.get('gen_tokens', 0)} | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB")

        # Reward: balanced speed/quality with log-PPL to prevent catastrophic penalty
        alpha, beta = 0.9, 0.1
        eps = 1e-8
        speed_gain = (pruned_metrics['tok_s'] - base_metrics['tok_s']) / (base_metrics['tok_s'] + eps)
        # Log-PPL penalty: bounded and proportional (avoids heavy-tailed PPL explosion)
        log_ppl_base = np.log(max(base_metrics['perplexity'], 1.01))
        log_ppl_pruned = np.log(max(pruned_metrics['perplexity'], 1.01))
        ppl_penalty = max(0.0, log_ppl_pruned - log_ppl_base)
        relative_reward = alpha * speed_gain - beta * ppl_penalty
        # Clamp reward to [-2, 2] to prevent extreme outliers from destabilizing training
        relative_reward = float(np.clip(relative_reward, -2.0, 2.0))

        next_state_tensor = rl_agent._get_state_vector(prompt, prompt_ppl=prompt_ppl, token_len=token_len, lcr_score=complexity_score, early_features=early_features)
        rl_agent.train_step(state_tensor, pruning_action.action_index, relative_reward, next_state_tensor)
        print(f"[reward] {relative_reward:.3f}")
        
        # Collect detailed metrics for analysis and plots
        metrics_list.append({
            'episode': i+1,
            'token_len': token_len,
            'prompt_ppl': prompt_ppl,
            'complexity': complexity_score,
            'baseline_time_ms': base_metrics['time_ms'],
            'baseline_tok_s': base_metrics['tok_s'],
            'baseline_ppl': base_metrics['perplexity'],
            'baseline_gen_tokens': base_metrics.get('gen_tokens', None),
            'time_ms': lcr_inference_time_ms + rl_inference_time_ms + pruned_metrics['time_ms'],
            'model_time_ms': pruned_metrics['time_ms'],
            'rl_inference_time_ms': rl_inference_time_ms,
            'lcr_inference_time_ms': lcr_inference_time_ms,
            'tok_s': pruned_metrics['tok_s'],
            'ppl': pruned_metrics['perplexity'],
            'gen_tokens': pruned_metrics.get('gen_tokens', None),
            'action_index': pruning_action.action_index,
            'target': pruning_action.target,
            'intensity': pruning_action.intensity,
            'reward': relative_reward,
            'epsilon': rl_agent.epsilon,
            'vram_used_gb': vram_used_gb,
            'vram_total_gb': vram_total_gb
        })
        
        model_engine.restore_model()

    print("\n[System] ✓ RL training loop completed!")
    
    # Save the trained RL policy
    if checkpoint_path:
        rl_agent.save(checkpoint_path)
    
    # Generate reports and comparative plots
    generate_report(metrics_list)
    generate_comparative_plots(metrics_list)

    # Save metrics for later report generation and analysis
    import json
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics_list, f)
    
    # Organize training reports into folders
    organize_training_reports()

    # Run oracle dataset zero-shot evaluation (MMLU & BoolQ) after training
    _train_zs_results = {}
    if train_dataset and train_dataset.endswith('.csv') and os.path.exists(train_dataset):
        try:
            _train_zs_results = run_oracle_dataset_zeroshot(
                model_engine, train_dataset, split_filter='test',
                phase_label='Post-Training', max_seq_len=512,
            )
        except Exception as e:
            print(f"[Oracle ZS] Post-training zero-shot eval failed: {e}")

    # Auto-run testing on held-out split if train-test split was used
    _test_zs_results = {}
    if test_prompts_split:
        test_cap = min(len(test_prompts_split), test_samples)
        print(f"\n[System] Running evaluation on {test_cap} held-out test prompts (of {len(test_prompts_split)} available)...")
        test_agent(model_engine, rl_agent, benchmark,
                   num_test_episodes=test_cap,
                   max_new_tokens=max_new_tokens,
                   prompts=test_prompts_split[:test_cap])

        # Run oracle dataset zero-shot after testing too
        if train_dataset and train_dataset.endswith('.csv') and os.path.exists(train_dataset):
            try:
                _test_zs_results = run_oracle_dataset_zeroshot(
                    model_engine, train_dataset, split_filter='test',
                    phase_label='Post-Testing', max_seq_len=512,
                )
            except Exception as e:
                print(f"[Oracle ZS] Post-testing zero-shot eval failed: {e}")

    # Generate combined comparison graph if both phases were evaluated
    if _train_zs_results and _test_zs_results:
        try:
            generate_zeroshot_comparison_graph(_train_zs_results, _test_zs_results)
        except Exception as e:
            print(f"[Oracle ZS] Comparison graph generation failed: {e}")

def organize_test_reports(is_report_mode: bool = False):
    """Organize test reports into numbered subfolders under 'Test Report'."""
    import os
    import shutil
    import re
    
    # Create Test Report folder if not exists
    report_dir = 'Test Report'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        print(f"[Organize] Created directory: {report_dir}")
    
    # Find existing Test folders and determine next number
    subfolders = [f for f in os.listdir(report_dir) if os.path.isdir(os.path.join(report_dir, f)) and re.match(r'Test \d+', f)]
    numbers = [int(re.search(r'\d+', f).group()) for f in subfolders if re.search(r'\d+', f)]
    next_num = max(numbers) + 1 if numbers else 1
    test_folder = f'Test {next_num}'
    test_path = os.path.join(report_dir, test_folder)
    os.makedirs(test_path)
    print(f"[Organize] Created subfolder: {test_path}")
    
    # Files to move
    files_to_move = [
        'inference_time_compare.png',
        'inference_time.png',
        'length_vs_ppl.png',
        'perplexity_compare.png',
        'perplexity.png',
        'wikitext2_inference_time_compare.png',
        'wikitext2_perplexity_compare.png',
        'wikitext2_token_speed_compare.png',
        'wikitext2_metrics.json',
        'boolq_zeroshoot_accuracy.png',
        'hellaswag_zeroshoot_accuracy.png',
        'mmlu_zeroshoot_accuracy.png',
        'boolq_zeroshoot_metrics.json',
        'hellaswag_zeroshoot_metrics.json',
        'mmlu_zeroshoot_metrics.json',
        'zero_shot_baseline_accuracy.png',
        'zero_shot_baseline_metrics.json',
        'accuracy_compare.png',
        'accuracy_benchmark_baseline.png',
        'accuracy_benchmark_pruned.png',
        'pruning_action_usage.png',
        'inference_time_per_action.png',
        'perplexity_per_action.png',
        'token_speed_compare.png',
        'time_breakdown.png',
        'reward_progression.png',
        'quality_vs_speed.png',
        'epsilon_decay.png',
        'cumulative_reward.png',
        'vram_usage.png',
        'test_report.txt'
    ]
    if not is_report_mode:
        files_to_move.append('test_metrics.json')
    
    for file in files_to_move:
        if os.path.exists(file):
            shutil.move(file, os.path.join(test_path, file))
            print(f"[Organize] Moved {file} to {test_path}")
        else:
            print(f"[Organize] Warning: {file} not found, skipping")
    
    print(f"[Organize] Test reports organized into {test_path}")

def test_agent(model_engine, rl_agent, benchmark, num_test_episodes=10, max_new_tokens: int = 50, test_dataset: str = None, force_action=None, prompts: List[str] = None):
    """Test the trained RL agent on new prompts without training."""
    print(f"\n[Test] Evaluating trained agent on {num_test_episodes} test prompts...")
    
    # GPU warmup: flush cold-start overhead before timing
    benchmark._warmup_once(model_engine)

    # Set epsilon to 0 for pure exploitation
    original_epsilon = rl_agent.epsilon
    rl_agent.epsilon = 0.0
    metrics_list = []
    
    if prompts is not None:
        test_prompts = prompts
    elif test_dataset and test_dataset.endswith('.csv'):
        test_prompts = load_training_prompts(test_dataset, samples=max(num_test_episodes, 50), split_type='test')
    else:
        # Ensure we always have >= num_test_episodes prompts by sampling with replacement
        test_prompts = [
            "Explain quantum computing in simple terms.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Describe the water cycle.",
            "Why is exercise important for health?"
        ]

    if not test_prompts:
        test_prompts = ["Explain quantum computing in simple terms."]

    # If dataset is smaller than requested episodes, sample with replacement so plots have multiple points.
    if len(test_prompts) < num_test_episodes:
        test_prompts = random.choices(test_prompts, k=num_test_episodes)
    else:
        if prompts is None:  # Only shuffle if not pre-split
            random.shuffle(test_prompts)
    test_prompts = test_prompts[:num_test_episodes]
    
    if force_action:
        try:
            target, intensity_str = force_action.split(':')
            intensity = float(intensity_str)
            forced_action = type('Action', (), {'target': target, 'intensity': intensity, 'action_index': 0})()  # Dummy action_index
            print(f"[Test] Forcing action: {target} ({intensity}) for all episodes.")
        except ValueError:
            print(f"[Test] Invalid force_action format: {force_action}. Expected 'target:intensity'. Using RL.")
            forced_action = None
    else:
        forced_action = None
    
    for i in range(len(test_prompts)):
        prompt = test_prompts[i]
        print(f"\n[Test Episode {i+1}/{len(test_prompts)}]")
        print(f"[Test] Prompt: '{prompt[:80]}...'")
        
        # Calculate prompt perplexity and token length for metrics
        token_len = len(rl_agent.tokenizer.encode(prompt))
        prompt_ppl = benchmark._calculate_perplexity(model_engine, prompt)
        
        # Baseline: no pruning
        model_engine.restore_model()
        base_metrics = benchmark.benchmark_and_get_reward(model_engine, prompt, max_new_tokens=max_new_tokens, return_metrics=True)
        print(f"[Test Baseline] Time: {base_metrics['time_ms']:.2f}ms | Tok/s: {base_metrics['tok_s']:.2f} | PPL: {base_metrics['perplexity']:.2f}")
        
        # LCR score + early-Llama features (timed together)
        start_lcr = time.time()
        if getattr(rl_agent, 'lcr_scorer', None) is not None:
            complexity_score = float(rl_agent.lcr_scorer.score(prompt))
        else:
            llm_norm = min(1.0, token_len / 200.0)
            ppl_norm = min(1.0, prompt_ppl / 50.0)
            complexity_score = 0.6 * llm_norm + 0.4 * ppl_norm
        try:
            early_features = model_engine.extract_early_features(prompt)
        except Exception:
            early_features = {"hidden_norm": 0.0, "attn_entropy": 0.0, "attn_max": 0.0}
        lcr_inference_time_ms = (time.time() - start_lcr) * 1000
        
        # Pruned: forced or RL-selected action
        if forced_action:
            action = forced_action
            rl_inference_time_ms = 0.0  # No RL inference for forced actions
        else:
            start_rl = time.time()
            state = rl_agent._get_state_vector(prompt, prompt_ppl=prompt_ppl, token_len=token_len, lcr_score=complexity_score, early_features=early_features)
            action = rl_agent.get_action(prompt, prompt_ppl=prompt_ppl, token_len=token_len, state_tensor=state)
            rl_inference_time_ms = (time.time() - start_rl) * 1000
        
        model_engine.apply_pruning(action)
        # Track VRAM usage specifically for pruning + inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        metrics = benchmark.benchmark_and_get_reward(model_engine, prompt, max_new_tokens=max_new_tokens, return_metrics=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            vram_used_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            vram_total_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        else:
            vram_used_gb = 0.0
            vram_total_gb = 0.0
        total_pruned_time = lcr_inference_time_ms + rl_inference_time_ms + metrics['time_ms']
        print(f"[Test Pruned] Action: {action.target} ({action.intensity}) | LCR Time: {lcr_inference_time_ms:.2f}ms | RL Time: {rl_inference_time_ms:.2f}ms | Model Time: {metrics['time_ms']:.2f}ms | Total Time: {total_pruned_time:.2f}ms | Tok/s: {metrics['tok_s']:.2f} | PPL: {metrics['perplexity']:.2f} | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB")
        
        model_engine.restore_model()
        
        metrics_list.append({
            'episode': i+1,
            'token_len': token_len,
            'prompt_ppl': prompt_ppl,
            'complexity': complexity_score,
            'baseline_time_ms': base_metrics['time_ms'],
            'baseline_tok_s': base_metrics['tok_s'],
            'baseline_ppl': base_metrics['perplexity'],
            'time_ms': total_pruned_time,
            'model_time_ms': metrics['time_ms'],
            'rl_inference_time_ms': rl_inference_time_ms,
            'lcr_inference_time_ms': lcr_inference_time_ms,
            'tok_s': metrics['tok_s'],
            'ppl': metrics['perplexity'],
            'action_index': action.action_index,
            'target': action.target,
            'intensity': action.intensity,
            'vram_used_gb': vram_used_gb,
            'vram_total_gb': vram_total_gb,
        })
    
    # Generate test reports
    generate_report(metrics_list, report_filename='test_report.txt', header='Test Report')
    generate_comparative_plots(metrics_list)
    import json
    with open('test_metrics.json', 'w') as f:
        json.dump(metrics_list, f)
    organize_test_reports()
    
    # Restore original epsilon
    rl_agent.epsilon = original_epsilon

def run_wikitext_eval(model_engine, benchmark, split: str = 'test', samples: int = 200, max_new_tokens: int = 50):
    try:
        from datasets import load_dataset
    except Exception:
        print("[Eval] Please install 'datasets' to run WikiText-2 eval: pip install datasets")
        return
    print(f"\n[Eval] Running WikiText-2 ({split}) on {samples} samples...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    texts = [r['text'] for r in ds if r['text'] and r['text'].strip()]
    random.shuffle(texts)
    texts = texts[:samples]

    total_ppl, total_tok_s, total_ms = 0.0, 0.0, 0.0
    for t in texts:
        prompt = t[:512]
        start = time.time()
        _ = model_engine.generate_response(prompt, max_length=max_new_tokens)
        elapsed = time.time() - start
        ms = elapsed * 1000
        tok_s = (max_new_tokens / elapsed) if elapsed > 0 else 0.0
        ppl = benchmark._calculate_perplexity_on_continuation(model_engine, prompt, t[512:] if len(t) > 512 else "")
        total_ms += ms
        total_tok_s += tok_s
        total_ppl += ppl
    n = float(len(texts)) if texts else 1.0
    print(f"[WikiText-2] Avg PPL: {total_ppl/n:.2f} | Avg Tok/s: {total_tok_s/n:.2f} | Avg Time: {total_ms/n:.2f}ms")

def _load_streaming_samples(dataset_name: str, dataset_config: Optional[str], split: str, samples: int, seed: int = 42, buffer_size: int = 10000) -> List[dict]:
    try:
        from datasets import load_dataset
    except Exception:
        print("[Eval] Please install 'datasets' to run eval: pip install datasets")
        return []
    try:
        ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    except Exception:
        ds = None
    if ds is None:
        # Fallback: try to load only a slice of the split so we don't download the entire dataset.
        # This is still not guaranteed to be minimal for every dataset, but it's much safer than full split download.
        try:
            slice_n = max(int(samples), int(min(buffer_size, samples * 5)))
            ds = load_dataset(dataset_name, dataset_config, split=f"{split}[:{slice_n}]")
        except Exception:
            try:
                ds = load_dataset(dataset_name, dataset_config, split=split)
                print(f"[Eval] Warning: streaming and split slicing failed for {dataset_name}; this may download the full dataset.")
            except Exception:
                return []
    try:
        if getattr(ds, 'shuffle', None) is not None:
            try:
                ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
            except TypeError:
                ds = ds.shuffle(seed=seed)
    except Exception:
        pass
    out = []
    try:
        it = iter(ds)
        while len(out) < samples:
            out.append(next(it))
    except StopIteration:
        pass
    except Exception:
        try:
            for ex in itertools.islice(ds, samples):
                out.append(ex)
        except Exception:
            pass
    return out

def _parse_force_action(force_action_str: Optional[str]):
    if not force_action_str:
        return None
    try:
        t, i = force_action_str.split(':')
        return PruningAction(3, float(i), t, 0)
    except Exception:
        return None

def _tokenize_ids(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer(text, add_special_tokens=False).get('input_ids', [])
    except Exception:
        try:
            return tokenizer.encode(text)
        except Exception:
            return []

def _compute_loss_ppl_and_tokens_on_ids(
    engine: RealModelEngine,
    prompt_ids: List[int],
    cont_ids: List[int],
) -> Tuple[float, float, int]:
    cont_ids = list(cont_ids or [])
    if not cont_ids:
        return 20.0, 50000.0, 0
    ids = (prompt_ids or []) + cont_ids
    if not ids:
        return 20.0, 50000.0, 0
    input_ids = torch.tensor([ids], dtype=torch.long, device=engine.model.device)
    labels = input_ids.clone()
    pl = len(prompt_ids or [])
    if pl > 0:
        labels[:, :pl] = -100
    with torch.no_grad():
        out = engine.model(input_ids=input_ids, labels=labels)
    loss = float(out.loss.detach().float().item())
    ppl = float(torch.exp(out.loss).detach().float().item())
    if (ppl is None) or (not torch.isfinite(torch.tensor(ppl))):
        return 20.0, 50000.0, 0
    return loss, ppl, int(len(cont_ids))


def _compute_ppl_on_ids(engine: RealModelEngine, prompt_ids: List[int], cont_ids: List[int]) -> float:
    _, ppl, _ = _compute_loss_ppl_and_tokens_on_ids(engine, prompt_ids, cont_ids)
    return float(ppl)

def _score_options_mean_logprob(engine: RealModelEngine, prompt: str, options: List[str], max_seq_len: int = 512) -> List[float]:
    tok = engine.tokenizer
    base_prompt_ids = _tokenize_ids(tok, prompt)
    scores = []
    if not options:
        return scores
    option_ids_list = []
    prompt_ids_list = []
    for opt in options:
        opt_ids = _tokenize_ids(tok, opt)
        if not opt_ids:
            option_ids_list.append([])
            prompt_ids_list.append(base_prompt_ids)
            continue
        keep_prompt = base_prompt_ids
        max_prompt_len = max(1, max_seq_len - len(opt_ids))
        if len(keep_prompt) > max_prompt_len:
            keep_prompt = keep_prompt[-max_prompt_len:]
        option_ids_list.append(opt_ids)
        prompt_ids_list.append(keep_prompt)
    batch_ids = []
    for p_ids, o_ids in zip(prompt_ids_list, option_ids_list):
        batch_ids.append((p_ids or []) + (o_ids or []))
    max_len = max(len(x) for x in batch_ids) if batch_ids else 0
    if max_len <= 1:
        return [float('-inf') for _ in options]
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long, device=engine.model.device)
    attn = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=engine.model.device)
    for i, ids in enumerate(batch_ids):
        if not ids:
            continue
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=engine.model.device)
        attn[i, :len(ids)] = 1
    with torch.no_grad():
        logits = engine.model(input_ids=input_ids, attention_mask=attn).logits
        logp = torch.log_softmax(logits, dim=-1)
    for i, (p_ids, o_ids) in enumerate(zip(prompt_ids_list, option_ids_list)):
        if not o_ids:
            scores.append(float('-inf'))
            continue
        pl = len(p_ids)
        ol = len(o_ids)
        start = max(0, pl - 1)
        end = start + ol
        if end > (max_len - 1):
            scores.append(float('-inf'))
            continue
        target = input_ids[i, pl:pl + ol]
        pred_lp = logp[i, start:end, :].gather(1, target.unsqueeze(1)).squeeze(1)
        denom = float(max(1, ol))
        scores.append(float(pred_lp.sum().item() / denom))
    return scores

def run_wikitext2_comparative_eval(
    engine: RealModelEngine,
    benchmark: RealBenchmark,
    samples: int = 1000,
    split: str = 'test',
    seed: int = 42,
    max_new_tokens: int = 50,
    max_seq_len: int = 512,
    min_cont_tokens: int = 32,
    force_action_str: Optional[str] = None,
):
    print(f"\n[Eval] Running WikiText-2 comparative eval on {samples} samples...")
    rows = _load_streaming_samples('wikitext', 'wikitext-2-raw-v1', split=split, samples=samples, seed=seed)
    texts = []
    for r in rows:
        t = r.get('text') if isinstance(r, dict) else None
        if t and str(t).strip():
            texts.append(str(t))
    if not texts:
        print("[Eval] WikiText-2: no usable samples")
        return
    engine.model.eval()
    metrics = []

    def make_prompt_cont(text: str):
        ids = _tokenize_ids(engine.tokenizer, text)
        min_prompt_tokens = 8
        cont_len = max(1, int(min_cont_tokens))
        if cont_len >= int(max_seq_len):
            cont_len = max(1, int(max_seq_len) - 1)
        if len(ids) < (min_prompt_tokens + cont_len):
            return None, None

        max_prompt_len = max(1, int(max_seq_len) - cont_len)
        ideal_prompt_len = max(min_prompt_tokens, int(max_seq_len) // 2)
        prompt_len = min(max_prompt_len, ideal_prompt_len, len(ids) - cont_len)
        prompt_len = max(min_prompt_tokens, int(prompt_len))
        if prompt_len + cont_len > len(ids):
            return None, None

        prompt_ids = ids[:prompt_len]
        cont_ids = ids[prompt_len:prompt_len + cont_len]
        if len(cont_ids) < cont_len:
            return None, None
        return prompt_ids, cont_ids

    pairs = []
    skipped_too_short = 0
    for t in texts:
        prompt_ids, cont_ids = make_prompt_cont(t)
        if prompt_ids is None or cont_ids is None:
            skipped_too_short += 1
            continue
        pairs.append((prompt_ids, cont_ids))
    if not pairs:
        print("[Eval] WikiText-2: no usable prompt/continuation pairs after filtering")
        return

    print("[Eval] Baseline pass...")
    try:
        engine.restore_model()
    except Exception:
        pass
    base_times = []
    base_ppls = []
    base_losses = []
    base_eval_toks = []
    base_tok_s = []
    for i, (prompt_ids, cont_ids) in enumerate(pairs, start=1):
        prompt_text = engine.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        start = time.time()
        gen = engine.generate_response(prompt_text, max_length=max_new_tokens)
        elapsed = max(1e-12, time.time() - start)
        tok_count = len(engine.tokenizer.encode(gen))
        base_times.append(elapsed * 1000.0)
        base_tok_s.append((tok_count / elapsed) if elapsed > 0 else 0.0)
        loss, ppl, eval_toks = _compute_loss_ppl_and_tokens_on_ids(engine, prompt_ids, cont_ids)
        base_losses.append(float(loss))
        base_ppls.append(float(ppl))
        base_eval_toks.append(int(eval_toks))
        if i % 50 == 0:
            print(f"[Eval] Baseline {i}/{len(pairs)}")

    target_action = _parse_force_action(force_action_str)
    if not target_action:
        try:
            target_action = PruningAction(3, 0.25, 'transformer_layers', 0)
        except Exception:
            target_action = None
    if target_action:
        print(f"[Eval] Applying pruning action: {target_action}")
        try:
            engine.restore_model()
        except Exception:
            pass
        engine.apply_pruning(target_action)
    else:
        print("[Eval] No pruning action available; skipping pruned pass")
        return

    print("[Eval] Pruned pass...")
    pr_times = []
    pr_ppls = []
    pr_losses = []
    pr_eval_toks = []
    pr_tok_s = []
    for i, (prompt_ids, cont_ids) in enumerate(pairs, start=1):
        prompt_text = engine.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        start = time.time()
        gen = engine.generate_response(prompt_text, max_length=max_new_tokens)
        elapsed = max(1e-12, time.time() - start)
        tok_count = len(engine.tokenizer.encode(gen))
        pr_times.append(elapsed * 1000.0)
        pr_tok_s.append((tok_count / elapsed) if elapsed > 0 else 0.0)
        loss, ppl, eval_toks = _compute_loss_ppl_and_tokens_on_ids(engine, prompt_ids, cont_ids)
        pr_losses.append(float(loss))
        pr_ppls.append(float(ppl))
        pr_eval_toks.append(int(eval_toks))
        if i % 50 == 0:
            print(f"[Eval] Pruned {i}/{len(pairs)}")

    n = min(len(base_times), len(pr_times), len(base_ppls), len(pr_ppls), len(base_losses), len(pr_losses), len(base_eval_toks), len(pr_eval_toks))
    if n <= 0:
        print("[Eval] WikiText-2: no aligned samples")
        return
    for i in range(n):
        metrics.append({
            'sample': i + 1,
            'prompt_tokens': int(len(pairs[i][0])) if i < len(pairs) else 0,
            'eval_tokens': int(base_eval_toks[i]),
            'baseline_time_ms': float(base_times[i]),
            'baseline_tok_s': float(base_tok_s[i]),
            'baseline_loss': float(base_losses[i]),
            'baseline_ppl': float(base_ppls[i]),
            'time_ms': float(pr_times[i]),
            'tok_s': float(pr_tok_s[i]),
            'loss': float(pr_losses[i]),
            'ppl': float(pr_ppls[i]),
        })

    def _token_weighted_ppl(losses: List[float], tokens: List[int]) -> Tuple[float, float, int]:
        total_toks = int(sum(int(t) for t in tokens))
        if total_toks <= 0:
            return 50000.0, 20.0, 0
        weighted_loss = float(sum(float(l) * float(t) for l, t in zip(losses, tokens)))
        mean_loss = float(weighted_loss / float(total_toks))
        ppl = float(np.exp(mean_loss))
        return ppl, mean_loss, total_toks

    base_ppl_tw, base_loss_tw, base_total_toks = _token_weighted_ppl(base_losses[:n], base_eval_toks[:n])
    pr_ppl_tw, pr_loss_tw, pr_total_toks = _token_weighted_ppl(pr_losses[:n], pr_eval_toks[:n])

    import json
    with open('wikitext2_metrics.json', 'w') as f:
        json.dump({
            'samples_used': n,
            'samples_skipped_too_short': int(skipped_too_short),
            'min_cont_tokens': int(min_cont_tokens),
            'max_seq_len': int(max_seq_len),
            'baseline': {
                'avg_time_ms': float(sum(base_times[:n]) / n),
                'avg_tok_s': float(sum(base_tok_s[:n]) / n),
                # Legacy (arithmetic mean of per-sample PPL). Kept for backward-compat.
                'avg_ppl': float(sum(base_ppls[:n]) / n),
                # Defense-safe: token-weighted PPL = exp(weighted mean loss)
                'ppl_token_weighted': float(base_ppl_tw),
                'mean_loss_token_weighted': float(base_loss_tw),
                'total_eval_tokens': int(base_total_toks),
            },
            'pruned': {
                'avg_time_ms': float(sum(pr_times[:n]) / n),
                'avg_tok_s': float(sum(pr_tok_s[:n]) / n),
                # Legacy (arithmetic mean of per-sample PPL). Kept for backward-compat.
                'avg_ppl': float(sum(pr_ppls[:n]) / n),
                # Defense-safe: token-weighted PPL = exp(weighted mean loss)
                'ppl_token_weighted': float(pr_ppl_tw),
                'mean_loss_token_weighted': float(pr_loss_tw),
                'total_eval_tokens': int(pr_total_toks),
            },
            'per_sample': metrics,
        }, f)
    print("[Report] Saved wikitext2_metrics.json")

    xs = [m['sample'] for m in metrics]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, [m['baseline_time_ms'] for m in metrics], color='red', linewidth=1.2, alpha=0.85, label='Baseline Time (ms)')
    plt.plot(xs, [m['time_ms'] for m in metrics], color='blue', linewidth=1.2, alpha=0.85, label='Pruned Time (ms)')
    plt.title('WikiText-2 Inference Time (Baseline vs Pruned)')
    plt.xlabel('Sample'); plt.ylabel('Time (ms)'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig('wikitext2_inference_time_compare.png', dpi=300)
    plt.close()
    print('[Report] Saved wikitext2_inference_time_compare.png')

    plt.figure(figsize=(10, 6))
    plt.plot(xs, [m['baseline_ppl'] for m in metrics], color='red', linewidth=1.2, alpha=0.85, label='Baseline PPL')
    plt.plot(xs, [m['ppl'] for m in metrics], color='blue', linewidth=1.2, alpha=0.85, label='Pruned PPL')
    plt.title('WikiText-2 Perplexity (Baseline vs Pruned)')
    plt.xlabel('Sample'); plt.ylabel('Perplexity'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig('wikitext2_perplexity_compare.png', dpi=300)
    plt.close()
    print('[Report] Saved wikitext2_perplexity_compare.png')

    plt.figure(figsize=(10, 6))
    plt.plot(xs, [m['baseline_tok_s'] for m in metrics], color='red', linewidth=1.2, alpha=0.85, label='Baseline Tok/s')
    plt.plot(xs, [m['tok_s'] for m in metrics], color='blue', linewidth=1.2, alpha=0.85, label='Pruned Tok/s')
    plt.title('WikiText-2 Token Speed (Baseline vs Pruned)')
    plt.xlabel('Sample'); plt.ylabel('Tokens/sec'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig('wikitext2_token_speed_compare.png', dpi=300)
    plt.close()
    print('[Report] Saved wikitext2_token_speed_compare.png')

def _plot_two_bar(out_file: str, title: str, base_val: float, pruned_val: float, ylabel: str):
    plt.figure(figsize=(6.5, 5.5))
    labels = ['Baseline', 'Pruned']
    vals = [base_val, pruned_val]
    colors = ['#1f77b4', '#ff7f0e']
    bars = plt.bar(labels, vals, color=colors, alpha=0.85)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.35)
    for b in bars:
        h = float(b.get_height())
        plt.text(b.get_x() + b.get_width() / 2.0, min(99.0, h + 1.0), f"{h:.2f}%", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[Report] Saved {out_file}")


# =========================================================================
# ORACLE DATASET ZERO-SHOT EVALUATION (from CSV, not streaming)
# =========================================================================

def _load_oracle_dataset_rows(dataset_path: str, source_filter: str, split_filter: str = 'test') -> List[dict]:
    """Load rows from oracle_dataset.csv filtered by SourceDataset and Split."""
    rows = []
    if not os.path.exists(dataset_path):
        return rows
    try:
        with open(dataset_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = (row.get('SourceDataset') or '').strip().lower()
                sp = (row.get('Split') or '').strip().lower()
                if source_filter.lower() in src and sp == split_filter.lower():
                    rows.append(row)
    except Exception as e:
        print(f"[Oracle ZS] Failed to read {dataset_path}: {e}")
    return rows


def _eval_boolq_from_dataset(engine: RealModelEngine, rows: List[dict], max_seq_len: int = 512) -> dict:
    """Evaluate BoolQ zero-shot accuracy from oracle dataset rows using mean log-prob."""
    correct, total = 0, 0
    for row in rows:
        prompt_text = (row.get('Prompt') or '').strip()
        if not prompt_text:
            continue
        answer_idx_str = (row.get('AnswerIndex') or '').strip()
        if not answer_idx_str:
            continue
        try:
            gold = int(answer_idx_str)
        except Exception:
            continue
        # BoolQ: A. False, B. True → options [' A', ' B']
        options = [' A', ' B']
        prompt_for_scoring = prompt_text.rstrip() + '\nAnswer:'
        scores = _score_options_mean_logprob(engine, prompt_for_scoring, options, max_seq_len=max_seq_len)
        pred = int(np.argmax(scores)) if scores else 0
        correct += (1 if pred == gold else 0)
        total += 1
    acc = float(correct) / float(total) if total > 0 else 0.0
    return {'accuracy': acc, 'correct': correct, 'n': total}


def _eval_mmlu_from_dataset(engine: RealModelEngine, rows: List[dict], max_seq_len: int = 512) -> dict:
    """Evaluate MMLU zero-shot accuracy from oracle dataset rows using mean log-prob."""
    correct, total = 0, 0
    for row in rows:
        prompt_text = (row.get('Prompt') or '').strip()
        if not prompt_text:
            continue
        answer_idx_str = (row.get('AnswerIndex') or '').strip()
        if not answer_idx_str:
            continue
        try:
            gold = int(answer_idx_str)
        except Exception:
            continue
        # MMLU: A/B/C/D options already in prompt
        options = [' A', ' B', ' C', ' D']
        prompt_for_scoring = prompt_text.rstrip() + '\nAnswer:'
        scores = _score_options_mean_logprob(engine, prompt_for_scoring, options, max_seq_len=max_seq_len)
        pred = int(np.argmax(scores)) if scores else 0
        correct += (1 if pred == gold else 0)
        total += 1
    acc = float(correct) / float(total) if total > 0 else 0.0
    return {'accuracy': acc, 'correct': correct, 'n': total}


def run_oracle_dataset_zeroshot(
    engine: RealModelEngine,
    dataset_path: str,
    split_filter: str = 'test',
    force_action_str: Optional[str] = None,
    phase_label: str = 'Post-Training',
    max_seq_len: int = 512,
) -> dict:
    """Run zero-shot accuracy on MMLU & BoolQ from the oracle dataset CSV.
    
    Evaluates BOTH dense (before pruned) and pruned (after pruned) accuracy.
    Returns dict with results for each task.
    """
    import json as _json

    print(f"\n[Oracle ZS] === {phase_label}: Zero-Shot Evaluation from Oracle Dataset ===")
    print(f"[Oracle ZS] Dataset: {dataset_path}, Split: {split_filter}")

    # Load rows for each task
    boolq_rows = _load_oracle_dataset_rows(dataset_path, 'boolq', split_filter)
    mmlu_rows = _load_oracle_dataset_rows(dataset_path, 'mmlu', split_filter)
    print(f"[Oracle ZS] BoolQ rows: {len(boolq_rows)}, MMLU rows: {len(mmlu_rows)}")

    if not boolq_rows and not mmlu_rows:
        print("[Oracle ZS] No evaluation rows found, skipping")
        return {}

    results = {}

    # --- Dense (before pruned) evaluation ---
    print(f"[Oracle ZS] Evaluating DENSE model (before pruning)...")
    try:
        engine.restore_model()
    except Exception:
        pass
    engine.model.eval()

    if boolq_rows:
        dense_boolq = _eval_boolq_from_dataset(engine, boolq_rows, max_seq_len)
        results['boolq_dense'] = dense_boolq
        print(f"[Oracle ZS] BoolQ Dense: {dense_boolq['accuracy']*100:.2f}% ({dense_boolq['correct']}/{dense_boolq['n']})")

    if mmlu_rows:
        dense_mmlu = _eval_mmlu_from_dataset(engine, mmlu_rows, max_seq_len)
        results['mmlu_dense'] = dense_mmlu
        print(f"[Oracle ZS] MMLU Dense: {dense_mmlu['accuracy']*100:.2f}% ({dense_mmlu['correct']}/{dense_mmlu['n']})")

    # --- Pruned (after pruned) evaluation ---
    target_action = _parse_force_action(force_action_str) if force_action_str else None
    if not target_action:
        try:
            target_action = PruningAction(3, 0.25, 'transformer_layers', 0)
        except Exception:
            target_action = None

    if target_action:
        print(f"[Oracle ZS] Evaluating PRUNED model ({target_action.target} @ {target_action.intensity})...")
        try:
            engine.restore_model()
        except Exception:
            pass
        engine.apply_pruning(target_action)
        engine.model.eval()

        if boolq_rows:
            pruned_boolq = _eval_boolq_from_dataset(engine, boolq_rows, max_seq_len)
            results['boolq_pruned'] = pruned_boolq
            print(f"[Oracle ZS] BoolQ Pruned: {pruned_boolq['accuracy']*100:.2f}% ({pruned_boolq['correct']}/{pruned_boolq['n']})")

        if mmlu_rows:
            pruned_mmlu = _eval_mmlu_from_dataset(engine, mmlu_rows, max_seq_len)
            results['mmlu_pruned'] = pruned_mmlu
            print(f"[Oracle ZS] MMLU Pruned: {pruned_mmlu['accuracy']*100:.2f}% ({pruned_mmlu['correct']}/{pruned_mmlu['n']})")

        try:
            engine.restore_model()
        except Exception:
            pass
    else:
        print("[Oracle ZS] No pruning action available, skipping pruned evaluation")

    # Save results JSON
    out_json = f'oracle_zeroshot_{phase_label.lower().replace(" ", "_")}.json'
    with open(out_json, 'w') as f:
        _json.dump({'phase': phase_label, 'split': split_filter, 'dataset': dataset_path, 'results': results}, f, indent=2)
    print(f"[Oracle ZS] Saved {out_json}")

    # Generate per-phase bar chart (dense vs pruned)
    _plot_oracle_zeroshot_bars(results, phase_label)

    return results


def _plot_oracle_zeroshot_bars(results: dict, phase_label: str):
    """Plot dense vs pruned accuracy for MMLU & BoolQ."""
    tasks = []
    dense_accs = []
    pruned_accs = []

    for task_name in ['mmlu', 'boolq']:
        dk = f'{task_name}_dense'
        pk = f'{task_name}_pruned'
        if dk in results:
            tasks.append(task_name.upper())
            dense_accs.append(results[dk]['accuracy'] * 100)
            pruned_accs.append(results.get(pk, {}).get('accuracy', 0.0) * 100)

    if not tasks:
        return

    x = np.arange(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, dense_accs, width, label='Dense (Before Pruned)', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width/2, pruned_accs, width, label='Pruned (After Pruned)', color='#FF9800', alpha=0.85)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{phase_label}: Zero-Shot Accuracy — Dense vs Pruned', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., min(99, h + 1), f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fname = f'oracle_zeroshot_{phase_label.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[Report] Saved {fname}')


def generate_zeroshot_comparison_graph(train_results: dict, test_results: dict, output_path: str = 'oracle_zeroshot_comparison.png'):
    """Generate 3-bar comparison: Dense vs Pruned(Train) vs Pruned(Test) for MMLU & BoolQ.
    
    Called after both training and testing phases to show progression.
    """
    tasks = []
    dense_accs = []
    pruned_train_accs = []
    pruned_test_accs = []

    for task_name in ['mmlu', 'boolq']:
        # Use dense from either phase (should be same model)
        dense_key = f'{task_name}_dense'
        pruned_key = f'{task_name}_pruned'
        
        dense_acc = 0.0
        if dense_key in train_results:
            dense_acc = train_results[dense_key]['accuracy'] * 100
        elif dense_key in test_results:
            dense_acc = test_results[dense_key]['accuracy'] * 100

        train_pruned_acc = train_results.get(pruned_key, {}).get('accuracy', 0.0) * 100
        test_pruned_acc = test_results.get(pruned_key, {}).get('accuracy', 0.0) * 100

        if dense_acc > 0 or train_pruned_acc > 0 or test_pruned_acc > 0:
            tasks.append(task_name.upper())
            dense_accs.append(dense_acc)
            pruned_train_accs.append(train_pruned_acc)
            pruned_test_accs.append(test_pruned_acc)

    if not tasks:
        print("[Report] No comparison data available for graph")
        return

    x = np.arange(len(tasks))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 7))
    bars1 = ax.bar(x - width, dense_accs, width, label='Before Pruned (Dense)', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x, pruned_train_accs, width, label='After Pruned (Training)', color='#4CAF50', alpha=0.85)
    bars3 = ax.bar(x + width, pruned_test_accs, width, label='After Pruned (Testing)', color='#FF9800', alpha=0.85)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Zero-Shot Accuracy Comparison: Before vs After Pruning', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bars in [bars1, bars2, bars3]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., min(99, h + 0.5), f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[Report] Saved {output_path}')


def run_zero_shot_baseline_eval(engine: RealModelEngine, samples: int = 200, seed: int = 42, max_seq_len: int = 512, phase_label: str = 'Pre-Training'):
    """Evaluate DENSE (unpruned) model zero-shot accuracy on BoolQ and MMLU. Produces a combined bar chart."""
    import json as _json
    try:
        engine.restore_model()
    except Exception:
        pass
    engine.model.eval()

    tasks_config = [
        ('boolq', 'boolq', None, 'validation'),
        ('mmlu', 'cais/mmlu', 'all', 'test'),
    ]

    results = {}
    for task_name, ds_name, ds_config, ds_split in tasks_config:
        rows = _load_streaming_samples(ds_name, ds_config, split=ds_split, samples=samples, seed=seed)
        if not rows and task_name == 'mmlu':
            rows = _load_streaming_samples('mmlu', None, split='test', samples=samples, seed=seed)
        if not rows:
            print(f"[Zero-shot] {task_name}: no samples available, skipping")
            continue

        correct, total = 0, 0
        for ex in rows:
            if not isinstance(ex, dict):
                continue
            if task_name == 'boolq':
                passage = str(ex.get('passage', '')).strip()
                question = str(ex.get('question', '')).strip()
                ans = ex.get('answer', None)
                if not passage or not question or ans is None:
                    continue
                prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
                options = [' yes', ' no']
                scores = _score_options_mean_logprob(engine, prompt, options, max_seq_len=max_seq_len)
                pred = int(np.argmax(scores)) if scores else 0
                gold = 0 if bool(ans) else 1
                correct += (1 if pred == gold else 0)
                total += 1
            elif task_name == 'mmlu':
                question = str(ex.get('question', '')).strip()
                choices = ex.get('choices', None)
                ans = ex.get('answer', None)
                if not question or not choices or ans is None or len(choices) < 4:
                    continue
                gold = None
                try:
                    gold = int(ans)
                except Exception:
                    try:
                        a = str(ans).strip().upper()
                        if a in ['A', 'B', 'C', 'D']:
                            gold = ['A', 'B', 'C', 'D'].index(a)
                    except Exception:
                        pass
                if gold is None:
                    continue
                prompt = f"Question: {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
                options = [' A', ' B', ' C', ' D']
                scores = _score_options_mean_logprob(engine, prompt, options, max_seq_len=max_seq_len)
                pred = int(np.argmax(scores)) if scores else 0
                correct += (1 if pred == gold else 0)
                total += 1

        acc = float(correct) / float(total) if total > 0 else 0.0
        results[task_name] = {'accuracy': acc, 'n': total, 'correct': correct}
        print(f"[Zero-shot] {task_name}: {acc*100:.2f}% ({correct}/{total})")

    if not results:
        print("[Zero-shot] No results to plot")
        return results

    # Combined bar chart
    task_names = list(results.keys())
    accuracies = [results[t]['accuracy'] * 100 for t in task_names]
    palette = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    plt.figure(figsize=(8, 6))
    bars = plt.bar([t.upper() for t in task_names], accuracies, color=palette[:len(task_names)], alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., min(99, h + 1), f'{h:.1f}%', ha='center', va='bottom', fontsize=11)
    plt.title(f'{phase_label} Zero-Shot Accuracy (Dense Model)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('zero_shot_baseline_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[Report] Saved zero_shot_baseline_accuracy.png')

    with open('zero_shot_baseline_metrics.json', 'w') as f:
        _json.dump({'phase': phase_label, 'tasks': results}, f)
    print('[Report] Saved zero_shot_baseline_metrics.json')
    return results

def run_zeroshoot_accuracy(engine: RealModelEngine, task: str, samples: int = 1000, seed: int = 42, max_seq_len: int = 512, force_action_str: Optional[str] = None):
    task_l = str(task).lower().strip()
    if task_l == 'boolq':
        split = 'validation'
        rows = _load_streaming_samples('boolq', None, split=split, samples=samples, seed=seed)
    elif task_l == 'hellaswag':
        split = 'validation'
        rows = _load_streaming_samples('hellaswag', None, split=split, samples=samples, seed=seed)
    elif task_l == 'mmlu':
        split = 'test'
        rows = []
        for cfg in ['all', None]:
            try:
                rows = _load_streaming_samples('cais/mmlu', cfg, split=split, samples=samples, seed=seed)
                if rows:
                    break
            except Exception:
                pass
        if not rows:
            rows = _load_streaming_samples('mmlu', None, split=split, samples=samples, seed=seed)
    else:
        print(f"[Eval] Unknown task: {task}")
        return
    if not rows:
        print(f"[Eval] {task}: no samples")
        return
    engine.model.eval()

    def eval_on_rows():
        correct = 0
        total = 0
        for ex in rows:
            if not isinstance(ex, dict):
                continue
            if task_l == 'boolq':
                passage = str(ex.get('passage', '')).strip()
                question = str(ex.get('question', '')).strip()
                ans = ex.get('answer', None)
                if passage == '' or question == '' or ans is None:
                    continue
                prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
                options = [' yes', ' no']
                scores = _score_options_mean_logprob(engine, prompt, options, max_seq_len=max_seq_len)
                pred = int(np.argmax(scores)) if scores else 0
                gold = 0 if bool(ans) else 1
                correct += 1 if pred == gold else 0
                total += 1
            elif task_l == 'hellaswag':
                ctx_a = str(ex.get('ctx_a', '')).strip()
                ctx_b = str(ex.get('ctx_b', '')).strip()
                endings = ex.get('endings', None)
                label = ex.get('label', None)
                if not endings or label is None:
                    continue
                try:
                    gold = int(label)
                except Exception:
                    continue
                prompt = (ctx_a + ' ' + ctx_b).strip()
                options = []
                for e in endings:
                    et = str(e)
                    if not et.startswith(' '):
                        et = ' ' + et
                    options.append(et)
                scores = _score_options_mean_logprob(engine, prompt, options, max_seq_len=max_seq_len)
                pred = int(np.argmax(scores)) if scores else 0
                correct += 1 if pred == gold else 0
                total += 1
            else:
                question = str(ex.get('question', '')).strip()
                choices = ex.get('choices', None)
                ans = ex.get('answer', None)
                if question == '' or not choices or ans is None:
                    continue
                gold = None
                # Support multiple common MMLU answer formats: int index (0-3) or letter (A-D).
                try:
                    gold = int(ans)
                except Exception:
                    try:
                        a = str(ans).strip().upper()
                        if a in ['A', 'B', 'C', 'D']:
                            gold = ['A', 'B', 'C', 'D'].index(a)
                        elif a in ['0', '1', '2', '3']:
                            gold = int(a)
                    except Exception:
                        gold = None
                if gold is None:
                    continue
                if len(choices) < 4:
                    continue
                prompt = f"Question: {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
                options = [' A', ' B', ' C', ' D']
                scores = _score_options_mean_logprob(engine, prompt, options, max_seq_len=max_seq_len)
                pred = int(np.argmax(scores)) if scores else 0
                correct += 1 if pred == gold else 0
                total += 1
        acc = (float(correct) / float(total)) if total > 0 else 0.0
        return acc, total

    print(f"\n[Eval] Zero-shot {task} baseline...")
    try:
        engine.restore_model()
    except Exception:
        pass
    base_acc, base_n = eval_on_rows()

    target_action = _parse_force_action(force_action_str)
    if not target_action:
        try:
            target_action = PruningAction(3, 0.25, 'transformer_layers', 0)
        except Exception:
            target_action = None
    if not target_action:
        print("[Eval] No pruning action available")
        return
    print(f"[Eval] Zero-shot {task} pruned...")
    try:
        engine.restore_model()
    except Exception:
        pass
    engine.apply_pruning(target_action)
    pr_acc, pr_n = eval_on_rows()

    import json
    out_json = f"{task_l}_zeroshoot_metrics.json"
    with open(out_json, 'w') as f:
        json.dump({
            'task': task_l,
            'samples_requested': int(samples),
            'baseline': {'accuracy': float(base_acc), 'n': int(base_n)},
            'pruned': {'accuracy': float(pr_acc), 'n': int(pr_n)},
        }, f)
    print(f"[Report] Saved {out_json}")
    out_png = f"{task_l}_zeroshoot_accuracy.png"
    _plot_two_bar(out_png, f"{task_l.upper()} Zero-shot Accuracy", base_acc * 100.0, pr_acc * 100.0, 'Accuracy (%)')

def run_lm_eval_harness(model_engine, tasks_list: List[str], batch_size: int = 1, export_dir: str = 'export/pruned_model', limit: int = None):
    """Run lm-eval-harness tasks and generate accuracy plot."""
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except Exception:
        print("[lm-eval] Please install 'lm-eval': pip install lm-eval")
        return

    # Save current model/tokenizer
    model_engine.save_pretrained(export_dir)
    
    # AGGRESSIVE MEMORY CLEANUP for 8GB GPU
    # Completely delete the model reference to force Python/PyTorch to release the memory
    device_str = str(model_engine.model.device) # 'cuda:0' or 'cpu'
    print(f"[lm-eval] Deleting original model from memory to free VRAM (was on {device_str})...")
    
    # 1. Move to CPU first (sometimes helps release CUDA context)
    model_engine.model.to('cpu')
    
    # 2. Delete the object
    del model_engine.model
    torch.cuda.empty_cache()
    
    # 3. Force Garbage Collection
    import gc
    gc.collect()
    
    # 4. Try to set expandable segments to avoid fragmentation (if PyTorch > 2.0)
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    except Exception:
        pass

    model_args = {
        'pretrained': export_dir,
        'tokenizer': export_dir,
        'dtype': 'float16', # Force float16 to save memory
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': batch_size,
        'trust_remote_code': True,
    }
    
    print(f"[lm-eval] Evaluating tasks: {tasks_list} | Batch Size: {batch_size} | Limit: {limit}")
    try:
        results = evaluator.simple_evaluate(
            model=HFLM(**model_args),
            tasks=tasks_list,
            batch_size=batch_size,
            limit=limit,
        )
    except Exception as e:
        print(f"[lm-eval] Error during evaluation: {e}")
        return
    
    # Note: We destroyed the model, so we cannot verify/restore it easily
    # But since this is the end of the script (test mode), it is acceptable.
    print("[lm-eval] Evaluation complete.")
    
    # JSON Output
    print("[lm-eval] Results summary:")
    try:
        import json
        res_dict = results.get('results', {}) if isinstance(results, dict) else {}
        print(json.dumps(res_dict if res_dict else results, indent=2))
        
        # Plotting
        if res_dict:
            try:
                task_names = []
                acc_scores = []
                
                # Special handling for MMLU aggregation
                mmlu_scores = []
                other_scores = {}
                
                for task, metrics in res_dict.items():
                    # Check for accuracy metric
                    val = metrics.get('acc_norm,none') or metrics.get('acc,none') or metrics.get('acc_norm') or metrics.get('acc')
                    if val is not None:
                        # lm-eval sometimes returns tasks like mmlu_*, hendrycksTest-*, etc.
                        if 'mmlu' in task.lower() or 'hendrycks' in task.lower():
                            mmlu_scores.append(val)
                        else:
                            short_name = task.split('_')[0]
                            other_scores[short_name] = val
                
                # Add non-MMLU tasks first
                for name, score in other_scores.items():
                    task_names.append(name)
                    acc_scores.append(score * 100)
                
                # Add MMLU average if present
                if mmlu_scores:
                    avg_mmlu = sum(mmlu_scores) / len(mmlu_scores)
                    task_names.append('MMLU (Avg)')
                    acc_scores.append(avg_mmlu * 100)
                
                if task_names:
                    plt.figure(figsize=(10, 6))
                    # Dynamic colors based on count
                    colors = plt.cm.viridis(np.linspace(0, 1, len(task_names)))
                    bars = plt.bar(task_names, acc_scores, color=colors, alpha=0.8)
                    plt.title(f'Accuracy Benchmark (Limit={limit or "All"})', fontsize=16, fontweight='bold')
                    plt.xlabel('Datasets', fontsize=12)
                    plt.ylabel('Accuracy (%)', fontsize=12)
                    plt.ylim(0, 100)
                    plt.grid(axis='y', linestyle='--', alpha=0.5)
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{height:.1f}%', ha='center', va='bottom')
                    
                    out_file = 'accuracy_benchmark.png'
                    plt.tight_layout()
                    plt.savefig(out_file, dpi=300)
                    plt.close()
                    print(f"[lm-eval] Accuracy graph saved to {out_file}")
            except Exception as e:
                print(f"[lm-eval] Failed to generate plot: {e}")

    except Exception:
        print(results)

def _evaluate_model_path(model_path: str, tasks_list: List[str], batch_size: int = 1, limit: int = None) -> dict:
    """Helper to evaluate a specifically saved model path to ensure VRAM safety."""
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    
    # 1. Force Garbage Collection before loading new model
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    except:
        pass

    print(f"[lm-eval] Loading model from {model_path} for evaluation...")
    model_args = {
        'pretrained': model_path,
        'tokenizer': model_path,
        'dtype': 'float16', 
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': batch_size,
        'trust_remote_code': True,
    }
    
    try:
        results = evaluator.simple_evaluate(
            model=HFLM(**model_args),
            tasks=tasks_list,
            batch_size=batch_size,
            limit=limit,
        )
        return results
    except Exception as e:
        print(f"[lm-eval] Error during evaluation of {model_path}: {e}")
        return {}
    finally:
        # Cleanup
        del model_args
        torch.cuda.empty_cache()
        gc.collect()

def _plot_comparative_accuracy(base_res, pruned_res, limit):
    """Generate Grouped Bar Chart for Baseline vs Pruned"""
    try:
        tasks = []
        base_scores = []
        pruned_scores = []
        
        # Helper to parse specific tasks (BoolQ, HellaSwag, MMLU Avg)
        def parse_res(res):
            data = {}
            mmlu_vals = []
            res_dict = res.get('results', {})
            for t, m in res_dict.items():
                val = m.get('acc_norm,none') or m.get('acc,none') or m.get('acc_norm') or m.get('acc')
                if val:
                    if 'mmlu' in t: mmlu_vals.append(val)
                    else: data[t.split('_')[0]] = val * 100
            if mmlu_vals: data['MMLU (Avg)'] = (sum(mmlu_vals)/len(mmlu_vals))*100
            return data

        b_data = parse_res(base_res)
        p_data = parse_res(pruned_res)
        
        all_keys = sorted(list(set(b_data.keys()) | set(p_data.keys())))
        if not all_keys: return
        
        for k in all_keys:
            tasks.append(k)
            base_scores.append(b_data.get(k, 0))
            pruned_scores.append(p_data.get(k, 0))
            
        x = np.arange(len(tasks))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, base_scores, width, label='Baseline', color='#1f77b4')
        plt.bar(x + width/2, pruned_scores, width, label='Pruned', color='#ff7f0e')
        
        plt.xlabel('Benchmarks')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy Comparison (Baseline vs Pruned)\nLimit={limit or "All"}')
        plt.xticks(x, tasks)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('accuracy_compare.png', dpi=300)
        print("[Report] Saved accuracy_compare.png")
    except Exception as e:
        print(f"[Report] Failed to plot comparison: {e}")

def _plot_accuracy_benchmark_single(results: dict, limit: int, out_file: str, title_prefix: str):
    """Plot a single-model accuracy benchmark bar chart from lm-eval results."""
    try:
        import numpy as np
        res_dict = results.get('results', {}) if isinstance(results, dict) else {}
        if not res_dict:
            return

        task_names = []
        acc_scores = []

        mmlu_scores = []
        other_scores = {}
        for task, metrics in res_dict.items():
            val = metrics.get('acc_norm,none') or metrics.get('acc,none') or metrics.get('acc_norm') or metrics.get('acc')
            if val is None:
                continue
            t = str(task).lower()
            if ('mmlu' in t) or ('hendrycks' in t):
                mmlu_scores.append(float(val))
            else:
                short_name = str(task).split('_')[0]
                other_scores[short_name] = float(val)

        # Stable order: known tasks first if present
        preferred = ['boolq', 'hellaswag']
        for p in preferred:
            if p in other_scores:
                task_names.append(p)
                acc_scores.append(other_scores[p] * 100)
        for name, score in sorted(other_scores.items()):
            if name in preferred:
                continue
            task_names.append(name)
            acc_scores.append(score * 100)

        if mmlu_scores:
            task_names.append('MMLU (Avg)')
            acc_scores.append((sum(mmlu_scores) / len(mmlu_scores)) * 100)

        if not task_names:
            return

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(task_names)))
        bars = plt.bar(task_names, acc_scores, color=colors, alpha=0.85)
        plt.title(f"{title_prefix} (Limit={limit or 'All'})", fontsize=14, fontweight='bold')
        plt.xlabel('Datasets', fontsize=11)
        plt.ylabel('Accuracy (%)', fontsize=11)
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # Value labels: keep clean and non-overlapping
        for bar in bars:
            height = float(bar.get_height())
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(99.0, height + 1.0),
                f"{height:.1f}%",
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=0,
            )

        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"[lm-eval] Accuracy graph saved to {out_file}")
    except Exception as e:
        print(f"[lm-eval] Failed to generate plot {out_file}: {e}")

def run_comparative_eval(engine, agent, tasks_list: List[str], batch_size: int = 1, limit: int = None, force_action_str: str = None):
    """Run evaluation on Baseline AND Pruned model, then plot comparison."""
    export_base = 'export/baseline_model'
    export_pruned = 'export/pruned_model'
    
    # 1. Evaluate Baseline
    print("\n[Comparative Eval] Preparing BASELINE model...")
    if hasattr(engine, 'restore_model'):
        engine.restore_model()
        
    engine.save_pretrained(export_base)
    # Aggressive cleanup of CURRENT engine
    print("[Comparative Eval] Offloading current engine to CPU...")
    engine.model.to('cpu')
    torch.cuda.empty_cache()
    
    print("[Comparative Eval] Running Baseline Evaluation...")
    base_results = _evaluate_model_path(export_base, tasks_list, batch_size, limit)
    
    # 2. Evaluate Pruned
    print("\n[Comparative Eval] Preparing PRUNED model...")
    # Fix: engine.device is not defined, use detection
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine.model.to(default_device) # Restore for pruning
    
    target_action = None
    if force_action_str:
        try:
            t, i = force_action_str.split(':')
            target_action = PruningAction(3, float(i), t, 0)
            print(f"[Comparative Eval] Applying forced action: {target_action}")
        except: pass
    
    if not target_action:
        # Default representative action: 25% layer skipping
        # Assuming PruningAction is available in scope
        try:
            target_action = PruningAction(3, 0.25, "transformer_layers", 0)
        except:
             # Fallback if PruningAction class not directly available (rare but possible)
             pass
        print(f"[Comparative Eval] Using default representative action: {target_action}")

    if target_action:
        engine.apply_pruning(target_action)
    
    engine.save_pretrained(export_pruned)
    
    engine.model.to('cpu')
    torch.cuda.empty_cache()

    print("[Comparative Eval] Running Pruned Evaluation...")
    pruned_results = _evaluate_model_path(export_pruned, tasks_list, batch_size, limit)
    
    # Restore
    engine.model.to(default_device)
    engine.restore_model()

    # 3. Process & Plot Results
    _plot_comparative_accuracy(base_results, pruned_results, limit)
    _plot_accuracy_benchmark_single(base_results, limit, out_file='accuracy_benchmark_baseline.png', title_prefix='Accuracy Benchmark (Baseline)')
    _plot_accuracy_benchmark_single(pruned_results, limit, out_file='accuracy_benchmark_pruned.png', title_prefix='Accuracy Benchmark (Pruned)')
    
    # NEW: Overall Dashboard
    try:
        import dashboard_gen
        
        # Calculate Average Accuracy
        base_acc = _get_acc_from_results(base_results)
        pruned_acc = _get_acc_from_results(pruned_results)
        
        # For PPL and Time, we don't have them in this specific run scope easily unless we run them.
        # But per user request, we want an overall graph.
        # We will use "Placeholder" or "Test Report" values if we had them.
        # For now, since this script ONLY runs Accuracy, we will set PPL/Time to 0 or placeholders
        # and rely on the USER knowing they come from different tests, OR we can try to run a quick PPL/Time check here.
        
        # IMPROVEMENT: Run a quick Benchmark for PPL/Time here so the dashboard is real.
        print("[Comparative Eval] Running Quick Latency/PPL Benchmark for Dashboard...")
        # Restore Baseline for Bench
        engine.model.to('cpu')
        del engine.model
        torch.cuda.empty_cache()
        if hasattr(engine, 'restore_model'): engine.restore_model()
        
        # Base Bench
        engine.model.to(default_device)
        b_bench = engine.benchmark_and_get_reward(engine, "The quick brown fox", max_new_tokens=50, return_metrics=True)
        
        # Pruned Bench
        engine.apply_pruning(target_action)
        p_bench = engine.benchmark_and_get_reward(engine, "The quick brown fox", max_new_tokens=50, return_metrics=True)
        
        base_metrics = {'acc': base_acc, 'ppl': b_bench['perplexity'], 'time': b_bench['time_ms']}
        pruned_metrics = {'acc': pruned_acc, 'ppl': p_bench['perplexity'], 'time': p_bench['time_ms']}
        
        dashboard_gen.plot_dashboard(base_metrics, pruned_metrics)
        
    except Exception as e:
        print(f"[Dashboard] Failed to generate overall dashboard: {e}")

    print("[Comparative Eval] Baseline Results Summary:")
    try:
        import json
        print(json.dumps(base_results.get('results', base_results), indent=2))
        print("[Comparative Eval] Pruned Results Summary:")
        print(json.dumps(pruned_results.get('results', pruned_results), indent=2))
    except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Pruning RL Controller")
    parser.add_argument('--mode', choices=['train','test','report'], default='test')
    parser.add_argument('--checkpoint', type=str, default=os.path.join('checkpoints','rl_policy.pt'))
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-new-tokens', type=int, default=50)
    parser.add_argument('--wikitext-samples', type=int, default=200)
    parser.add_argument('--wikitext2', action='store_true', help='Run WikiText-2 subset (PPL + inference) evaluation only')
    parser.add_argument('--boolq', action='store_true', help='Run BoolQ zero-shot accuracy evaluation only')
    parser.add_argument('--hellaswag', action='store_true', help='Run HellaSwag zero-shot accuracy evaluation only')
    parser.add_argument('--mmlu', action='store_true', help='Run MMLU zero-shot accuracy evaluation only')
    parser.add_argument('--eval-samples', type=int, default=1000)
    parser.add_argument('--eval-seed', type=int, default=42)
    parser.add_argument('--eval-max-seq-len', type=int, default=512)
    parser.add_argument('--wikitext-min-cont-tokens', type=int, default=32, help='Minimum continuation tokens used when computing WikiText-2 perplexity (avoids degenerate 1-token PPL).')
    parser.add_argument('--train-dataset', type=str, default='lcr_mixture.final.csv')
    parser.add_argument('--test-dataset', type=str, default='lcr_mixture.final.csv')
    parser.add_argument('--train-split', type=str, default='train')
    parser.add_argument('--train-samples', type=int, default=5000)
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'gpu', 'auto'], help='Device to load model on: cpu, gpu, or auto')
    parser.add_argument('--static-profiles', action='store_true', help='Enable prebuilt static pruning profiles (Phase A)')
    parser.add_argument('--sparsity-2to4', action='store_true', help='Enable 2:4 semi-structured sparsity packing on supported GPUs (Phase B)')
    parser.add_argument('--compile', dest='compile_profiles', action='store_true', help='Compile profiles with torch.compile if available (Phase B)')
    parser.add_argument('--kv-compress', action='store_true', help='Enable KV-cache compression scaffold (Phase D)')
    parser.add_argument('--kv-keep-ratio', type=float, default=1.0, help='KV tokens keep ratio [0,1] (Phase D)')
    parser.add_argument('--split-ratio', type=float, default=1.0, help='Train-test split ratio (e.g. 0.7 for 70%% train, 30%% test). Default: 1.0 (no split)')
    parser.add_argument('--test-samples', type=int, default=100, help='Max test episodes for auto-test after training (default: 100)')
    parser.add_argument('--force-action', type=str, default=None, help='Force a specific action for test mode (format: target:intensity, e.g., transformer_layers:0.25)')
    parser.add_argument('--lm-eval', action='store_true', help='Enable lm-eval harness benchmarks (BoolQ, MMLU, HellaSwag)')
    parser.add_argument('--eval-limit', type=int, default=None, help='Limit number of samples per lm-eval task (e.g., 100)')
    parser.add_argument('--eval-batch-size', type=int, default=1, help='Batch size for lm-eval')
    parser.add_argument('--eval-tasks', type=str, default='boolq,hellaswag,mmlu', help='Comma-separated list of lm-eval tasks')
    args = parser.parse_args()
    if args.mode == 'train':
        main(num_episodes=args.episodes, checkpoint_path=args.checkpoint, max_new_tokens=args.max_new_tokens,
             train_dataset=args.train_dataset, train_split=args.train_split, train_samples=args.train_samples, split_type='train', device=args.device,
             static_profiles=args.static_profiles, sparsity_2to4=args.sparsity_2to4, compile_profiles=args.compile_profiles,
             kv_compress=args.kv_compress, kv_keep_ratio=args.kv_keep_ratio, split_ratio=args.split_ratio, test_samples=args.test_samples)
    elif args.mode == 'report':
        import json
        with open('training_metrics.json', 'r') as f:
            metrics_list = json.load(f)
        generate_report(metrics_list)
        generate_comparative_plots(metrics_list)
        # Organize training reports into folders
        organize_training_reports(is_report_mode=True)
    else:
        engine = RealModelEngine(device=args.device, enable_static_profiles=args.static_profiles, enable_2to4=args.sparsity_2to4, enable_compile=args.compile_profiles, enable_kv_compression=args.kv_compress, kv_keep_ratio=args.kv_keep_ratio)
        agent = RLControllerAgent(engine.tokenizer)
        if os.path.exists(args.checkpoint):
            agent.load(args.checkpoint)
        bench = RealBenchmark()
        specific_eval = bool(getattr(args, 'wikitext2', False) or getattr(args, 'boolq', False) or getattr(args, 'hellaswag', False) or getattr(args, 'mmlu', False))
        if not specific_eval:
            # Pre-test zero-shot accuracy evaluation (dense model baseline)
            print("\n[System] Running pre-test zero-shot accuracy evaluation...")
            try:
                run_zero_shot_baseline_eval(engine, samples=200, phase_label='Pre-Test')
            except Exception as e:
                print(f"[Zero-shot] Warning: pre-test eval skipped due to error: {e}")
            test_agent(engine, agent, bench, num_test_episodes=args.episodes, max_new_tokens=args.max_new_tokens, test_dataset=args.test_dataset, force_action=args.force_action)

            # Run oracle dataset zero-shot after testing (MMLU & BoolQ)
            _test_ds = args.test_dataset
            if _test_ds and _test_ds.endswith('.csv') and os.path.exists(_test_ds):
                try:
                    run_oracle_dataset_zeroshot(
                        engine, _test_ds, split_filter='test',
                        force_action_str=args.force_action,
                        phase_label='Post-Testing', max_seq_len=args.eval_max_seq_len,
                    )
                except Exception as e:
                    print(f"[Oracle ZS] Post-testing zero-shot eval failed: {e}")

            if getattr(args, 'wikitext_samples', 0):
                try:
                    engine.restore_model()
                except Exception:
                    pass
                run_wikitext_eval(engine, bench, split='test', samples=args.wikitext_samples, max_new_tokens=args.max_new_tokens)
            if hasattr(args, 'lm_eval') and args.lm_eval:
                tasks = [t.strip() for t in args.eval_tasks.split(',')]
                run_comparative_eval(engine, agent, tasks_list=tasks, batch_size=args.eval_batch_size, limit=args.eval_limit, force_action_str=args.force_action)
        else:
            if getattr(args, 'wikitext2', False):
                run_wikitext2_comparative_eval(engine, bench, samples=args.eval_samples, split='test', seed=args.eval_seed, max_new_tokens=args.max_new_tokens, max_seq_len=args.eval_max_seq_len, min_cont_tokens=args.wikitext_min_cont_tokens, force_action_str=args.force_action)
            if getattr(args, 'boolq', False):
                run_zeroshoot_accuracy(engine, 'boolq', samples=args.eval_samples, seed=args.eval_seed, max_seq_len=args.eval_max_seq_len, force_action_str=args.force_action)
            if getattr(args, 'hellaswag', False):
                run_zeroshoot_accuracy(engine, 'hellaswag', samples=args.eval_samples, seed=args.eval_seed, max_seq_len=args.eval_max_seq_len, force_action_str=args.force_action)
            if getattr(args, 'mmlu', False):
                run_zeroshoot_accuracy(engine, 'mmlu', samples=args.eval_samples, seed=args.eval_seed, max_seq_len=args.eval_max_seq_len, force_action_str=args.force_action)
            organize_test_reports()
