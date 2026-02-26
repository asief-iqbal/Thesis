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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt
import argparse
import os
import re
import itertools
import warnings
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
    """Defines all possible pruning actions: none, heads, layers at various intensities."""
    def __init__(self):
        # Include layer skipping alongside heads (layer skipping capped inside engine at <=12.5%)
        self.actions = [
            PruningAction(level=0, intensity=0.0, target="none", action_index=0),
            # Attention heads (structural, GQA-safe): 5–50%
            PruningAction(level=2, intensity=0.05, target="attention_heads", action_index=1),
            PruningAction(level=2, intensity=0.10, target="attention_heads", action_index=2),
            PruningAction(level=2, intensity=0.15, target="attention_heads", action_index=3),
            PruningAction(level=2, intensity=0.20, target="attention_heads", action_index=4),
            PruningAction(level=2, intensity=0.25, target="attention_heads", action_index=5),
            PruningAction(level=2, intensity=0.30, target="attention_heads", action_index=6),
            PruningAction(level=2, intensity=0.50, target="attention_heads", action_index=7),
            # Transformer layers (functional skipping): 5–50%
            PruningAction(level=3, intensity=0.05, target="transformer_layers", action_index=8),
            PruningAction(level=3, intensity=0.10, target="transformer_layers", action_index=9),
            PruningAction(level=3, intensity=0.15, target="transformer_layers", action_index=10),
            PruningAction(level=3, intensity=0.20, target="transformer_layers", action_index=11),
            PruningAction(level=3, intensity=0.25, target="transformer_layers", action_index=12),
            PruningAction(level=3, intensity=0.30, target="transformer_layers", action_index=13),
            PruningAction(level=3, intensity=0.50, target="transformer_layers", action_index=14),
        ]
        print(f"[RL Agent] Action space initialized with {len(self.actions)} actions.")

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

        self.state_dim = 7
        self.action_dim = len(self.action_space.actions)
        
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        
        # Full exploration by default; slight decay after warm-up for convergence
        self.epsilon, self.epsilon_decay, self.epsilon_min, self.gamma = 1.0, 0.999, 0.1, 0.95

        # Replay buffer and training params
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.target_update_interval = 200  # steps
        self.train_steps = 0

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)
        print(f"[RL Agent] Checkpoint saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
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
        self.epsilon = 0.0  # exploitation by default when loading
        print(f"[RL Agent] Checkpoint loaded from {path}")

    def _get_state_vector(self, prompt: str, prompt_ppl: Optional[float] = None, token_len: Optional[int] = None) -> torch.Tensor:
        device_state = self.device_monitor.get_state()
        
        # Enhanced prompt complexity: token length, perplexity, math density, code density
        tokens = token_len if token_len is not None else len(self.tokenizer.encode(prompt))
        
        # Analyze content for structural complexity (math/code)
        math_ops, code_syms = self._analyze_prompt_content(prompt)
        text_len = max(1, len(prompt))
        math_density = math_ops / text_len
        code_density = code_syms / text_len
        
        # Normalization
        llm_norm = min(1.0, tokens / 200.0)
        ppl_norm = min(1.0, prompt_ppl / 50.0) if prompt_ppl is not None else 0.5
        math_norm = min(1.0, math_density / 0.05)  # 5% math symbols is very high
        code_norm = min(1.0, code_density / 0.05)  # 5% code symbols is very high
        
        # Weighted complexity score
        complexity_score = (0.4 * llm_norm) + (0.3 * ppl_norm) + (0.15 * math_norm) + (0.15 * code_norm)
        
        state = [
            device_state.cpu_utilization / 100.0,
            device_state.memory_available_gb / 16.0,
            device_state.battery_percent / 100.0,
            float(device_state.gpu_available),
            device_state.gpu_memory_free_gb / 24.0,
            device_state.gpu_utilization / 100.0,
            complexity_score
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
        # RL-driven action selection: epsilon-greedy
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            state = state_tensor if state_tensor is not None else self._get_state_vector(prompt, prompt_ppl, token_len)
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_index = q_values.argmax().item()
        
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
            _ = engine.generate_response("Hello", max_length=4)
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
            return 50000.0 # Return a high penalty value.
        return ppl

    def _calculate_perplexity(self, engine: RealModelEngine, text: str) -> float:
        inputs = engine.tokenizer(text, return_tensors="pt").to(engine.model.device)
        with torch.no_grad():
            outputs = engine.model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()

    def benchmark_and_get_reward(self, engine: RealModelEngine, prompt: str, max_new_tokens: int = 50, return_metrics: bool = False, profile: bool = False):
        # Use planned token budget for fair throughput comparison
        planned_tokens = max_new_tokens
        start_time = time.time()
        if profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                generated_response = engine.generate_response(prompt, max_length=planned_tokens)
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        else:
            generated_response = engine.generate_response(prompt, max_length=planned_tokens)
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
    plt.figure(figsize=(10,6))
    plt.scatter(e_b, base_tok_s, color='red', alpha=0.6, label='Baseline Tok/s')
    plt.scatter(e_p, pruned_tok_s, color='blue', alpha=0.6, label='Pruned Tok/s')
    if len(e_b) > 1:
        coeff = np.polyfit(e_b, base_tok_s, 1)
        plt.plot(e_b, np.polyval(coeff, e_b), color='red')
    if len(e_p) > 1:
        coeff = np.polyfit(e_p, pruned_tok_s, 1)
        plt.plot(e_p, np.polyval(coeff, e_p), color='blue')
    plt.title('Token Speed per Episode (Baseline vs Pruned)')
    plt.xlabel('Episode'); plt.ylabel('Tokens/sec'); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig('token_speed_compare.png'); plt.close(); print('[Report] Saved token_speed_compare.png')

    # 2) Inference time comparison
    base_time = [m['baseline_time_ms'] for m in metrics]
    pruned_time = [m['time_ms'] for m in metrics]
    e_b, base_time = remove_outliers_xy(episodes, base_time)
    e_p, pruned_time = remove_outliers_xy(episodes, pruned_time)
    plt.figure(figsize=(10,6))
    plt.scatter(e_b, base_time, color='red', alpha=0.6, label='Baseline Time (ms)')
    plt.scatter(e_p, pruned_time, color='blue', alpha=0.6, label='Pruned Time (ms)')
    if len(e_b) > 1:
        coeff = np.polyfit(e_b, base_time, 1)
        plt.plot(e_b, np.polyval(coeff, e_b), color='red')
    if len(e_p) > 1:
        coeff = np.polyfit(e_p, pruned_time, 1)
        plt.plot(e_p, np.polyval(coeff, e_p), color='blue')
    plt.title('Inference Time per Episode (Baseline vs Pruned)')
    plt.xlabel('Episode'); plt.ylabel('Time (ms)'); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig('inference_time_compare.png'); plt.close(); print('[Report] Saved inference_time_compare.png')

    # 3) Perplexity comparison
    base_ppl = [m['baseline_ppl'] for m in metrics]
    pruned_ppl = [m['ppl'] for m in metrics]
    e_b, base_ppl = remove_outliers_xy(episodes, base_ppl)
    e_p, pruned_ppl = remove_outliers_xy(episodes, pruned_ppl)
    plt.figure(figsize=(10,6))
    plt.scatter(e_b, base_ppl, color='red', alpha=0.6, label='Baseline PPL')
    plt.scatter(e_p, pruned_ppl, color='blue', alpha=0.6, label='Pruned PPL')
    if len(e_b) > 1:
        coeff = np.polyfit(e_b, base_ppl, 1)
        plt.plot(e_b, np.polyval(coeff, e_b), color='red')
    if len(e_p) > 1:
        coeff = np.polyfit(e_p, pruned_ppl, 1)
        plt.plot(e_p, np.polyval(coeff, e_p), color='blue')
    plt.title('Perplexity per Episode (Baseline vs Pruned)')
    plt.xlabel('Episode'); plt.ylabel('Perplexity'); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig('perplexity_compare.png'); plt.close(); print('[Report] Saved perplexity_compare.png')

    # 4) Token length vs prompt perplexity correlation
    token_lens = [m['token_len'] for m in metrics]
    prompt_ppls = [m['prompt_ppl'] for m in metrics]
    x, y = remove_outliers_xy(token_lens, prompt_ppls)
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, color='purple', alpha=0.6, label='TokenLen vs Prompt PPL')
    if len(x) > 1:
        coeff = np.polyfit(x, y, 1)
        plt.plot(x, np.polyval(coeff, x), color='purple')
    plt.title('Correlation: Token Length vs Prompt Perplexity')
    plt.xlabel('Token Length'); plt.ylabel('Prompt PPL'); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig('length_vs_ppl.png'); plt.close(); print('[Report] Saved length_vs_ppl.png')

    # 5) Time breakdown: RL Agent Time, Model Time, Total Time per Episode (Stacked Bar Chart)
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
    rl_times_filtered = [rl_times[i] for i in filtered_indices]
    model_times_filtered = [model_times[i] for i in filtered_indices]
    total_times_filtered = [total_times[i] for i in filtered_indices]
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(episodes_filtered))
    plt.bar(x, rl_times_filtered, color='orange', label='RL Agent Time (ms)', alpha=0.8)
    plt.bar(x, model_times_filtered, bottom=rl_times_filtered, color='blue', label='Model Time (ms)', alpha=0.8)
    plt.plot(x, total_times_filtered, color='red', marker='o', linestyle='-', linewidth=2, markersize=4, label='Total Time (ms)')
    plt.title('Inference Time Breakdown per Episode', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.xticks(x, episodes_filtered, rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[Report] Time breakdown plot saved to time_breakdown.png")

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

    # Save report to file
    with open(report_filename, 'w') as f:
        f.write(f"{header}\n")
        f.write(f"Episodes: {n}\n")
        f.write(f"Overall Avg Time: {total_time/n:.2f}ms\n")
        f.write(f"Overall Avg PPL: {total_ppl/n:.2f}\n\n")
        f.write("Averages by Prune Type and Intensity:\n")
        for (target, intensity), ms in groups.items():
            avg_time = sum(m['time_ms'] for m in ms) / len(ms)
            avg_ppl = sum(m['ppl'] for m in ms) / len(ms)
            f.write(f"  {target} {intensity}: Avg Time {avg_time:.2f}ms, Avg PPL {avg_ppl:.2f} ({len(ms)} samples)\n")
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
    """
    try:
        from datasets import load_dataset
    except Exception:
        print("[Train] Install 'datasets' to use training datasets. Falling back to hardcoded prompts.")
        return [
            "What is the capital of France?",
            "Explain the concept of machine learning in simple terms.",
            "Why is the sky blue during the day?",
            "Write a python function to calculate the factorial of a number.",
        ]

    # Use RedPajama by default if the user hasn't specified a custom CSV
    name = dataset_name.strip()
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
         kv_keep_ratio: float = 1.0):
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
    prompt_pool = load_training_prompts(train_dataset, split=train_split, samples=train_samples, split_type=split_type)
    # Honor CLI episodes: cap to requested number
    limit = min(num_episodes, len(prompt_pool))
    prompt_pool = prompt_pool[:limit]
    num_episodes = limit

    # Calibrate importance scores for heads/FFN/layers using a small subset
    try:
        calib_samples = min(64, len(prompt_pool))
        model_engine.calibrate_importances(prompt_pool, max_samples=calib_samples, max_seq_len=128)
    except Exception as e:
        print(f"[Calib] Warning: calibration skipped due to error: {e}")
    
    print(f"\n[System] Starting RL Training for {num_episodes} episodes...")
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

        # Phase 1.3: Compute and print complexity score
        llm_norm = min(1.0, token_len / 200.0)
        ppl_norm = min(1.0, prompt_ppl / 50.0)
        complexity_score = 0.6 * llm_norm + 0.4 * ppl_norm
        print(f"[Complexity] Score: {complexity_score:.3f} (llm_norm={llm_norm:.3f}, ppl_norm={ppl_norm:.3f})")

        # Phase 1.4: RL-based pruning and pruned metrics
        start_rl = time.time()
        state_tensor = rl_agent._get_state_vector(prompt, prompt_ppl=prompt_ppl, token_len=token_len)
        pruning_action = rl_agent.get_action(prompt, prompt_ppl=prompt_ppl, token_len=token_len, state_tensor=state_tensor)
        rl_inference_time_ms = (time.time() - start_rl) * 1000
        model_engine.apply_pruning(pruning_action)
        pruned_metrics = benchmark.benchmark_and_get_reward(
            model_engine, prompt, max_new_tokens=max_new_tokens, return_metrics=True
        )
        print(f"[Pruned] Action: {pruning_action.target} ({pruning_action.intensity}) | RL Time: {rl_inference_time_ms:.2f}ms | Model Time: {pruned_metrics['time_ms']:.2f}ms | Total Time: {rl_inference_time_ms + pruned_metrics['time_ms']:.2f}ms | Tok/s: {pruned_metrics['tok_s']:.2f} | PPL: {pruned_metrics['perplexity']:.2f} | GenTokens: {pruned_metrics.get('gen_tokens', 0)}")

        # Reward: balanced speed/quality with stability
        alpha, beta = 0.7, 0.3
        eps = 1e-8
        relative_reward = (
            alpha * (pruned_metrics['tok_s'] - base_metrics['tok_s']) / (base_metrics['tok_s'] + eps)
            - beta * (pruned_metrics['perplexity'] - base_metrics['perplexity']) / (base_metrics['perplexity'] + eps)
        )

        # Track effective layer-skip intensity (post-cap) for fair analysis
        effective_intensity = pruning_action.intensity
        if pruning_action.target == 'transformer_layers':
            layers = getattr(model_engine.model.model, 'layers', [])
            L = len(layers)
            if L > 0:
                k = max(1, int(round(L * pruning_action.intensity)))
                k = min(k, max(1, L // 8))
                k = min(k, L)
                effective_intensity = k / float(L)

        next_state_tensor = rl_agent._get_state_vector(prompt, prompt_ppl=prompt_ppl, token_len=token_len)
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
            'time_ms': pruned_metrics['time_ms'] + rl_inference_time_ms,
            'model_time_ms': pruned_metrics['time_ms'],
            'rl_inference_time_ms': rl_inference_time_ms,
            'tok_s': pruned_metrics['tok_s'],
            'ppl': pruned_metrics['perplexity'],
            'gen_tokens': pruned_metrics.get('gen_tokens', None),
            'action_index': pruning_action.action_index,
            'target': pruning_action.target,
            'intensity': pruning_action.intensity,
            'effective_intensity': effective_intensity,
            'reward': relative_reward
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
        'accuracy_compare.png',
        'accuracy_benchmark_baseline.png',
        'accuracy_benchmark_pruned.png',
        'pruning_action_usage.png',
        'inference_time_per_action.png',
        'perplexity_per_action.png',
        'token_speed_compare.png',
        'time_breakdown.png',
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

def test_agent(model_engine, rl_agent, benchmark, num_test_episodes=10, max_new_tokens: int = 50, test_dataset: str = None, force_action=None):
    """Test the trained RL agent on new prompts without training."""
    print(f"\n[Test] Evaluating trained agent on {num_test_episodes} test prompts...")
    
    # Set epsilon to 0 for pure exploitation
    original_epsilon = rl_agent.epsilon
    rl_agent.epsilon = 0.0
    metrics_list = []
    
    if test_dataset and test_dataset.endswith('.csv'):
        test_prompts = load_training_prompts(test_dataset, samples=max(num_test_episodes, 50))
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
        random.shuffle(test_prompts)
    
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
    
    for i in range(num_test_episodes):
        prompt = test_prompts[i]
        print(f"\n[Test Episode {i+1}/{num_test_episodes}]")
        print(f"[Test] Prompt: '{prompt}'")
        
        # Calculate prompt perplexity and token length for metrics
        token_len = len(rl_agent.tokenizer.encode(prompt))
        prompt_ppl = benchmark._calculate_perplexity(model_engine, prompt)
        
        # Baseline: no pruning
        model_engine.restore_model()
        base_metrics = benchmark.benchmark_and_get_reward(model_engine, prompt, max_new_tokens=max_new_tokens, return_metrics=True)
        print(f"[Test Baseline] Time: {base_metrics['time_ms']:.2f}ms | Tok/s: {base_metrics['tok_s']:.2f} | PPL: {base_metrics['perplexity']:.2f}")
        
        # Pruned: forced or RL-selected action
        if forced_action:
            action = forced_action
            rl_inference_time_ms = 0.0  # No RL inference for forced actions
        else:
            start_rl = time.time()
            state = rl_agent._get_state_vector(prompt)
            action = rl_agent.get_action(prompt, state_tensor=state)
            rl_inference_time_ms = (time.time() - start_rl) * 1000
        
        effective_intensity = action.intensity
        if action.target == 'transformer_layers':
            layers = getattr(model_engine.model.model, 'layers', [])
            L = len(layers)
            if L > 0:
                k = max(1, int(round(L * action.intensity)))
                k = min(k, max(1, L // 8))
                k = min(k, L)
                effective_intensity = k / float(L)
        
        model_engine.apply_pruning(action)
        metrics = benchmark.benchmark_and_get_reward(model_engine, prompt, max_new_tokens=max_new_tokens, return_metrics=True)
        print(f"[Test Pruned] Action: {action.target} ({action.intensity}) | RL Time: {rl_inference_time_ms:.2f}ms | Model Time: {metrics['time_ms']:.2f}ms | Total Time: {rl_inference_time_ms + metrics['time_ms']:.2f}ms | Tok/s: {metrics['tok_s']:.2f} | PPL: {metrics['perplexity']:.2f}")
        
        model_engine.restore_model()
        
        metrics_list.append({
            'episode': i+1,
            'token_len': token_len,
            'prompt_ppl': prompt_ppl,
            'complexity': 0.0,  # Not calculated for test
            'baseline_time_ms': base_metrics['time_ms'],
            'baseline_tok_s': base_metrics['tok_s'],
            'baseline_ppl': base_metrics['perplexity'],
            'time_ms': metrics['time_ms'] + rl_inference_time_ms,
            'model_time_ms': metrics['time_ms'],
            'rl_inference_time_ms': rl_inference_time_ms,
            'tok_s': metrics['tok_s'],
            'ppl': metrics['perplexity'],
            'action_index': action.action_index,
            'target': action.target,
            'intensity': action.intensity,
            'effective_intensity': effective_intensity
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

def _compute_ppl_on_ids(engine: RealModelEngine, prompt_ids: List[int], cont_ids: List[int]) -> float:
    if not cont_ids:
        return 50000.0
    ids = (prompt_ids or []) + (cont_ids or [])
    if not ids:
        return 50000.0
    input_ids = torch.tensor([ids], dtype=torch.long, device=engine.model.device)
    labels = input_ids.clone()
    pl = len(prompt_ids or [])
    if pl > 0:
        labels[:, :pl] = -100
    with torch.no_grad():
        out = engine.model(input_ids=input_ids, labels=labels)
    ppl = torch.exp(out.loss).item()
    if ppl is None or not torch.isfinite(torch.tensor(ppl)):
        return 50000.0
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

def run_wikitext2_comparative_eval(engine: RealModelEngine, benchmark: RealBenchmark, samples: int = 1000, split: str = 'test', seed: int = 42, max_new_tokens: int = 50, max_seq_len: int = 512, force_action_str: Optional[str] = None):
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
        if len(ids) < 8:
            return None, None
        cut = min(len(ids) - 1, max(8, max_seq_len // 2))
        prompt_ids = ids[:cut]
        cont_ids = ids[cut:min(len(ids), cut + max(1, max_seq_len - cut))]
        if not cont_ids:
            return None, None
        return prompt_ids, cont_ids

    print("[Eval] Baseline pass...")
    try:
        engine.restore_model()
    except Exception:
        pass
    base_times = []
    base_ppls = []
    base_tok_s = []
    for i, t in enumerate(texts, start=1):
        pc = make_prompt_cont(t)
        if not pc or pc[0] is None:
            continue
        prompt_ids, cont_ids = pc
        prompt_text = engine.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        start = time.time()
        gen = engine.generate_response(prompt_text, max_length=max_new_tokens)
        elapsed = max(1e-12, time.time() - start)
        tok_count = len(engine.tokenizer.encode(gen))
        base_times.append(elapsed * 1000.0)
        base_tok_s.append((tok_count / elapsed) if elapsed > 0 else 0.0)
        base_ppls.append(_compute_ppl_on_ids(engine, prompt_ids, cont_ids))
        if i % 50 == 0:
            print(f"[Eval] Baseline {i}/{len(texts)}")

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
    pr_tok_s = []
    for i, t in enumerate(texts, start=1):
        pc = make_prompt_cont(t)
        if not pc or pc[0] is None:
            continue
        prompt_ids, cont_ids = pc
        prompt_text = engine.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        start = time.time()
        gen = engine.generate_response(prompt_text, max_length=max_new_tokens)
        elapsed = max(1e-12, time.time() - start)
        tok_count = len(engine.tokenizer.encode(gen))
        pr_times.append(elapsed * 1000.0)
        pr_tok_s.append((tok_count / elapsed) if elapsed > 0 else 0.0)
        pr_ppls.append(_compute_ppl_on_ids(engine, prompt_ids, cont_ids))
        if i % 50 == 0:
            print(f"[Eval] Pruned {i}/{len(texts)}")

    n = min(len(base_times), len(pr_times), len(base_ppls), len(pr_ppls))
    if n <= 0:
        print("[Eval] WikiText-2: no aligned samples")
        return
    for i in range(n):
        metrics.append({
            'sample': i + 1,
            'baseline_time_ms': float(base_times[i]),
            'baseline_tok_s': float(base_tok_s[i]),
            'baseline_ppl': float(base_ppls[i]),
            'time_ms': float(pr_times[i]),
            'tok_s': float(pr_tok_s[i]),
            'ppl': float(pr_ppls[i]),
        })

    import json
    with open('wikitext2_metrics.json', 'w') as f:
        json.dump({
            'samples_used': n,
            'baseline': {
                'avg_time_ms': float(sum(base_times[:n]) / n),
                'avg_tok_s': float(sum(base_tok_s[:n]) / n),
                'avg_ppl': float(sum(base_ppls[:n]) / n),
            },
            'pruned': {
                'avg_time_ms': float(sum(pr_times[:n]) / n),
                'avg_tok_s': float(sum(pr_tok_s[:n]) / n),
                'avg_ppl': float(sum(pr_ppls[:n]) / n),
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
    parser.add_argument('--train-dataset', type=str, default='Prompt Dataset Train.csv')
    parser.add_argument('--test-dataset', type=str, default='Prompt Dataset Test.csv')
    parser.add_argument('--train-split', type=str, default='train')
    parser.add_argument('--train-samples', type=int, default=5000)
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'gpu', 'auto'], help='Device to load model on: cpu, gpu, or auto')
    parser.add_argument('--static-profiles', action='store_true', help='Enable prebuilt static pruning profiles (Phase A)')
    parser.add_argument('--sparsity-2to4', action='store_true', help='Enable 2:4 semi-structured sparsity packing on supported GPUs (Phase B)')
    parser.add_argument('--compile', dest='compile_profiles', action='store_true', help='Compile profiles with torch.compile if available (Phase B)')
    parser.add_argument('--kv-compress', action='store_true', help='Enable KV-cache compression scaffold (Phase D)')
    parser.add_argument('--kv-keep-ratio', type=float, default=1.0, help='KV tokens keep ratio [0,1] (Phase D)')
    parser.add_argument('--force-action', type=str, default=None, help='Force a specific action for test mode (format: target:intensity, e.g., ffn_neurons:0.2)')
    parser.add_argument('--lm-eval', action='store_true', help='Enable lm-eval harness benchmarks (BoolQ, MMLU, HellaSwag)')
    parser.add_argument('--eval-limit', type=int, default=None, help='Limit number of samples per lm-eval task (e.g., 100)')
    parser.add_argument('--eval-batch-size', type=int, default=1, help='Batch size for lm-eval')
    parser.add_argument('--eval-tasks', type=str, default='boolq,hellaswag,mmlu', help='Comma-separated list of lm-eval tasks')
    args = parser.parse_args()
    if args.mode == 'train':
        main(num_episodes=args.episodes, checkpoint_path=args.checkpoint, max_new_tokens=args.max_new_tokens,
             train_dataset=args.train_dataset, train_split=args.train_split, train_samples=args.train_samples, split_type='train', device=args.device,
             static_profiles=args.static_profiles, sparsity_2to4=args.sparsity_2to4, compile_profiles=args.compile_profiles,
             kv_compress=args.kv_compress, kv_keep_ratio=args.kv_keep_ratio)
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
            test_agent(engine, agent, bench, num_test_episodes=args.episodes, max_new_tokens=args.max_new_tokens, test_dataset=args.test_dataset, force_action=args.force_action)
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
                run_wikitext2_comparative_eval(engine, bench, samples=args.eval_samples, split='test', seed=args.eval_seed, max_new_tokens=args.max_new_tokens, max_seq_len=args.eval_max_seq_len, force_action_str=args.force_action)
            if getattr(args, 'boolq', False):
                run_zeroshoot_accuracy(engine, 'boolq', samples=args.eval_samples, seed=args.eval_seed, max_seq_len=args.eval_max_seq_len, force_action_str=args.force_action)
            if getattr(args, 'hellaswag', False):
                run_zeroshoot_accuracy(engine, 'hellaswag', samples=args.eval_samples, seed=args.eval_seed, max_seq_len=args.eval_max_seq_len, force_action_str=args.force_action)
            if getattr(args, 'mmlu', False):
                run_zeroshoot_accuracy(engine, 'mmlu', samples=args.eval_samples, seed=args.eval_seed, max_seq_len=args.eval_max_seq_len, force_action_str=args.force_action)
            organize_test_reports()
