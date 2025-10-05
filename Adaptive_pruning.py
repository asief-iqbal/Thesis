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
import warnings
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
# ============================================================================
@dataclass
class DeviceState:
    cpu_utilization: float
    memory_available_gb: float
    battery_percent: float
    gpu_available: bool
    gpu_memory_free_gb: float
    gpu_utilization: float

class EnhancedDeviceMonitor:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.nvml_ok = bool(self.gpu_available and _NVML_AVAILABLE)
        if self.nvml_ok:
            try:
                self.handle = nvml.nvmlDeviceGetHandleByIndex(0)
                name = nvml.nvmlDeviceGetName(self.handle).decode("utf-8") if hasattr(nvml, "nvmlDeviceGetName") else "GPU"
                print(f"[Monitor] GPU detected ({name}); NVML telemetry enabled.")
            except Exception:
                self.nvml_ok = False
                print("[Monitor] GPU detected; NVML unavailable, using torch fallbacks.")
        else:
            print(f"[Monitor] GPU {'detected' if self.gpu_available else 'not detected' }.")

    def get_state(self) -> DeviceState:
        cpu_util = psutil.cpu_percent(interval=0.1)
        mem_free_gb = psutil.virtual_memory().available / (1024 ** 3)
        batt = psutil.sensors_battery().percent if psutil.sensors_battery() else 100.0
        if self.nvml_ok:
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.handle)
                gpu_mem_free_gb = mem_info.free / (1024 ** 3)
                gpu_util = float(util)
            except Exception:
                gpu_util = random.uniform(10, 80)
                gpu_mem_free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3) if self.gpu_available else 0.0
        else:
            gpu_util = random.uniform(10, 80)
            gpu_mem_free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3) if self.gpu_available else 0.0
        return DeviceState(
            cpu_utilization=cpu_util,
            memory_available_gb=mem_free_gb,
            battery_percent=batt,
            gpu_available=self.gpu_available,
            gpu_memory_free_gb=gpu_mem_free_gb,
            gpu_utilization=gpu_util
        )
# =========================================================================
@dataclass
class PruningAction:
    level: int; intensity: float; target: str; action_index: int
class ActionSpace:
    def __init__(self):
        # Expanded to include intensities 0.1,0.2,0.3,0.4,0.5 for all pruning categories
        self.actions = [
            PruningAction(level=0, intensity=0.0, target="none", action_index=0),
            # KV cache (runtime length reduction)
            PruningAction(level=1, intensity=0.1, target="kv_cache", action_index=1),
            PruningAction(level=1, intensity=0.2, target="kv_cache", action_index=2),
            PruningAction(level=1, intensity=0.3, target="kv_cache", action_index=3),
            PruningAction(level=1, intensity=0.4, target="kv_cache", action_index=4),
            PruningAction(level=1, intensity=0.5, target="kv_cache", action_index=5),
            # Attention heads (GQA-safe masking/slicing)
            PruningAction(level=2, intensity=0.1, target="attention_heads", action_index=6),
            PruningAction(level=2, intensity=0.2, target="attention_heads", action_index=7),
            PruningAction(level=2, intensity=0.3, target="attention_heads", action_index=8),
            PruningAction(level=2, intensity=0.4, target="attention_heads", action_index=9),
            PruningAction(level=2, intensity=0.5, target="attention_heads", action_index=10),
            # FFN channels (use calibration-aware scoring)
            PruningAction(level=2, intensity=0.1, target="ffn_neurons", action_index=11),
            PruningAction(level=2, intensity=0.2, target="ffn_neurons", action_index=12),
            PruningAction(level=2, intensity=0.3, target="ffn_neurons", action_index=13),
            PruningAction(level=2, intensity=0.4, target="ffn_neurons", action_index=14),
            PruningAction(level=2, intensity=0.5, target="ffn_neurons", action_index=15),
            # Transformer layers (skip very few; capped in engine)
            PruningAction(level=3, intensity=0.1, target="transformer_layers", action_index=16),
            PruningAction(level=3, intensity=0.2, target="transformer_layers", action_index=17),
            PruningAction(level=3, intensity=0.3, target="transformer_layers", action_index=18),
            PruningAction(level=3, intensity=0.4, target="transformer_layers", action_index=19),
            PruningAction(level=3, intensity=0.5, target="transformer_layers", action_index=20),
        ]
        print(f"[RL Agent] Action space initialized with {len(self.actions)} actions.")

    def get_action(self, index: int) -> PruningAction:
        return self.actions[index]

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)

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
        
        self.epsilon, self.epsilon_decay, self.epsilon_min, self.gamma = 0.9, 0.995, 0.05, 0.95

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
        self.policy_net.load_state_dict(ckpt['policy_state'])
        if 'target_state' in ckpt:
            self.target_net.load_state_dict(ckpt['target_state'])
        if 'optimizer_state' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception:
                pass
        self.epsilon = 0.0  # exploitation by default when loading
        print(f"[RL Agent] Checkpoint loaded from {path}")

    def _get_state_vector(self, prompt: str, prompt_ppl: Optional[float] = None, token_len: Optional[int] = None) -> torch.Tensor:
        device_state = self.device_monitor.get_state()
        
        # Simple prompt complexity: token length normalized + prompt perplexity if provided
        tokens = token_len if token_len is not None else len(self.tokenizer.encode(prompt))
        llm_norm = min(1.0, tokens / 200.0)
        ppl_norm = min(1.0, prompt_ppl / 50.0) if prompt_ppl is not None else 0.5
        complexity_score = 0.6 * llm_norm + 0.4 * ppl_norm
        
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

    def get_action(self, prompt: str, prompt_ppl: Optional[float] = None, token_len: Optional[int] = None) -> PruningAction:
        # RL-driven action selection: epsilon-greedy
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            state = self._get_state_vector(prompt, prompt_ppl, token_len)
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
        # Disabled for full exploration at ε=1.0
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"[RL Agent] Train Step: Loss={loss.item():.4f}, Epsilon={self.epsilon:.3f}")

# ============================================================================
# (UNCHANGED) REAL MODEL ENGINE & BENCHMARKING SYSTEM
# ============================================================================

class RealBenchmark:
    def _calculate_perplexity(self, engine: RealModelEngine, text: str) -> float:
        inputs = engine.tokenizer(text, return_tensors="pt").to(engine.model.device)
        with torch.no_grad():
            outputs = engine.model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()

    def benchmark_and_get_reward(self, engine: RealModelEngine, prompt: str, validation_text: str, max_new_tokens: int = 50, return_metrics: bool = False):
        start_time = time.time()
        generated_response = engine.generate_response(prompt, max_length=max_new_tokens)
        elapsed_s = (time.time() - start_time)
        inference_time_ms = elapsed_s * 1000
        # Use actual generated token count for tokens/sec
        gen_token_count = len(engine.tokenizer.encode(generated_response))
        tokens_per_sec = (gen_token_count / elapsed_s) if elapsed_s > 0 else 0.0
        # Compute perplexity on the generated response (prompt + response) for relevance
        full_text = prompt + " " + generated_response
        perplexity = self._calculate_perplexity(engine, full_text)
        # Reward balances speed and accuracy
        speed_bonus = tokens_per_sec
        accuracy_penalty = perplexity / 10.0
        reward = (0.6 * speed_bonus) - (0.4 * accuracy_penalty)
        print(f"[Benchmark] Time: {inference_time_ms:.2f}ms, GenTokens: {gen_token_count}, Tok/s: {tokens_per_sec:.2f}, PPL: {perplexity:.2f} -> Reward: {reward:.3f}")
        if return_metrics:
            return reward, { 'time_ms': inference_time_ms, 'tok_s': tokens_per_sec, 'perplexity': perplexity, 'gen_tokens': gen_token_count }
        return reward

def generate_report(metrics_list: List[Dict[str, Any]]):
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

def generate_report(metrics_list: List[Dict[str, Any]]):
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

    # Plots: Two separate PNGs for inference time and perplexity
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
    with open('training_report.txt', 'w') as f:
        f.write("Training Report\n")
        f.write(f"Episodes: {n}\n")
        f.write(f"Overall Avg Time: {total_time/n:.2f}ms\n")
        f.write(f"Overall Avg PPL: {total_ppl/n:.2f}\n\n")
        f.write("Averages by Prune Type and Intensity:\n")
        for (target, intensity), ms in groups.items():
            avg_time = sum(m['time_ms'] for m in ms) / len(ms)
            avg_ppl = sum(m['ppl'] for m in ms) / len(ms)
            f.write(f"  {target} {intensity}: Avg Time {avg_time:.2f}ms, Avg PPL {avg_ppl:.2f} ({len(ms)} samples)\n")
    print("[Report] Report saved to training_report.txt")

    # Save report to file
    with open('training_report.txt', 'w') as f:
        f.write("Training Report\n")
        f.write(f"Episodes: {n}\n")
        f.write(f"Overall Avg Time: {total_time/n:.2f}ms\n")
        f.write(f"Overall Avg PPL: {total_ppl/n:.2f}\n\n")
        f.write("Averages by Prune Type and Intensity:\n")
        for (target, intensity), ms in groups.items():
            avg_time = sum(m['time_ms'] for m in ms) / len(ms)
            avg_ppl = sum(m['ppl'] for m in ms) / len(ms)
            f.write(f"  {target} {intensity}: Avg Time {avg_time:.2f}ms, Avg PPL {avg_ppl:.2f} ({len(ms)} samples)\n")
    print("[Report] Report saved to training_report.txt")
def load_training_prompts(dataset_name: str, split: str = 'train', samples: int = 5000, split_type: str = 'train') -> List[str]:
    """Load a proper prompt dataset to train the RL controller.
    Supported: 'databricks/databricks-dolly-15k', 'tatsu-lab/alpaca', CSV files with 80/20 split, fallback to WikiText-2.
    split_type: 'train' or 'test' for CSV split.
    Returns a list of prompt strings.
    """
    try:
        from datasets import load_dataset
    except Exception:
        print("[Train] Install 'datasets' to use training datasets. Falling back to hardcoded prompts.")
        return [
            "What is the capital of France?",
            "Explain the core concept of machine learning in simple terms.",
            "Why is the sky blue during the day?",
            "Write a python function to calculate the factorial of a number.",
        ]
    name = dataset_name.strip()
    print(f"[Train] Loading dataset: {name} ({split}) ...")
    try:
        if name.endswith('.csv'):
            ds = load_dataset('csv', data_files=name)
            prompts = [r['Prompt'] for r in ds['train'] if r['Prompt'] and r['Prompt'].strip()]
            random.shuffle(prompts)
            # 80/20 split
            split_idx = int(0.8 * len(prompts))
            train_prompts = prompts[:split_idx]
            test_prompts = prompts[split_idx:]
            if split_type == 'test':
                return test_prompts[:samples]
            else:
                return train_prompts[:samples]
        else:
            ds = load_dataset(name, split=split)
    except Exception as e:
        print(f"[Train] Failed to load {name}: {e}. Falling back to WikiText-2.")
        try:
            ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            prompts = [r['text'] for r in ds if r['text'] and r['text'].strip()]
            random.shuffle(prompts)
            return prompts[:samples]
        except Exception:
            return ["Describe the water cycle.", "Who wrote 'Hamlet'?", "Define photosynthesis."]

    if not name.endswith('.csv'):
        prompts: List[str] = []
        # Map fields for known datasets
        if 'databricks' in name:
            # databricks/databricks-dolly-15k
            for r in ds:
                instr = (r.get('instruction') or '').strip()
                ctx = (r.get('context') or '').strip()
                if instr:
                    prompts.append(instr if not ctx else f"{instr}\n\nContext: {ctx}")
        elif 'alpaca' in name:
            for r in ds:
                instr = (r.get('instruction') or '').strip()
                ipt = (r.get('input') or '').strip()
                if instr:
                    prompts.append(instr if not ipt else f"{instr}\n\nInput: {ipt}")
        else:
            # Generic fallback: use any 'text' field
            for r in ds:
                t = (r.get('text') or '').strip()
                if t:
                    prompts.append(t)
    random.shuffle(prompts)
    if not prompts:
        prompts = ["Explain quantum computing in simple terms."]
    return prompts[:samples]
def main(num_episodes: int = 50,
         checkpoint_path: Optional[str] = None,
         max_new_tokens: int = 10000,
         train_dataset: str = 'Prompt Dataset.csv',
         train_split: str = 'train',
         train_samples: int = 5000,
         split_type: str = 'train'):
    model_engine = RealModelEngine()
    rl_agent = RLControllerAgent(model_engine.tokenizer)
    benchmark = RealBenchmark()
    
    validation_text = "The field of artificial intelligence has seen rapid advancements."
    # Load a proper training prompt pool from the specified dataset
    prompt_pool = load_training_prompts(train_dataset, split=train_split, samples=train_samples, split_type=split_type)
    num_episodes = len(prompt_pool)  # Train on every prompt in the pool
    
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
        _reward_base, base_metrics = benchmark.benchmark_and_get_reward(
            model_engine, prompt, validation_text, max_new_tokens=max_new_tokens, return_metrics=True
        )
        print(f"[Baseline] Time: {base_metrics['time_ms']:.2f}ms | Tok/s: {base_metrics['tok_s']:.2f} | PPL: {base_metrics['perplexity']:.2f} | GenTokens: {base_metrics.get('gen_tokens', 0)}")

        # Phase 1.3: Compute and print complexity score
        llm_norm = min(1.0, token_len / 200.0)
        ppl_norm = min(1.0, prompt_ppl / 50.0)
        complexity_score = 0.6 * llm_norm + 0.4 * ppl_norm
        print(f"[Complexity] Score: {complexity_score:.3f} (llm_norm={llm_norm:.3f}, ppl_norm={ppl_norm:.3f})")

        # Phase 1.4: RL-based pruning and pruned metrics
        state_tensor = rl_agent._get_state_vector(prompt, prompt_ppl=prompt_ppl, token_len=token_len)
        pruning_action = rl_agent.get_action(prompt, prompt_ppl=prompt_ppl, token_len=token_len)
        model_engine.apply_pruning(pruning_action)
        _reward_pruned, pruned_metrics = benchmark.benchmark_and_get_reward(
            model_engine, prompt, validation_text, max_new_tokens=max_new_tokens, return_metrics=True
        )
        print(f"[Pruned] Action: {pruning_action.target} ({pruning_action.intensity}) | Time: {pruned_metrics['time_ms']:.2f}ms | Tok/s: {pruned_metrics['tok_s']:.2f} | PPL: {pruned_metrics['perplexity']:.2f} | GenTokens: {pruned_metrics.get('gen_tokens', 0)}")

        # Compute relative reward: alpha * (pruned_tok_s / base_tok_s) - beta * (pruned_ppl / base_ppl)
        alpha, beta = 0.6, 0.4
        relative_reward = alpha * (pruned_metrics['tok_s'] / base_metrics['tok_s']) - beta * (pruned_metrics['perplexity'] / base_metrics['perplexity'])

        next_state_tensor = rl_agent._get_state_vector(prompt, prompt_ppl=prompt_ppl, token_len=token_len)
        rl_agent.train_step(state_tensor, pruning_action.action_index, relative_reward, next_state_tensor)
        
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
            'time_ms': pruned_metrics['time_ms'],
            'tok_s': pruned_metrics['tok_s'],
            'ppl': pruned_metrics['perplexity'],
            'gen_tokens': pruned_metrics.get('gen_tokens', None),
            'action_index': pruning_action.action_index,
            'target': pruning_action.target,
            'intensity': pruning_action.intensity,
            'reward': relative_reward
        })
        
        model_engine.restore_model()

    print("\n[System] ✓ RL training loop completed!")
    
    # Generate reports and comparative plots
    generate_report(metrics_list)
    
    # Phase 2: Create comparative scatter plots (baseline vs pruned) and correlation plot
    def generate_comparative_plots(metrics: List[Dict[str, Any]]):
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

    generate_comparative_plots(metrics_list)

    # Save metrics for later report generation and analysis
    import json
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics_list, f)
    
    if checkpoint_path:
        rl_agent.save(checkpoint_path)
    return rl_agent

def test_agent(model_engine, rl_agent, benchmark, num_test_episodes=10, max_new_tokens: int = 10000, dataset_name: str = None):
    """Test the trained RL agent on new prompts without training."""
    print(f"\n[Test] Evaluating trained agent on {num_test_episodes} test prompts...")
    
    # Set epsilon to 0 for pure exploitation
    original_epsilon = rl_agent.epsilon
    rl_agent.epsilon = 0.0
    
    if dataset_name and dataset_name.endswith('.csv'):
        test_prompts = load_training_prompts(dataset_name, split_type='test', samples=num_test_episodes)
    else:
        test_prompts = [
            "Explain quantum computing in simple terms.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Describe the water cycle.",
            "Why is exercise important for health?"
        ]
    
    total_reward = 0.0
    total_time = 0.0
    total_tok_s = 0.0
    total_ppl = 0.0
    validation_text = "The field of artificial intelligence has seen rapid advancements."
    
    for i in range(min(num_test_episodes, len(test_prompts))):
        prompt = test_prompts[i]
        print(f"\n[Test Episode {i+1}/{num_test_episodes}]")
        print(f"[Test] Prompt: '{prompt}'")
        
        # Get state and action (exploitation only)
        state = rl_agent._get_state_vector(prompt)
        action = rl_agent.get_action(prompt)
        
        # Apply action and benchmark
        model_engine.apply_pruning(action)
        reward, metrics = benchmark.benchmark_and_get_reward(model_engine, prompt, validation_text, max_new_tokens=max_new_tokens, return_metrics=True)
        total_reward += reward
        total_time += metrics['time_ms']
        total_tok_s += metrics['tok_s']
        total_ppl += metrics['perplexity']
        
        model_engine.restore_model()
    
    n = float(num_test_episodes)
    avg_reward = total_reward / n
    print(f"\n[Test] ✓ Testing completed! Average Reward: {avg_reward:.3f}")
    print(f"[Test] Avg Time: {total_time/n:.2f}ms | Avg Tok/s: {total_tok_s/n:.2f} | Avg PPL: {total_ppl/n:.2f}")
    
    # Restore original epsilon
    rl_agent.epsilon = original_epsilon

def run_wikitext_eval(model_engine, benchmark, split: str = 'test', samples: int = 200, max_new_tokens: int = 10000):
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
        ppl = benchmark._calculate_perplexity(model_engine, t[:1024])
        total_ms += ms
        total_tok_s += tok_s
        total_ppl += ppl
    n = float(len(texts)) if texts else 1.0
    print(f"[WikiText-2] Avg PPL: {total_ppl/n:.2f} | Avg Tok/s: {total_tok_s/n:.2f} | Avg Time: {total_ms/n:.2f}ms")

def run_lm_eval_harness(model_engine, tasks_list: List[str], batch_size: int = 1, export_dir: str = 'export/pruned_model'):
    """Run lm-eval-harness tasks (ARC, HellaSwag, Winogrande, LAMBADA) on the current (possibly pruned) model.
    Requires 'lm_eval' (pip install lm-eval). Saves model to export_dir then evaluates.
    """
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except Exception:
        print("[lm-eval] Please install 'lm-eval': pip install lm-eval")
        return
    # Save current model/tokenizer to a local dir for stable loading
    model_engine.save_pretrained(export_dir)
    model_args = {
        'pretrained': export_dir,
        'tokenizer': export_dir,
        'dtype': 'auto',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': batch_size,
        'trust_remote_code': True,
    }
    print(f"[lm-eval] Evaluating tasks: {tasks_list}")
    results = evaluator.simple_evaluate(
        model=HFLM(**model_args),
        tasks=tasks_list,
        batch_size=batch_size,
    )
    print("[lm-eval] Results summary:")
    try:
        import json
        print(json.dumps(results.get('results', results), indent=2))
    except Exception:
        print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Pruning RL Controller")
    parser.add_argument('--mode', choices=['train','test','report'], default='test')
    parser.add_argument('--checkpoint', type=str, default=os.path.join('checkpoints','rl_policy.pt'))
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-new-tokens', type=int, default=10000)
    parser.add_argument('--wikitext-samples', type=int, default=200)
    parser.add_argument('--train-dataset', type=str, default='Prompt Dataset.csv')
    parser.add_argument('--train-split', type=str, default='train')
    parser.add_argument('--train-samples', type=int, default=5000)
    parser.add_argument('--lm-eval', action='store_true', help='Run lm-eval-harness tasks after test')
    args = parser.parse_args()

    if args.mode == 'train':
        main(num_episodes=args.episodes, checkpoint_path=args.checkpoint, max_new_tokens=args.max_new_tokens,
             train_dataset=args.train_dataset, train_split=args.train_split, train_samples=args.train_samples, split_type='train')
    elif args.mode == 'report':
        import json
        with open('training_metrics.json', 'r') as f:
            metrics_list = json.load(f)
        generate_report(metrics_list)
    else:
        engine = RealModelEngine()
        agent = RLControllerAgent(engine.tokenizer)
        if os.path.exists(args.checkpoint):
            agent.load(args.checkpoint)
        bench = RealBenchmark()
        test_agent(engine, agent, bench, num_test_episodes=10, max_new_tokens=args.max_new_tokens, dataset_name=args.train_dataset)
        run_wikitext_eval(engine, bench, split='test', samples=args.wikitext_samples, max_new_tokens=args.max_new_tokens)
        if args.lm_eval:
            run_lm_eval_harness(engine, tasks_list=['arc_easy','hellaswag','winogrande','lambada'], batch_size=1)
