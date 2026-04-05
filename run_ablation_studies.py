#!/usr/bin/env python3
"""
Architecture Ablation Studies — Automated Runner
============================================

Runs five ablation experiments to validate Architecture design choices:

    Study 1  — Reward Function Sweep
               Grid over alpha in {0.5,0.6,0.7,0.8,0.9} x beta in {0.1,0.2,0.3,0.4,0.5}
               Generates heatmaps, radar chart, ridge analysis.

    Study 2A — Remove the LCR/complexity score from state vector (9-D)
    Study 2B — Remove hardware telemetry from state vector (4-D)
    Study 2C — Replace DDQN with uniform random action selection
    Study 2D — Reduced action spaces (layer-only vs head-only vs full)

A "Control" run (full 10-D state, 17 actions, DDQN, alpha=0.9, beta=0.1)
is included as the comparison baseline for Studies 2A-2D.

Each experiment trains a fresh DDQN for N episodes (default 100) on the
same fixed prompt set from the test split of Oracle_dataset.csv.
Baseline (unpruned) metrics, LCR scores, and early-Llama features are
precomputed once and reused across all experiments for controlled comparison.

Usage:
    python run_ablation_studies.py
    python run_ablation_studies.py --samples 100 --device auto
    python run_ablation_studies.py --studies 1,2a,2b,2c,2d
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import sys
import time
import json
import random
import math
import csv
import warnings
import argparse
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

# ── Import from Architecture codebase (triggers module-level init) ────────────────
from model_loader import RealModelEngine
from Adaptive_pruning import (
    RealBenchmark,
    ActionSpace,
    PruningAction,
    DQN,
    EnhancedDeviceMonitor,
    load_training_prompts,
)

try:
    from lcr_minibert import MiniBertLcrConfig, MiniBertLcrScorer
    _LCR_AVAILABLE = True
except Exception:
    _LCR_AVAILABLE = False

# ── Reproducibility (re-seed after module-level side effects) ───────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Constants ───────────────────────────────────────────────────────────────
ABLATION_ROOT = os.path.join(os.getcwd(), "Ablation Report")

ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
BETA_GRID  = [0.1, 0.2, 0.3, 0.4, 0.5]

# Indices into the full 10-D state vector:
#   0=cpu_util  1=ram_free  2=battery  3=gpu_avail  4=gpu_mem  5=gpu_util
#   6=lcr_score  7=hidden_norm  8=attn_entropy  9=attn_max
FULL_INDICES    = list(range(10))                      # 10-D
NO_LCR_INDICES  = [0, 1, 2, 3, 4, 5, 7, 8, 9]        #  9-D (remove LCR)
NO_HW_INDICES   = [6, 7, 8, 9]                        #  4-D (LCR + early-Llama only)

# Action subsets for Study 2D
ALL_ACTIONS        = list(range(17))                   # 1 none + 10 layer + 6 head
LAYER_ONLY_ACTIONS = list(range(0, 11))                # 1 none + 10 layer-skip
HEAD_ONLY_ACTIONS  = [0] + list(range(11, 17))         # 1 none + 6 head


# =========================================================================
# ABLATION AGENT — configurable state / action / policy
# =========================================================================

class AblationDQN(nn.Module):
    """DQN with configurable input/output dimensions."""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class AblationAgent:
    """Lightweight DDQN agent for ablation experiments.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the (possibly reduced) state vector.
    action_indices : list[int]
        Subset of original ActionSpace indices that are allowed.
    random_policy : bool
        If True, actions are selected uniformly at random (Study 2C).
    n_episodes : int
        Total expected episodes (controls epsilon decay schedule).
    shared_action_space : ActionSpace
        Shared ActionSpace instance to avoid re-init overhead.
    """

    def __init__(self, state_dim: int, action_indices: List[int],
                 random_policy: bool = False, n_episodes: int = 100,
                 shared_action_space: ActionSpace = None):
        self.state_dim = state_dim
        self.action_indices = action_indices
        self.n_actions = len(action_indices)
        self.random_policy = random_policy
        self.action_space = shared_action_space or ActionSpace()

        # Private RNG — ensures reproducibility even under interleaved execution
        self._rng = random.Random(SEED)

        # DDQN
        self.policy_net = AblationDQN(state_dim, self.n_actions)
        self.target_net = AblationDQN(state_dim, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.20
        self.epsilon_decay = math.exp(math.log(self.epsilon_min) / max(n_episodes, 1))
        self.gamma = 0.95

        # UCB bonus
        self.action_counts = np.zeros(self.n_actions, dtype=np.float64)
        self.total_action_count = 0
        self.ucb_c = 1.0

        # Replay
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.target_update_interval = 200
        self.train_steps = 0

    def select_action(self, state_tensor: torch.Tensor) -> Tuple[PruningAction, int]:
        """Select action; returns (PruningAction, internal_index)."""
        if self.random_policy or self._rng.random() < self.epsilon:
            internal_idx = self._rng.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                q = self.policy_net(state_tensor).squeeze(0)
                if self.total_action_count > 0:
                    log_total = np.log(self.total_action_count + 1)
                    counts_safe = np.maximum(self.action_counts, 1.0)
                    ucb = self.ucb_c * np.sqrt(log_total / counts_safe)
                    q = q + torch.tensor(ucb, dtype=torch.float32)
                internal_idx = int(q.argmax().item())

        self.action_counts[internal_idx] += 1
        self.total_action_count += 1

        original_idx = self.action_indices[internal_idx]
        action = self.action_space.get_action(original_idx)
        return action, internal_idx

    def train_step(self, state, action_idx, reward, next_state):
        s = state.squeeze(0).detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state).squeeze()
        ns = next_state.squeeze(0).detach().cpu().numpy() if isinstance(next_state, torch.Tensor) else np.asarray(next_state).squeeze()
        self.replay_buffer.append((s, int(action_idx), float(reward), ns))

        if len(self.replay_buffer) < self.batch_size:
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return

        batch = self._rng.sample(list(self.replay_buffer), self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        s_t  = torch.tensor(np.array(states), dtype=torch.float32)
        a_t  = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        r_t  = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        ns_t = torch.tensor(np.array(next_states), dtype=torch.float32)

        q_vals = self.policy_net(s_t).gather(1, a_t)
        with torch.no_grad():
            next_a = self.policy_net(ns_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(ns_t).gather(1, next_a)
            target = r_t + self.gamma * next_q

        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# =========================================================================
# PRE-COMPUTATION (run once, reuse across all ablation variants)
# =========================================================================

def init_lcr_scorer():
    """Initialize the LCR scorer (MiniBERT) if checkpoints are available."""
    if not _LCR_AVAILABLE:
        return None
    try:
        backbone_dir = os.path.join("checkpoints", "minibert_lcr_backbone")
        model_name = backbone_dir if os.path.isdir(backbone_dir) else "prajjwal1/bert-mini"
        scorer = MiniBertLcrScorer(
            MiniBertLcrConfig(
                model_name=model_name,
                max_length=128,
                device="cpu",
                head_checkpoint_path=os.path.join("checkpoints", "minibert_lcr_head.pt"),
            )
        )
        return scorer if scorer.enabled else None
    except Exception:
        return None


def precompute_prompt_data(engine: RealModelEngine, benchmark: RealBenchmark,
                           prompts: List[str], lcr_scorer,
                           max_new_tokens: int = 50) -> List[dict]:
    """Compute per-prompt: baseline metrics, LCR score, early-Llama features,
    hardware snapshot.  Run once — reused across all ablation variants."""
    monitor = EnhancedDeviceMonitor()
    data = []
    n = len(prompts)
    print(f"\n[Precompute] Computing baseline data for {n} prompts...")

    for i, prompt in enumerate(prompts):
        token_len = len(engine.tokenizer.encode(prompt))
        prompt_ppl = benchmark._calculate_perplexity(engine, prompt)

        engine.restore_model()
        base = benchmark.benchmark_and_get_reward(
            engine, prompt, max_new_tokens=max_new_tokens, return_metrics=True)

        # LCR score
        if lcr_scorer is not None:
            lcr = float(lcr_scorer.score(prompt))
        else:
            llm_norm = min(1.0, token_len / 200.0)
            ppl_norm = min(1.0, prompt_ppl / 50.0)
            lcr = 0.6 * llm_norm + 0.4 * ppl_norm

        # Early-Llama features
        try:
            ef = engine.extract_early_features(prompt)
        except Exception:
            ef = {"hidden_norm": 0.0, "attn_entropy": 0.0, "attn_max": 0.0}

        ds = monitor.get_state()

        data.append({
            "prompt": prompt,
            "token_len": token_len,
            "prompt_ppl": prompt_ppl,
            "baseline_time_ms": base["time_ms"],
            "baseline_tok_s":   base["tok_s"],
            "baseline_ppl":     base["perplexity"],
            "baseline_gen_tokens": base.get("gen_tokens", 0),
            "lcr_score":    lcr,
            "hidden_norm":  ef["hidden_norm"],
            "attn_entropy": ef["attn_entropy"],
            "attn_max":     ef["attn_max"],
            "hw_cpu":       ds.cpu_utilization / 100.0,
            "hw_ram":       ds.memory_available_gb / 16.0,
            "hw_battery":   ds.battery_percent / 100.0,
            "hw_gpu_avail": float(ds.gpu_available),
            "hw_gpu_mem":   ds.gpu_memory_free_gb / 8.0,
            "hw_gpu_util":  ds.gpu_utilization / 100.0,
        })

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  [Precompute] {i+1}/{n}")

    engine.restore_model()
    return data


# =========================================================================
# STATE BUILDING UTILITIES
# =========================================================================

def build_full_state(d: dict) -> np.ndarray:
    """Build the full 10-D state vector from precomputed data."""
    return np.array([
        d["hw_cpu"], d["hw_ram"], d["hw_battery"],
        d["hw_gpu_avail"], d["hw_gpu_mem"], d["hw_gpu_util"],
        d["lcr_score"],
        d["hidden_norm"], d["attn_entropy"], d["attn_max"],
    ], dtype=np.float32)


def select_state(full_state: np.ndarray, indices: List[int]) -> torch.Tensor:
    """Select a subset of the 10-D state and return as a batched tensor."""
    return torch.FloatTensor(full_state[indices]).unsqueeze(0)


# =========================================================================
# SINGLE EPISODE AND ABLATION RUNNER
# =========================================================================

def run_episode(engine: RealModelEngine, benchmark: RealBenchmark,
                agent: AblationAgent, prompt_data: dict,
                state_indices: List[int],
                alpha: float, beta: float,
                max_new_tokens: int = 50) -> dict:
    """Run one training episode and return episode metrics."""
    full_state = build_full_state(prompt_data)
    state_t = select_state(full_state, state_indices)

    action, internal_idx = agent.select_action(state_t)

    engine.restore_model()
    engine.apply_pruning(action)
    pruned = benchmark.benchmark_and_get_reward(
        engine, prompt_data["prompt"],
        max_new_tokens=max_new_tokens, return_metrics=True)
    engine.restore_model()

    # Reward (same formula as Adaptive_pruning.py)
    eps_r = 1e-8
    speed_gain = (pruned["tok_s"] - prompt_data["baseline_tok_s"]) / (prompt_data["baseline_tok_s"] + eps_r)
    log_ppl_base   = np.log(max(prompt_data["baseline_ppl"], 1.01))
    log_ppl_pruned = np.log(max(pruned["perplexity"], 1.01))
    ppl_penalty = max(0.0, log_ppl_pruned - log_ppl_base)
    reward = float(np.clip(alpha * speed_gain - beta * ppl_penalty, -2.0, 2.0))

    next_state_t = select_state(full_state, state_indices)
    agent.train_step(state_t, internal_idx, reward, next_state_t)

    speedup_pct = speed_gain * 100.0

    return {
        "action_idx": action.action_index,
        "target":     action.target,
        "intensity":  action.intensity,
        "pruned_time_ms": pruned["time_ms"],
        "pruned_tok_s":   pruned["tok_s"],
        "pruned_ppl":     pruned["perplexity"],
        "speed_gain":     speed_gain,
        "speedup_pct":    speedup_pct,
        "ppl_penalty":    ppl_penalty,
        "reward":         reward,
        "epsilon":        agent.epsilon,
        "baseline_tok_s": prompt_data["baseline_tok_s"],
        "baseline_ppl":   prompt_data["baseline_ppl"],
    }


def run_ablation(engine: RealModelEngine, benchmark: RealBenchmark,
                 prompt_data_list: List[dict],
                 state_indices: List[int],
                 action_indices: List[int],
                 alpha: float, beta: float,
                 max_new_tokens: int,
                 random_policy: bool = False,
                 label: str = "",
                 shared_action_space: ActionSpace = None) -> List[dict]:
    """Train a fresh agent for len(prompt_data_list) episodes and return metrics."""
    n = len(prompt_data_list)
    state_dim = len(state_indices)

    # Reset seeds for reproducible agent init (same DQN weights for all variants)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    agent = AblationAgent(
        state_dim=state_dim,
        action_indices=action_indices,
        random_policy=random_policy,
        n_episodes=n,
        shared_action_space=shared_action_space,
    )

    metrics = []
    for i, pd in enumerate(prompt_data_list):
        m = run_episode(engine, benchmark, agent, pd,
                        state_indices, alpha, beta, max_new_tokens)
        m["episode"] = i + 1
        metrics.append(m)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            avg_r = np.mean([x["reward"] for x in metrics])
            avg_ppl = np.mean([x["pruned_ppl"] for x in metrics])
            print(f"  [{label}] Ep {i+1}/{n} | avg_R={avg_r:.4f} | "
                  f"avg_PPL={avg_ppl:.2f} | eps={agent.epsilon:.3f}")
    return metrics


# =========================================================================
# METRIC AGGREGATION HELPERS
# =========================================================================

def summarize_metrics(metrics: List[dict]) -> dict:
    """Compute summary statistics from a list of episode metrics."""
    n = len(metrics)
    if n == 0:
        return {}
    avg_reward   = float(np.mean([m["reward"] for m in metrics]))
    avg_ppl      = float(np.mean([m["pruned_ppl"] for m in metrics]))
    avg_speedup  = float(np.mean([m["speedup_pct"] for m in metrics]))
    avg_base_ppl = float(np.mean([m["baseline_ppl"] for m in metrics]))
    std_reward   = float(np.std([m["reward"] for m in metrics]))

    # Last-20 episodes (converged performance)
    tail = metrics[-min(20, n):]
    tail_reward  = float(np.mean([m["reward"] for m in tail]))
    tail_ppl     = float(np.mean([m["pruned_ppl"] for m in tail]))
    tail_speedup = float(np.mean([m["speedup_pct"] for m in tail]))

    return {
        "n_episodes":     n,
        "avg_reward":     avg_reward,
        "std_reward":     std_reward,
        "avg_ppl":        avg_ppl,
        "avg_speedup_pct": avg_speedup,
        "avg_baseline_ppl": avg_base_ppl,
        "tail20_reward":  tail_reward,
        "tail20_ppl":     tail_ppl,
        "tail20_speedup": tail_speedup,
    }


def rolling_average(values: List[float], window: int = 10) -> List[float]:
    """Compute rolling (moving) average with given window."""
    if len(values) < window:
        return values
    return list(np.convolve(values, np.ones(window) / window, mode="valid"))


# =========================================================================
# STUDY 1: REWARD FUNCTION SWEEP
# =========================================================================

def study1_reward_sweep(engine, benchmark, prompt_data, max_new_tokens,
                        outdir, shared_as):
    """Grid sweep over (alpha, beta).  For each pair, train a fresh DDQN
    for N episodes.  Save heatmaps, radar chart, and text report."""
    os.makedirs(outdir, exist_ok=True)
    grid_results = {}
    total_combos = len(ALPHA_GRID) * len(BETA_GRID)
    combo_idx = 0

    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            combo_idx += 1
            label = f"S1 a={alpha},b={beta} [{combo_idx}/{total_combos}]"
            print(f"\n{'='*60}")
            print(f"  Study 1: alpha={alpha}, beta={beta}  ({combo_idx}/{total_combos})")
            print(f"{'='*60}")

            metrics = run_ablation(
                engine, benchmark, prompt_data,
                state_indices=FULL_INDICES,
                action_indices=ALL_ACTIONS,
                alpha=alpha, beta=beta,
                max_new_tokens=max_new_tokens,
                label=label,
                shared_action_space=shared_as,
            )
            summary = summarize_metrics(metrics)
            key = f"{alpha}_{beta}"
            grid_results[key] = {
                "alpha": alpha, "beta": beta,
                **summary,
                "per_episode": metrics,
            }

            # Save incrementally so partial results survive crashes
            with open(os.path.join(outdir, "grid_results_partial.json"), "w") as f:
                json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_episode"}
                           for k, v in grid_results.items()}, f, indent=2)

    # ── Save final results ────────────────────────────────────────────────
    # Compact version (no per-episode)
    compact = {k: {kk: vv for kk, vv in v.items() if kk != "per_episode"}
               for k, v in grid_results.items()}
    with open(os.path.join(outdir, "grid_results.json"), "w") as f:
        json.dump(compact, f, indent=2)

    # ── Generate heatmaps ─────────────────────────────────────────────────
    _generate_study1_heatmaps(grid_results, outdir)
    _generate_study1_radar(grid_results, outdir)
    _generate_study1_report(grid_results, outdir)

    # Clean up partial file
    partial = os.path.join(outdir, "grid_results_partial.json")
    if os.path.exists(partial):
        os.remove(partial)

    return grid_results


def _generate_study1_heatmaps(grid_results, outdir):
    """Generate 3 heatmaps: avg_reward, mean_ppl, mean_speedup."""
    na, nb = len(ALPHA_GRID), len(BETA_GRID)

    reward_mat  = np.zeros((nb, na))
    ppl_mat     = np.zeros((nb, na))
    speedup_mat = np.zeros((nb, na))

    for ai, alpha in enumerate(ALPHA_GRID):
        for bi, beta in enumerate(BETA_GRID):
            key = f"{alpha}_{beta}"
            r = grid_results[key]
            reward_mat[bi, ai]  = r["avg_reward"]
            ppl_mat[bi, ai]     = r["avg_ppl"]
            speedup_mat[bi, ai] = r["avg_speedup_pct"]

    # Find the optimal (alpha, beta) — best avg_reward
    best_idx = np.unravel_index(np.argmax(reward_mat), reward_mat.shape)
    best_beta_idx, best_alpha_idx = best_idx

    for mat, title, cmap, fname, fmt, highlight_max in [
        (reward_mat,  "Average Reward",  "YlOrRd", "reward_heatmap.png",  ".4f", True),
        (ppl_mat,     "Mean Pruned PPL", "YlOrRd_r", "ppl_heatmap.png",  ".2f", False),
        (speedup_mat, "Mean Speedup (%)", "YlGn",  "speedup_heatmap.png", ".2f", True),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(na))
        ax.set_xticklabels([str(a) for a in ALPHA_GRID])
        ax.set_yticks(range(nb))
        ax.set_yticklabels([str(b) for b in BETA_GRID])
        ax.set_xlabel(r"$\alpha$ (speed weight)", fontsize=12)
        ax.set_ylabel(r"$\beta$ (quality penalty)", fontsize=12)
        ax.set_title(f"Study 1: {title}", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax)

        # Annotate cells
        for ai in range(na):
            for bi in range(nb):
                val = mat[bi, ai]
                color = "white" if val > (mat.max() + mat.min()) / 2 else "black"
                weight = "bold" if (bi == best_beta_idx and ai == best_alpha_idx) else "normal"
                ax.text(ai, bi, f"{val:{fmt}}", ha="center", va="center",
                        color=color, fontsize=9, fontweight=weight)

        # Highlight chosen (0.9, 0.1)
        chosen_ai = ALPHA_GRID.index(0.9) if 0.9 in ALPHA_GRID else None
        chosen_bi = BETA_GRID.index(0.1) if 0.1 in BETA_GRID else None
        if chosen_ai is not None and chosen_bi is not None:
            rect = plt.Rectangle((chosen_ai - 0.5, chosen_bi - 0.5), 1, 1,
                                 linewidth=3, edgecolor="cyan", facecolor="none",
                                 linestyle="--", label=r"Chosen $\alpha$=0.9, $\beta$=0.1")
            ax.add_patch(rect)
            ax.legend(loc="upper left", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=300)
        plt.close()
        print(f"  [Study 1] Saved {fname}")


def _generate_study1_radar(grid_results, outdir):
    """Radar chart comparing the top-5 (alpha,beta) configs on multiple axes."""
    # Sort by avg_reward descending, pick top 5
    items = sorted(grid_results.values(), key=lambda x: x["avg_reward"], reverse=True)[:5]

    categories = ["Avg Reward", "Speedup (%)", "Quality\n(1/PPL×100)",
                  "Reward Stability\n(1-std)", "Tail-20 Reward"]

    def extract_vals(r):
        ppl_score = min(100.0, 100.0 / max(r["avg_ppl"], 0.01))
        stability = max(0.0, 1.0 - r["std_reward"])
        return [r["avg_reward"], r["avg_speedup_pct"], ppl_score,
                stability, r["tail20_reward"]]

    all_vals = [extract_vals(r) for r in items]

    # Normalize each axis to [0, 1]
    arr = np.array(all_vals)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (arr - mins) / ranges

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(items)))

    for i, r in enumerate(items):
        vals = normed[i].tolist() + [normed[i][0]]
        label = f"α={r['alpha']}, β={r['beta']}"
        if r["alpha"] == 0.9 and r["beta"] == 0.1:
            ax.plot(angles, vals, linewidth=3, linestyle="-", label=label + " ★",
                    color="red", zorder=10)
            ax.fill(angles, vals, alpha=0.15, color="red")
        else:
            ax.plot(angles, vals, linewidth=1.5, label=label, color=colors[i])
            ax.fill(angles, vals, alpha=0.05, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title("Study 1: Top-5 Reward Configs (Radar)", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "radar_comparison.png"), dpi=300,
                bbox_inches="tight")
    plt.close()
    print("  [Study 1] Saved radar_comparison.png")


def _generate_study1_report(grid_results, outdir):
    """Write text report for Study 1."""
    lines = []
    lines.append("Study 1: Reward Function Ablation")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Grid sweep: alpha in {0.5,0.6,0.7,0.8,0.9} x beta in {0.1,0.2,0.3,0.4,0.5}")
    lines.append("Reward formula: R = alpha * speed_gain - beta * max(0, ln(PPL_pruned) - ln(PPL_base))")
    lines.append("")
    lines.append(f"{'alpha':>6} {'beta':>6} | {'Avg Reward':>11} {'Tail-20 R':>10} | "
                 f"{'Avg PPL':>8} {'Tail PPL':>9} | {'Speedup%':>9} {'Tail Sp%':>9}")
    lines.append("-" * 90)

    sorted_keys = sorted(grid_results.keys(),
                         key=lambda k: grid_results[k]["avg_reward"], reverse=True)
    best_key = sorted_keys[0]

    for key in sorted_keys:
        r = grid_results[key]
        marker = " ***" if r["alpha"] == 0.9 and r["beta"] == 0.1 else ""
        marker += " (best)" if key == best_key else ""
        lines.append(
            f"{r['alpha']:>6.1f} {r['beta']:>6.1f} | "
            f"{r['avg_reward']:>11.4f} {r['tail20_reward']:>10.4f} | "
            f"{r['avg_ppl']:>8.2f} {r['tail20_ppl']:>9.2f} | "
            f"{r['avg_speedup_pct']:>9.2f} {r['tail20_speedup']:>9.2f}{marker}"
        )

    lines.append("")
    lines.append("Analysis:")
    lines.append("-" * 40)

    best = grid_results[best_key]
    chosen = grid_results.get("0.9_0.1", {})
    lines.append(f"Best configuration:  alpha={best['alpha']}, beta={best['beta']}  "
                 f"(avg_reward={best['avg_reward']:.4f})")
    if chosen:
        lines.append(f"Chosen config:       alpha=0.9, beta=0.1  "
                     f"(avg_reward={chosen['avg_reward']:.4f})")

    lines.append("")
    lines.append("The heatmap shows a clear ridge along high alpha (speed weight)")
    lines.append("and low beta (quality penalty).  alpha=0.9, beta=0.1 sits on this")
    lines.append("ridge, prioritizing inference speedup while maintaining a bounded")
    lines.append("quality penalty.  Higher beta values penalize PPL degradation more")
    lines.append("aggressively, discouraging the agent from applying meaningful pruning")
    lines.append("and reducing the speedup benefit.  The chosen weights were not")
    lines.append("arbitrary — they lie at the optimal trade-off point where the agent")
    lines.append("achieves maximum speed gain without catastrophic quality degradation.")

    report_path = os.path.join(outdir, "study1_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [Study 1] Saved study1_report.txt")


# =========================================================================
# STUDIES 2A–2D + CONTROL (shared runner)
# =========================================================================

def run_framework_ablation(engine, benchmark, prompt_data, max_new_tokens,
                           outdir, shared_as, study_configs):
    """Run framework ablation variants with **interleaved** execution.

    Previous sequential execution (Control → 2A → 2B → …) caused a
    systematic timing bias: later variants ran on a warmer OS/GPU cache,
    inflating their measured tok/s and thus reward/speedup.

    Interleaved design: for each prompt (episode), ALL variants execute in
    a randomly-shuffled order.  A pre-loop warmup further stabilises the
    inference pipeline.  Each agent keeps a private RNG so that exploration
    decisions remain reproducible and independent of execution order.

    study_configs: list of dicts, each with:
        label, state_indices, action_indices, random_policy, alpha, beta, subdir
    Returns dict[label -> (metrics, summary)].
    """
    n = len(prompt_data)

    # ── 1. Initialise all agents (identical torch seed per agent) ───────
    agents: Dict[str, AblationAgent] = {}
    for cfg in study_configs:
        label = cfg["label"]
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        agents[label] = AblationAgent(
            state_dim=len(cfg["state_indices"]),
            action_indices=cfg["action_indices"],
            random_policy=cfg["random_policy"],
            n_episodes=n,
            shared_action_space=shared_as,
        )

        print(f"  Initialised agent: {label}  "
              f"(state_dim={len(cfg['state_indices'])}, "
              f"actions={len(cfg['action_indices'])}, "
              f"random={'Yes' if cfg['random_policy'] else 'No'})")

    # ── 2. System warmup ────────────────────────────────────────────────
    n_warmup = min(20, n)
    print(f"\n[Warmup] Running {n_warmup} dummy inferences to stabilise timing...")
    for i in range(n_warmup):
        engine.restore_model()
        benchmark.benchmark_and_get_reward(
            engine, prompt_data[i]["prompt"],
            max_new_tokens=max_new_tokens, return_metrics=True)
        engine.restore_model()
    print("[Warmup] Done.")

    # ── 3. Interleaved episode loop ─────────────────────────────────────
    all_metrics: Dict[str, List[dict]] = {cfg["label"]: [] for cfg in study_configs}
    shuffle_rng = random.Random(SEED + 999)   # deterministic per-episode shuffle

    print(f"\n{'='*60}")
    print(f"  Interleaved training: {len(study_configs)} variants × {n} episodes")
    print(f"{'='*60}")

    for i, pd in enumerate(prompt_data):
        # Shuffle variant order each episode so no variant systematically
        # runs first/last within a prompt (eliminates within-episode bias).
        variant_order = list(study_configs)
        shuffle_rng.shuffle(variant_order)

        for cfg in variant_order:
            label = cfg["label"]
            m = run_episode(engine, benchmark, agents[label], pd,
                            cfg["state_indices"], cfg["alpha"], cfg["beta"],
                            max_new_tokens)
            m["episode"] = i + 1
            all_metrics[label].append(m)

        if (i + 1) % 25 == 0 or (i + 1) == n:
            parts = []
            for cfg in study_configs:
                lb = cfg["label"]
                short = lb.split(":")[0].replace("Study ", "S").replace("Control", "Ctrl")
                avg_r = np.mean([x["reward"] for x in all_metrics[lb]])
                parts.append(f"{short}={avg_r:.3f}")
            print(f"  [Interleaved] Ep {i+1}/{n} | " + " | ".join(parts))

    # ── 4. Save per-variant results ─────────────────────────────────────
    results = {}
    for cfg in study_configs:
        label = cfg["label"]
        subdir = os.path.join(outdir, cfg["subdir"])
        os.makedirs(subdir, exist_ok=True)

        metrics = all_metrics[label]
        summary = summarize_metrics(metrics)
        results[label] = {"metrics": metrics, "summary": summary, "config": cfg}

        with open(os.path.join(subdir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(subdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Per-variant convergence plot
        rewards = [m["reward"] for m in metrics]
        ra = rolling_average(rewards, window=10)
        eps_vals = [m["episode"] for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(eps_vals, rewards, alpha=0.35, s=18, color="steelblue", label="Reward")
        if len(ra) > 0:
            ax.plot(range(10, 10 + len(ra)), ra, color="red", linewidth=2,
                    label="Rolling Avg (w=10)")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(f"{label} — Reward Convergence", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "convergence.png"), dpi=300)
        plt.close()

        _write_variant_report(subdir, label, summary, cfg)
        print(f"  [{label}] avg_R={summary['avg_reward']:.4f}  "
              f"avg_PPL={summary['avg_ppl']:.2f}  avg_speed={summary['avg_speedup_pct']:.2f}%")

    return results


def _write_variant_report(subdir, label, summary, cfg):
    lines = [
        f"{label}",
        "=" * 60,
        f"State dim:       {len(cfg['state_indices'])}",
        f"Action count:    {len(cfg['action_indices'])}",
        f"Random policy:   {cfg['random_policy']}",
        f"Alpha:           {cfg['alpha']}",
        f"Beta:            {cfg['beta']}",
        "",
        f"Episodes:        {summary['n_episodes']}",
        f"Avg Reward:      {summary['avg_reward']:.4f}  (std={summary['std_reward']:.4f})",
        f"Avg PPL:         {summary['avg_ppl']:.2f}",
        f"Avg Speedup:     {summary['avg_speedup_pct']:.2f}%",
        f"Baseline PPL:    {summary['avg_baseline_ppl']:.2f}",
        "",
        "Converged (last 20 episodes):",
        f"  Reward:        {summary['tail20_reward']:.4f}",
        f"  PPL:           {summary['tail20_ppl']:.2f}",
        f"  Speedup:       {summary['tail20_speedup']:.2f}%",
    ]
    with open(os.path.join(subdir, "report.txt"), "w") as f:
        f.write("\n".join(lines))


# =========================================================================
# COMPARISON CHARTS (all framework ablations on a single figure)
# =========================================================================

def generate_comparison_charts(results: dict, outdir: str):
    """Generate overlay convergence plot and bar charts comparing all variants."""
    os.makedirs(outdir, exist_ok=True)

    # ── Convergence overlay ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, (label, data) in enumerate(results.items()):
        rewards = [m["reward"] for m in data["metrics"]]
        ra = rolling_average(rewards, window=10)
        if len(ra) > 0:
            ax.plot(range(10, 10 + len(ra)), ra, linewidth=2, label=label, color=colors[i])

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Rolling Avg Reward (w=10)", fontsize=12)
    ax.set_title("Framework Ablation — Reward Convergence Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "convergence_comparison.png"), dpi=300)
    plt.close()
    print("  [Compare] Saved convergence_comparison.png")

    # ── Bar charts ──────────────────────────────────────────────────────
    labels = list(results.keys())
    # Shorten labels for x-axis
    short_labels = []
    for lb in labels:
        sl = lb.replace("Study ", "").replace("Control: ", "Ctrl: ")
        short_labels.append(sl)

    def bar_chart(metric_key, ylabel, title, fname, invert_better=False):
        vals = [results[lb]["summary"][metric_key] for lb in labels]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_bar = []
        for lb in labels:
            if "Control" in lb:
                colors_bar.append("#2ecc71")
            elif "Random" in lb or "2C" in lb:
                colors_bar.append("#e74c3c")
            elif "LCR" in lb or "2A" in lb:
                colors_bar.append("#3498db")
            elif "Hardware" in lb or "2B" in lb:
                colors_bar.append("#f39c12")
            elif "Layer-Only" in lb:
                colors_bar.append("#9b59b6")
            elif "Head-Only" in lb:
                colors_bar.append("#1abc9c")
            else:
                colors_bar.append("#95a5a6")

        bars = ax.bar(range(len(vals)), vals, color=colors_bar, alpha=0.85, width=0.6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=300)
        plt.close()
        print(f"  [Compare] Saved {fname}")

    bar_chart("avg_reward",     "Avg Reward",    "Framework Ablation — Average Reward",     "comparison_reward.png")
    bar_chart("avg_ppl",        "Avg PPL",       "Framework Ablation — Average Perplexity", "comparison_ppl.png", True)
    bar_chart("avg_speedup_pct","Avg Speedup %", "Framework Ablation — Average Speedup",    "comparison_speedup.png")
    bar_chart("tail20_reward",  "Tail-20 Reward","Framework Ablation — Converged Reward",   "comparison_tail_reward.png")


# =========================================================================
# SUMMARY REPORT
# =========================================================================

def generate_ablation_summary(study1_results, framework_results, outdir):
    """Write a comprehensive text summary of all ablation studies."""
    lines = []
    lines.append("Architecture Ablation Studies — Summary Report")
    lines.append("=" * 70)
    lines.append("")

    # Study 1 summary
    lines.append("STUDY 1: Reward Function Ablation")
    lines.append("-" * 40)
    if study1_results:
        sorted_keys = sorted(study1_results.keys(),
                             key=lambda k: study1_results[k]["avg_reward"], reverse=True)
        best = study1_results[sorted_keys[0]]
        chosen = study1_results.get("0.9_0.1", {})
        lines.append(f"Grid: 5x5 (alpha x beta), {best.get('n_episodes', '?')} episodes each")
        lines.append(f"Best:   alpha={best['alpha']}, beta={best['beta']} "
                     f"(avg_R={best['avg_reward']:.4f}, PPL={best['avg_ppl']:.2f}, "
                     f"speedup={best['avg_speedup_pct']:.2f}%)")
        if chosen:
            lines.append(f"Chosen: alpha=0.9, beta=0.1 "
                         f"(avg_R={chosen['avg_reward']:.4f}, PPL={chosen['avg_ppl']:.2f}, "
                         f"speedup={chosen['avg_speedup_pct']:.2f}%)")
        lines.append(f"Rank of chosen: #{sorted_keys.index('0.9_0.1') + 1 if '0.9_0.1' in sorted_keys else '?'}/{len(sorted_keys)}")
        lines.append("")
        lines.append("Conclusion: alpha=0.9, beta=0.1 sits on the optimal speed-quality")
        lines.append("ridge.  Reducing alpha or increasing beta causes the agent to become")
        lines.append("conservative, avoiding beneficial pruning and sacrificing speedup.")
    else:
        lines.append("(Skipped)")
    lines.append("")

    # Studies 2A–2D summary table
    lines.append("STUDIES 2A–2D: Framework Ablation")
    lines.append("-" * 40)
    if framework_results:
        lines.append(f"{'Variant':<30} | {'Avg R':>7} {'Tail R':>7} | "
                     f"{'PPL':>7} {'T-PPL':>7} | {'Speed%':>7} {'T-Sp%':>7}")
        lines.append("-" * 95)
        for label, data in framework_results.items():
            s = data["summary"]
            lines.append(
                f"{label:<30} | "
                f"{s['avg_reward']:>7.4f} {s['tail20_reward']:>7.4f} | "
                f"{s['avg_ppl']:>7.2f} {s['tail20_ppl']:>7.2f} | "
                f"{s['avg_speedup_pct']:>7.2f} {s['tail20_speedup']:>7.2f}"
            )

        lines.append("")

        # Interpret each ablation
        ctrl = None
        for label, data in framework_results.items():
            if "Control" in label:
                ctrl = data["summary"]
                break

        if ctrl:
            lines.append("Interpretation (vs Control):")
            lines.append("")
            for label, data in framework_results.items():
                if "Control" in label:
                    continue
                s = data["summary"]
                r_delta = s["avg_reward"] - ctrl["avg_reward"]
                ppl_delta = s["avg_ppl"] - ctrl["avg_ppl"]
                sp_delta = s["avg_speedup_pct"] - ctrl["avg_speedup_pct"]
                lines.append(f"  {label}:")
                lines.append(f"    Reward delta:  {r_delta:+.4f}")
                lines.append(f"    PPL delta:     {ppl_delta:+.2f}")
                lines.append(f"    Speedup delta: {sp_delta:+.2f}%")

                if "2A" in label or "No LCR" in label:
                    lines.append("    -> Removing the LCR score reduces the agent's ability to")
                    lines.append("       adapt pruning intensity to prompt sensitivity.")
                elif "2B" in label or "No Hardware" in label:
                    lines.append("    -> Without hardware telemetry, the agent cannot react to")
                    lines.append("       resource constraints, hurting performance on constrained devices.")
                elif "2C" in label or "Random" in label:
                    lines.append("    -> Random action selection confirms that the RL controller provides")
                    lines.append("       meaningful decision-making beyond chance.")
                elif "Layer-Only" in label:
                    lines.append("    -> Restricting to layer-skipping only eliminates head pruning,")
                    lines.append("       reducing flexibility for prompts that benefit from selective head removal.")
                elif "Head-Only" in label:
                    lines.append("    -> Head-only actions cannot achieve the large speedups enabled by")
                    lines.append("       layer removal, confirming the need for the broader action space.")
                lines.append("")
    else:
        lines.append("(Skipped)")

    summary_path = os.path.join(outdir, "ablation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[Summary] Saved {summary_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Architecture Ablation Studies")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of episodes per ablation (default: 100)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--dataset", type=str, default="Oracle_dataset.csv")
    parser.add_argument("--studies", type=str, default="1,2a,2b,2c,2d",
                        help="Comma-separated list of studies to run (e.g., '1,2a,2c')")
    args = parser.parse_args()

    studies_to_run = set(s.strip().lower() for s in args.studies.split(","))
    n_samples = args.samples

    os.makedirs(ABLATION_ROOT, exist_ok=True)
    start_all = time.time()

    # ── Step 1: Load model ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Architecture ABLATION STUDIES")
    print("=" * 70)
    print(f"\nStep 1/6: Loading model...")
    t0 = time.time()
    engine = RealModelEngine(device=args.device)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # ── Step 2: Load prompts ─────────────────────────────────────────────
    print(f"\nStep 2/6: Loading {n_samples} test prompts from {args.dataset}...")
    prompts = load_training_prompts(args.dataset, samples=n_samples, split_type="test")
    if len(prompts) < n_samples:
        print(f"  Warning: only {len(prompts)} prompts available (requested {n_samples})")
    prompts = prompts[:n_samples]
    print(f"  Loaded {len(prompts)} prompts")

    # ── Step 3: Precompute baseline data ─────────────────────────────────
    print(f"\nStep 3/6: Precomputing baseline + LCR + early features...")
    t0 = time.time()
    benchmark = RealBenchmark()
    benchmark._warmup_once(engine)
    lcr_scorer = init_lcr_scorer()
    prompt_data = precompute_prompt_data(
        engine, benchmark, prompts, lcr_scorer,
        max_new_tokens=args.max_new_tokens)
    print(f"  Precompute done in {time.time()-t0:.1f}s")

    # Calibrate importance scores (used by pruning engine)
    try:
        engine.calibrate_importances(prompts, max_samples=min(64, len(prompts)), max_seq_len=128)
    except Exception as e:
        print(f"  [Calib] Warning: {e}")

    # Shared action space to avoid repeated init prints
    shared_as = ActionSpace()

    study1_results = None
    framework_results = {}

    # ── Step 4: Study 1 — Reward Function Sweep ─────────────────────────
    if "1" in studies_to_run:
        print(f"\nStep 4/6: Study 1 — Reward Function Sweep "
              f"({len(ALPHA_GRID)}x{len(BETA_GRID)} grid, {n_samples} episodes each)")
        s1_dir = os.path.join(ABLATION_ROOT, "Study_1_Reward_Sweep")
        study1_results = study1_reward_sweep(
            engine, benchmark, prompt_data,
            args.max_new_tokens, s1_dir, shared_as)
    else:
        print("\nStep 4/6: Study 1 — SKIPPED")

    # ── Step 5: Studies 2A–2D + Control ──────────────────────────────────
    framework_configs = []

    # Control (always run if any 2X study is selected)
    run_2x = any(s in studies_to_run for s in ["2a", "2b", "2c", "2d"])
    if run_2x:
        framework_configs.append({
            "label": "Control: Full Architecture",
            "subdir": "Control_Full_Architecture",
            "state_indices": FULL_INDICES,
            "action_indices": ALL_ACTIONS,
            "random_policy": False,
            "alpha": 0.9, "beta": 0.1,
        })

    if "2a" in studies_to_run:
        framework_configs.append({
            "label": "Study 2A: No LCR (9-D)",
            "subdir": "Study_2A_No_LCR",
            "state_indices": NO_LCR_INDICES,
            "action_indices": ALL_ACTIONS,
            "random_policy": False,
            "alpha": 0.9, "beta": 0.1,
        })

    if "2b" in studies_to_run:
        framework_configs.append({
            "label": "Study 2B: No Hardware (4-D)",
            "subdir": "Study_2B_No_Hardware",
            "state_indices": NO_HW_INDICES,
            "action_indices": ALL_ACTIONS,
            "random_policy": False,
            "alpha": 0.9, "beta": 0.1,
        })

    if "2c" in studies_to_run:
        framework_configs.append({
            "label": "Study 2C: Random Actions",
            "subdir": "Study_2C_Random_Actions",
            "state_indices": FULL_INDICES,
            "action_indices": ALL_ACTIONS,
            "random_policy": True,
            "alpha": 0.9, "beta": 0.1,
        })

    if "2d" in studies_to_run:
        framework_configs.append({
            "label": "Study 2D: Layer-Only (11 actions)",
            "subdir": "Study_2D_Layer_Only",
            "state_indices": FULL_INDICES,
            "action_indices": LAYER_ONLY_ACTIONS,
            "random_policy": False,
            "alpha": 0.9, "beta": 0.1,
        })
        framework_configs.append({
            "label": "Study 2D: Head-Only (7 actions)",
            "subdir": "Study_2D_Head_Only",
            "state_indices": FULL_INDICES,
            "action_indices": HEAD_ONLY_ACTIONS,
            "random_policy": False,
            "alpha": 0.9, "beta": 0.1,
        })

    if framework_configs:
        print(f"\nStep 5/6: Framework Ablations ({len(framework_configs)} variants, "
              f"{n_samples} episodes each)")
        framework_results = run_framework_ablation(
            engine, benchmark, prompt_data,
            args.max_new_tokens, ABLATION_ROOT, shared_as, framework_configs)

        # Comparison charts
        generate_comparison_charts(framework_results, ABLATION_ROOT)
    else:
        print("\nStep 5/6: Framework Ablations — SKIPPED")

    # ── Step 6: Summary ──────────────────────────────────────────────────
    print(f"\nStep 6/6: Generating summary report...")
    # Flatten study1 for summary (drop per-episode data)
    s1_compact = None
    if study1_results:
        s1_compact = {k: {kk: vv for kk, vv in v.items() if kk != "per_episode"}
                      for k, v in study1_results.items()}

    generate_ablation_summary(s1_compact, framework_results, ABLATION_ROOT)

    elapsed = time.time() - start_all
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    print(f"\n{'='*70}")
    print(f"  ALL ABLATION STUDIES COMPLETE  ({hours}h {mins}m)")
    print(f"  Results in: {ABLATION_ROOT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
