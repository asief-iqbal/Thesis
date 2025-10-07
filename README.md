# CASRAP: Context Aware Structured Runtime Adaptive Pruning

A reinforcement learning (RL)-driven system for adaptive LLM pruning that balances inference speed, accuracy, and resource usage based on real-time hardware state and prompt complexity. Bridges static pruning (e.g., LLM-Pruner, SparseGPT) with dynamic runtime decisions using an interpretable yet learnable controller. Includes detailed per-episode logging (baseline vs pruned metrics) and comparative plots showing pruning effectiveness.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Table of Contents

- [Description](#description)
- [Recent Improvements](#recent-improvements)
- [Key Innovations](#key-innovations)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Modules](#modules)
- [CLI Arguments](#cli-arguments)
- [Benchmarks](#benchmarks)
- [Workflow](#workflow)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citations](#citations)
- [Acknowledgments](#acknowledgments)

## Description

CASRAP is an RL-driven adaptive LLM pruning system. Components: (1) RL Controller (Double DQN agent) selects pruning actions based on state (hardware telemetry + prompt complexity). State features: CPU/GPU utilization, memory/battery, token length, perplexity. Actions: prune attention heads, FFN neurons at intensities 0.1-0.3, or none. (2) RealModelEngine loads LLaMA-3.2-1B from Hugging Face (cached locally), applies importance-driven structural slicing for pruning. (3) Benchmarking evaluates speed (tokens/sec) vs accuracy (PPL) relative to baseline. (4) Training loop on CSV prompts, calibrates layer/head/FFN importances, trains DQN, generates reports/plots. (5) Pruners: StructuredHeadSlicer (importance-guided head pruning), StructuredFFNSlicer. Supports importance-aware structural pruning, static profiles, 2:4 sparsity, torch.compile, and KV compression scaffolds for real speedups.

This project implements an RL-driven adaptive pruning system for LLMs that balances inference speed, accuracy, and resource usage based on real-time hardware state and prompt complexity. It bridges static pruning (e.g., LLM-Pruner, SparseGPT) with dynamic runtime decisions using a learnable controller (Double DQN), now with advanced GPU acceleration features.

### Recent Improvements
- **GPU Detection Fix**: Disabled NVML initialization in `EnhancedDeviceMonitor` to prevent interference with `torch.cuda.is_available()`, enabling proper CUDA detection. Installed CUDA-enabled PyTorch (cu121) for GPU acceleration.
- **Importance-Driven Pruning**: Added calibration for heads, FFN channels, and layers using forward hooks to collect average absolute activations. Pruning now selects least-important units: heads/FFN via structural slicing (real speedups), layers via functional skipping (capped to ≤12.5% depth, excluding first/last).
- **Structural Pruning Enforcement**: Forced structural pruning for heads (GQA-safe) and FFN channels using `StructuredHeadSlicer` and `StructuredFFNSlicer`, reducing GEMM dimensions for true throughput gains. Falls back to magnitude-based scoring if calibration unavailable.
- **Action Space Optimization**: Trimmed to research-aligned actions: none, attention_heads (0.1-0.3), ffn_neurons (0.1-0.3). Removed transformer_layers actions to prevent PPL spikes.
- **Calibration Re-enabled**: Restored importance calibration before training to power importance-guided selections.
- **Phase A (Static Profiles)**: Prebuild and cache pruned model variants to avoid per-episode surgery, reducing overhead on CPU/GPU.
- **Phase B (GPU Acceleration)**: Added optional 2:4 semi-structured sparsity packing (cuSPARSELt) and torch.compile for 1.3–2.0× layer speedups on Ampere/Hopper GPUs.
- **Phase C (Calibration + Reconstruction)**: Lightweight post-prune reconstruction stabilizes perplexity using calibration prompts.
- **Phase D (KV Compression Scaffold)**: Framework for token selection in long contexts (placeholder for H2O/Keyformer integration).
- **Benchmarking Upgrade**: Separate encode/gen timing, warmup to avoid overhead, accurate throughput calculation.
- **Citations**: Based on Michel et al. (2019) for head pruning, Sanh et al. (2020) and Zhang et al. (2023) for FFN pruning, Frantar & Alistarh (2023) and Sun et al. (2023) for structural approaches, NVIDIA cuSPARSELt for 2:4 sparsity, PyTorch AO for compile.

These changes ensure CASRAP prunes least-important components, boosts token speed and reduces inference time drastically while preserving perplexity, aligning with state-of-the-art pruning research and adding production-ready acceleration.

## Key Innovations

- **RL Controller**: Learnable policy maps hardware + prompt complexity to pruning actions.
- **Importance-Aware Pruning**: Calibrates least-important heads/FFN/layers via activation hooks; structural slicing for real speedups.
- **Multi-Level Pruning**: Attention heads (GQA-safe), FFN channels, transformer layers at modest intensities (0.1-0.3).
- **Prompt-Centric Complexity**: Uses token length + model perplexity; no external NLP required.
- **Dataset Flexibility**: Automatic 80/20 splits for custom CSV datasets; standardized evaluation tooling.
- **Organized Reporting**: Training results automatically organized into numbered folders (Train 1, Train 2, etc.) under "Training Report".
- **Improved Reward Function**: Correctly rewards speed gains and perplexity reductions.

## Features

- **RL Controller (DQN)**: RL-driven pruning policy with epsilon-greedy action selection. Reward balances speed (tok/s) vs accuracy (PPL).
- **Prompt-Centric Complexity**: Token length + model perplexity on the prompt. No external NLP.
- **Pruning Methods**:
  - Importance-aware structural slicing (rebuilds Linear layers for speedups, GQA-safe head pruning).
  - Activation calibration for heads/FFN using forward hooks.
  - Static profiles: Prebuilt pruned model variants cached to avoid per-episode overhead.
- **GPU Acceleration (Phase B)**: Optional 2:4 semi-structured sparsity (cuSPARSELt) and torch.compile for 1.3–2.0× speedups on Ampere/Hopper.
- **Calibration + Reconstruction (Phase C)**: Post-prune reconstruction stabilizes perplexity.
- **KV Compression Scaffold (Phase D)**: Placeholder for long-context token selection.
- **Benchmarks**: WikiText-2 perplexity, lm-eval-harness tasks, latency/actual tokens/sec metrics with encode/gen breakdown.
- **Modes**: Separate train/test/report CLI modes with checkpointing.
- **Organized Reports**: Training outputs automatically organized into "Training Report/Train N" folders for each run.
- **Safety**: Reversible pruning, no permanent model damage.
- **Local Everything**: All caches, models, datasets stored in project folder.
- **Detailed Logging & Plots**: Per-episode baseline vs pruned metrics printed and saved; comparative scatter plots (token speed, inference time, perplexity) with trendlines; correlation plot (token length vs prompt PPL); pruning summary with average time and PPL subplots.
## Installation

### 1. Clone Repository
```bash
git clone https://github.com/asief-iqbal/Thesis.git
cd Thesis
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers psutil numpy accelerate datasets lm-eval matplotlib
```

### 4. Additional Downloads
No additional downloads required.

### 5. Setup Environment
Create `.env` file:
```
HUGGINGFACE_HUB_TOKEN=your_hf_token_here
STRUCTURAL_PRUNING=0  # 1 for structural pruning
```

**Note**: All caches (HF models/datasets) stored locally in project folder.

## Quick Start

### CPU Training with Static Profiles

```bash
python Adaptive_pruning.py --mode train --episodes 50 --train-samples 5000 --train-dataset "Prompt Dataset.csv" --device cpu --static-profiles
```

### GPU Training with All Features

```bash
python Adaptive_pruning.py --mode train --episodes 200 --train-samples 1200 --train-dataset "Prompt Dataset Train.csv" --device gpu --static-profiles --sparsity-2to4 --compile --kv-compress --kv-keep-ratio 0.5
```

### Testing Trained Agent

```bash
python Adaptive_pruning.py --mode test --checkpoint "checkpoints/rl_policy.pt" --device gpu --static-profiles
```

## Usage

### Training
Train the RL agent on a prompt dataset:
```bash
python Adaptive_pruning.py --mode train --episodes 100 --checkpoint checkpoints/rl_policy.pt --train-dataset "Prompt Dataset.csv" --train-samples 5000 --max-new-tokens 50
```
Uses 80% of the dataset for training.

**Training Outputs**:
- **Per-Episode Logs**: Terminal prints token length, prompt PPL, complexity score, baseline metrics (no-prune), pruned metrics (with RL action).
- **Saved Files**:
  - `training_metrics.json`: Detailed metrics for all episodes.
  - `token_speed_compare.png`: Baseline vs pruned token speed scatter + trendlines.
  - `inference_time_compare.png`: Baseline vs pruned inference time scatter + trendlines.
  - `perplexity_compare.png`: Baseline vs pruned perplexity scatter + trendlines.
  - `length_vs_ppl.png`: Correlation between token length and prompt perplexity.
- **Reports**: `training_report.txt` with averages by prune type.

**Faster Training Options** (for testing/debugging):
- `--episodes 50` (half episodes).
- `--train-samples 1000` (fewer prompts).
- `--max-new-tokens 20` (shorter generations).

Example fast run:
```bash
python Adaptive_pruning.py --mode train --episodes 10 --checkpoint checkpoints/rl_policy.pt --train-dataset "Prompt Dataset.csv" --train-samples 500 --max-new-tokens 20
```

### Testing
Evaluate the trained agent:
```bash
python Adaptive_pruning.py --mode test --checkpoint checkpoints/rl_policy.pt --max-new-tokens 50 --wikitext-samples 200 --lm-eval
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | - | `train`, `test`, or `report` |
| `--episodes` | 50 | Number of training episodes |
| `--checkpoint` | - | Path to save/load RL policy |
| `--train-dataset` | Prompt Dataset.csv | Dataset for training (supports CSV with 80/20 split for train/test) |
| `--train-samples` | 5000 | Number of training prompts |
| `--max-new-tokens` | 50 | Generation length |
| `--device` | auto | cpu/gpu/auto |
| `--static-profiles` | False | Enable prebuilt static pruning profiles (Phase A) |
| `--sparsity-2to4` | False | Enable 2:4 semi-structured sparsity packing on supported GPUs (Phase B) |
| `--compile` | False | Compile profiles with torch.compile if available (Phase B) |
| `--kv-compress` | False | Enable KV-cache compression scaffold (Phase D) |
| `--kv-keep-ratio` | 1.0 | KV tokens keep ratio [0,1] (Phase D) |
| `--wikitext-samples` | 200 | WikiText-2 eval samples |
| `--lm-eval` | False | Run lm-eval-harness |

## Architecture

### Core Components
- **RL Controller (DQN)**: State includes hardware (CPU/GPU, memory, battery) + prompt-centric complexity. Actions: pruning targets with varying intensities. Reward: 1.0 * (pruned_tok_s / base_tok_s) - 3.0 * max(0, (ppl_pruned - ppl_base)/ppl_base) (penalizes PPL increases strongly).
- **Prompt Analyzer**: Prompt-centric complexity (token length + model perplexity). No external NLP.
- **Model Engine**: Loads LLaMA-3.2-1B from HF, applies reversible pruning, generates responses, computes PPL.
- **Calibration System**: Pre-training activation statistics for head/FFN importance.
- **Benchmark System**: Measures latency (ms), tokens/sec (actual generated tokens), PPL. Logs baseline vs pruned per-episode.
- **Plot Generator**: Post-training comparative scatter plots with trendlines and outlier removal.

### Pruning Actions
- 0: `none` (0.0) - No pruning.
- 1: `attention_heads` (0.1) - Remove 10% least-important heads via structural slicing.
- 2: `attention_heads` (0.2) - Remove 20% least-important heads via structural slicing.
- 3: `attention_heads` (0.3) - Remove 30% least-important heads via structural slicing.
- 4: `ffn_neurons` (0.1) - Remove 10% least-important FFN channels via structural slicing.
- 5: `ffn_neurons` (0.2) - Remove 20% least-important FFN channels via structural slicing.
- 6: `ffn_neurons` (0.3) - Remove 30% least-important FFN channels via structural slicing.

### System Architecture Diagram

```mermaid
flowchart TD
    A[Load Model & Components] --> B[Calibration: Activation Stats for Heads/FFN]
    B --> C[User Prompt]
    C --> D[Compute Complexity: Tokens + PPL]
    D --> E[Collect Hardware State: CPU/GPU/Memory]
    E --> F[RL Controller: Selects Action from Varying Intensities]
    F --> G["Apply Pruning: Heads/FFN/Layers/KV (Calibration-Aware)"]
    G --> H[Generate Response]
    H --> I[Benchmark: Tok/s, PPL]
    I --> J[Compute Reward & Train Policy]
    J --> K[Restore Model]
    K --> C

    L[Dataset] --> M[Batch Evaluation]
    M --> N[Aggregate Results & Comparative Plots]
```

RL-driven with prompt-centric complexity; pruning includes activation-aware heads/FFN, sliding-window KV, capped layer skipping. Evaluation on datasets.

## Modules

- `model_loader.py`: HF loading, token auth, pruning apply/restore, save_pretrained, activation calibration.
- `nlp_analyzer.py`: Prompt analyzer (complexity via tokens + perplexity).
- `pruners/`: 
  - `head_pruner.py`, `ffn_pruner.py`, `layer_skipper.py` (functional masks).
  - `structured_ffn_slicer.py`, `structured_head_slicer.py` (structural slicing).
  - `kv_cache_pruner.py` (runtime KV cache size reduction).
- `.env`: Config (HF token, pruning mode).
- `checkpoints/`: RL policy files.
- `Training Report/`: Organized training results in numbered subfolders (Train 1, Train 2, etc.).
- `training_report.txt`: Post-training report.
- `training_metrics.json`: Detailed per-episode metrics.
- `token_speed_compare.png`: Baseline vs pruned token speed plot.
- `inference_time_compare.png`: Baseline vs pruned inference time plot.
- `perplexity_compare.png`: Baseline vs pruned perplexity plot.
- `length_vs_ppl.png`: Token length vs prompt perplexity correlation.
- `pruning_summary.png`: Average time and PPL per pruning action subplots.

## Benchmarks

- **WikiText-2**: Perplexity and inference speed.
- **lm-eval-harness**: Standardized accuracy on ARC-E, HellaSwag, Winogrande, LAMBADA.
- **Custom Metrics**: Latency (ms), tokens/sec, perplexity in train/test.

## Detailed Methodology: How the RL-Driven Pruning Architecture Works

This section provides a comprehensive, step-by-step explanation of the system's workflow, calculations, and decision-making processes.

### 1. Initialization Phase
- **Model Loading**: Load LLaMA-3.2-1B from Hugging Face using `transformers`. Requires HF token for access.
- **Pruner Setup**: Initialize all pruners:
  - Functional pruners (hooks for masking).
  - Structural slicers (rebuild layers, GQA-safe head pruning).
  - KV cache pruner (sliding window on past_key_values).
- **Calibration**: Collect activation-aware importance for heads/FFN using a small prompt set (64 samples, 128 seq len).
- **Prompt Analyzer**: Prompt-centric complexity (token length + model perplexity). No external NLP.
- **Hardware Monitor**: NVML (if GPU) or torch memory; psutil for CPU/memory/battery.

### 2. Prompt Processing (Workflow)

```mermaid
flowchart TD
    A[Input Prompt] --> B[Tokenizer]
    B --> C[LLM Tokens]
    A --> D[Model Perplexity]
    C --> E[llm_norm = tokens/200]
    D --> F[ppl_norm = ppl/50]
    E --> G[Complexity = 0.6*llm_norm + 0.4*ppl_norm]
    G --> H[Output Complexity]
```

**Step-by-step breakdown**:

1. **Tokenization**:
   - Use model tokenizer to count LLM tokens.

2. **Metrics**:
   - LLM tokens: Total tokens from model's tokenizer.
   - Prompt perplexity: PPL measured on the prompt by the current model.

3. **Normalization**:
   - llm_norm = min(1.0, llm_tokens / 200.0)
   - ppl_norm = min(1.0, ppl / 50.0)

4. **Complexity Score Calculation**:
   - Complexity Score = 0.6 * llm_norm + 0.4 * ppl_norm
   - Range: 0.0 (simple prompts) to 1.0 (complex prompts).

5. **Hardware State Collection**:
   - CPU utilization (%).
   - Memory available (GB).
   - GPU utilization (%) and free VRAM (GB).
   - Battery (%) if applicable.

### 3. Pruning Decision-Making (RL Policy) (Workflow)

```mermaid
flowchart TD
    A[State: Hardware + Complexity] --> B[Q-Network]
    B --> C[epsilon-greedy Action]
    C --> D[Apply Pruning]
    D --> E[Benchmark tok/s PPL]
    E --> F[Compute Reward]
    F --> G[Replay Buffer]
    G --> H[Optimize Double DQN]
    H --> I[Target Net Update]
```

Actions: `none`, `attention_heads`, `ffn_neurons` with preset varying intensities. Policy learns to pick actions based on state.

### 4. Pruning Application

- **Attention Heads**:
  - Importance: Activation stats (if calibrated) or L2 norms of Q/K/V/O projections per head.
  - Remove lowest-scoring heads (e.g., intensity 0.2 → remove 20% heads).
  - Functional: Mask with hooks; Structural: Rebuild layer without heads (GQA-safe).

- **FFN Neurons**:
  - Importance: Activation stats (if calibrated) or L2 norms of gate/up/down projections per channel.
  - Remove channels with lowest scores (mild intensities like 0.1).
  - Functional: Mask channels; Structural: Rebuild Linear layers.

- **Transformer Layers**:
  - Skip layers from the end (e.g., intensity 0.06 → skip 6% layers, max 2 of 16).
  - Functional: Hook to bypass layers (cache-safe identity).

- **KV Cache**:
  - Sliding-window pruning of past_key_values to keep only the most recent context (window size scales with intensity).
  - Reduces KV cache footprint during generation without forcing shorter outputs.

- **None**: No changes to model.

### 5. Response Generation
- Apply pruning to model.
- Generate response: `model.generate(prompt, max_new_tokens=effective_max)`.
- For KV pruning, past_key_values are trimmed during forward passes.

### 6. Benchmarking and Evaluation
- **Perplexity Calculation**: 
  - Full text = prompt + generated response.
  - PPL = exp(cross-entropy loss) on full text.
- **Performance Metrics**:
  - Inference time (ms): Time to generate response.
  - Tokens/sec: generated_tokens / inference_time (uses actual generated length).
- **Reward (for training)**: alpha * (pruned_tok_s / base_tok_s) + beta * (base_ppl / pruned_ppl), with alpha=0.6, beta=0.4. Balances relative speed gain vs accuracy improvement.

### 7. Model Restoration
- Restore all pruners to original state for next prompt.
- Ensures reversible pruning.

### 8. Training/Evaluation Loop (Phase 1 - During Training)
- **Calibration Phase**: Run activation collection on 64 prompts for head/FFN importances.
- For each prompt in dataset (80% train split):
  - **Step 1**: Compute token length and prompt perplexity (print and save).
  - **Step 2**: Measure baseline metrics (no pruning): inference time, tokens/sec, perplexity (print and save).
  - **Step 3**: Compute complexity score (print and save).
  - **Step 4**: RL selects action, apply pruning (using calibration if available), measure pruned metrics (print and save).
  - Process prompt → Decide pruning → Apply → Generate → Benchmark → Log metrics.
- Reward update and policy training on baseline vs pruned difference.

### 9. Post-Training Analysis (Phase 2 - After Training)
- Generate reports: Overall stats, per-prune-type averages.
- Produce comparative scatter plots (baseline red vs pruned blue, with trendlines, outliers removed):
  - Token speed per episode.
  - Inference time per episode.
  - Perplexity per episode.
- Produce correlation plot: Token length vs prompt perplexity.
- All plots saved as PNGs.

This methodology ensures learnable, hardware-aware pruning without randomness, adapting to prompt complexity for optimal speed-accuracy trade-offs.

## Roadmap

- **Phase 4**: Power/energy logging, baselines (SparseGPT, LLM-Pruner, quantization), Pareto plots.
- **Phase 5**: Expand RL state to more features, gradient-based saliency for pruning, ablations.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Activate venv and install deps.
   ```bash
   venv\Scripts\activate
   pip install torch transformers psutil numpy accelerate pynvml datasets lm-eval matplotlib
   ```

2. **CUDA OOM**: Enable structural pruning or use smaller model.
   ```bash
   STRUCTURAL_PRUNING=1
   ```

3. **HF Token**: Ensure `.env` has valid token.

4. **NVML Unavailable**: GPU memory metrics fall back to torch; GPU util set to 0.0.

5. **Slow Training**: Use fast options or GPU.

### Logs

Check console for errors. Training auto-saves checkpoints/reports.

## Contributing

- Fork repo.
- Create feature branch.
- Submit PR with description.
- Report bugs with logs.

## License

MIT License. See LICENSE file for details.

## Citations

If you use this work in your research, please cite:

```bibtex
@misc{rl_adaptive_pruning_llm,
  title={RL-Driven Adaptive Pruning for Large Language Models},
  author={Asief Iqbal},
  year={2025},
  howpublished={\url{https://github.com/asief-iqbal/Thesis}},
  note={An RL-driven system for adaptive LLM pruning balancing inference speed, accuracy, and resource usage with prompt-centric complexity and importance-guided structural pruning.}
}
```

### Additional Citations
- Michel, P., Levy, O., & Neubig, G. (2019). Are Sixteen Heads Really Better than One? NeurIPS.
- Sanh, V., Wolf, T., & Rush, A. M. (2020). Movement Pruning: Adaptive Sparsity by Fine-Tuning. NeurIPS.
- Zhang, T., Wang, S., Li, W., Qi, X., Zhang, Y., Wang, Z., & Tao, D. (2023). LLM-Pruner: On the Structural Pruning of Large Language Models. arXiv.
- Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. ICML.
- Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2023). Wanda: Pruning by Magnitude of Weights and Activations. ICML.
- NVIDIA. (2023). Exploiting Ampere Structured Sparsity with cuSPARSELt.
- PyTorch AO. (2024). Semi-Structured Sparse Tensors.

## Acknowledgments

- Hugging Face Transformers and Datasets for model and data handling.
- PyTorch for deep learning framework.
- lm-eval-harness for standardized evaluation.
