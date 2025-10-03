# Runtime-Adaptive Pruning via RL

An RL-driven system for adaptive LLM pruning that balances inference speed, accuracy, and resource usage based on real-time hardware state and prompt complexity. Bridges static pruning (e.g., LLM-Pruner, SparseGPT) with dynamic runtime decisions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Description](#description)
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
- [License](#license)
- [Citations](#citations)
- [Acknowledgments](#acknowledgments)

## Description

This project implements a reinforcement learning (RL) controller using a Deep Q-Network (DQN) to dynamically select structured pruning actions (attention heads, FFN channels, layers) for LLaMA models. The RL agent observes hardware telemetry (CPU/GPU utilization, memory, power via NVML) and prompt complexity (analyzed via spaCy/NLTK POS tagging) to optimize trade-offs between latency, perplexity, and energy.

The system is designed for A*-level research, comparing to SparseGPT, LLM-Pruner, PAT, RAP, with real pruning effects, standardized evaluation (lm-eval-harness), and rigorous RL training (replay buffer, target net).

## Key Innovations

- **Adaptive Pruning**: RL learns to prune more aggressively for complex prompts on constrained hardware.
- **Structural Speedups**: Magnitude-based slicing rebuilds model layers for real inference speedups (Phase 3).
- **Standardized Evaluation**: Integrates lm-eval-harness for ARC, HellaSwag, Winogrande, LAMBADA.
- **Dataset-Driven Training**: Trains on real prompt datasets (Dolly-15k, Alpaca).
- **Post-Training Reports**: Automatic generation of latency/PPL reports and graphs.

## Features

- **RL Controller**: Double DQN with replay buffer, target network, epsilon-greedy exploration.
- **NLP Analysis**: Hybrid spaCy/NLTK for rich prompt features (POS, noun chunks, dependency span).
- **Pruning Methods**:
  - Functional masking (hooks for validation).
  - Structural slicing (rebuilds Linear layers for speedups).
- **Benchmarks**: WikiText-2 perplexity, lm-eval-harness tasks, latency/tokens/sec metrics.
- **Modes**: Separate train/test CLI modes with checkpointing.
- **Safety**: Reversible pruning, no permanent model damage.
- **Local Everything**: All caches, models, datasets stored in project folder.

## Requirements

- **OS**: Windows/Linux/Mac (tested on Windows)
- **Python**: 3.9+
- **Hardware**: GPU recommended (CUDA), CPU fallback
- **Memory**: 8GB+ RAM, 16GB+ VRAM for LLaMA-3.2-1B
- **Disk**: 20GB+ for models/datasets

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
pip install torch transformers psutil numpy accelerate pynvml spacy datasets lm-eval nltk matplotlib
```

### 4. Download Data Locally
```bash
# NLTK data
python -c "import nltk; nltk.data.path.append('nltk_data'); nltk.download('punkt', download_dir='nltk_data')"

# spaCy model (optional, falls back to NLTK)
python -c "import spacy.cli; spacy.cli.download('en_core_web_sm', data_path='spacy_data')"
```

### 5. Setup Environment
Create `.env` file:
```
HUGGINGFACE_HUB_TOKEN=your_hf_token_here
STRUCTURAL_PRUNING=0  # 1 for structural pruning
```

**Note**: All caches (HF models/datasets, NLTK, spaCy) stored locally in project folder.

## Quick Start

### Train RL Agent (Fast Mode)
```bash
venv\Scripts\activate
python Adaptive_pruning.py --mode train --episodes 10 --checkpoint checkpoints/rl_policy.pt --train-dataset databricks/databricks-dolly-15k --train-samples 500 --max-new-tokens 20
```

### Test Trained Agent
```bash
venv\Scripts\activate
python Adaptive_pruning.py --mode test --checkpoint checkpoints/rl_policy.pt --max-new-tokens 50 --wikitext-samples 200 --lm-eval
```

## Usage

### Training
Train the RL agent on a prompt dataset:
```bash
python Adaptive_pruning.py --mode train --episodes 100 --checkpoint checkpoints/rl_policy.pt --train-dataset databricks/databricks-dolly-15k --train-samples 5000 --max-new-tokens 50
```

**Faster Training Options** (for testing/debugging):
- `--episodes 50` (half episodes).
- `--train-samples 1000` (fewer prompts).
- `--max-new-tokens 20` (shorter generations).

Example fast run:
```bash
python Adaptive_pruning.py --mode train --episodes 10 --checkpoint checkpoints/rl_policy.pt --train-dataset databricks/databricks-dolly-15k --train-samples 500 --max-new-tokens 20
```

### Testing
Evaluate the trained agent:
```bash
python Adaptive_pruning.py --mode test --checkpoint checkpoints/rl_policy.pt --max-new-tokens 50 --wikitext-samples 200 --lm-eval
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | - | `train` or `test` |
| `--episodes` | 50 | Number of training episodes |
| `--checkpoint` | - | Path to save/load RL policy |
| `--train-dataset` | databricks/databricks-dolly-15k | Dataset for training |
| `--train-samples` | 5000 | Number of training prompts |
| `--max-new-tokens` | 50 | Generation length |
| `--wikitext-samples` | 200 | WikiText-2 eval samples |
| `--lm-eval` | False | Run lm-eval-harness |

## Architecture

### Core Components
- **RL Controller (DQN)**: State includes hardware (CPU/GPU, memory, battery) + prompt complexity score. Actions: pruning targets/intensities. Reward: 0.6 * tokens/sec - 0.4 * (perplexity/10).
- **NLP Analyzer**: Hybrid spaCy/NLTK scoring (verbs, questions, tokens, noun chunks, dep span).
- **Model Engine**: Loads LLaMA-3.2-1B from HF, applies reversible pruning, generates responses, computes PPL.
- **Benchmark System**: Measures latency (ms), tokens/sec, PPL, VRAM, power.

### Pruning Actions
- 0: `none` (0.0) - No pruning.
- 1: `kv_cache` (0.3) - KV cache pruning.
- 2: `attention_heads` (0.4) - Attention head pruning.
- 3: `ffn_neurons` (0.5) - FFN channel pruning.
- 4: `transformer_layers` (0.3) - Layer pruning.

## Modules

- `Adaptive_pruning.py`: Main script (RL agent, loops, eval runners).
- `model_loader.py`: HF loading, token auth, pruning apply/restore, save_pretrained.
- `nlp_analyzer.py`: Prompt complexity analysis.
- `pruners/`: 
  - `head_pruner.py`, `ffn_pruner.py`, `layer_skipper.py` (functional masks).
  - `structured_ffn_slicer.py`, `structured_head_slicer.py` (structural slicing).
- `.env`: Config (HF token, pruning mode).
- `checkpoints/`: RL policy files.
- `training_report.txt`: Post-training report.
- `training_metrics.png`: Graphs.

## Benchmarks

- **WikiText-2**: Perplexity and inference speed.
- **lm-eval-harness**: Standardized accuracy on ARC-E, HellaSwag, Winogrande, LAMBADA.
- **Custom Metrics**: Latency (ms), tokens/sec, perplexity in train/test.

## Workflow

1. **Initialization**: Load HF token, model (LLaMA-3.2-1B), NLP analyzer, pruners.
2. **Training Loop**:
   - Sample prompt from dataset (Dolly/Alpaca).
   - RL observes state (hardware + NLP complexity).
   - Selects action (prune heads/FFN/layers at intensity).
   - Apply pruning to model.
   - Generate response, compute reward (speed vs. accuracy).
   - RL train_step (update policy/target nets).
   - Restore model for next episode.
3. **Testing Loop**:
   - Load RL checkpoint (epsilon=0 for exploitation).
   - Run on test prompts, apply actions, benchmark, average metrics.
4. **Evaluation**:
   - WikiText-2: Average PPL, tokens/sec, time over samples.
   - lm-eval-harness: Export pruned model, evaluate on tasks.
5. **Report Generation**: Automatic report and plots.

## Roadmap

- **Phase 4**: Power/energy logging, baselines (SparseGPT, LLM-Pruner, quantization), Pareto plots.
- **Phase 5**: Expand RL state to full NLP feature vector, gradient-based saliency for pruning, ablations.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Activate venv and install deps.
   ```bash
   venv\Scripts\activate
   pip install torch transformers psutil numpy accelerate pynvml spacy datasets lm-eval nltk matplotlib
   ```

2. **CUDA OOM**: Enable structural pruning or use smaller model.
   ```bash
   STRUCTURAL_PRUNING=1
   ```

3. **HF Token**: Ensure `.env` has valid token.

4. **spaCy Fail**: NLTK fallback works.

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
@misc{runtime_adaptive_pruning_rl,
  title={Runtime-Adaptive Pruning for LLMs via Reinforcement Learning},
  author={Asief Iqbal},
  year={2025},
  howpublished={\url{https://github.com/asief-iqbal/Thesis}},
  note={An RL-driven system for adaptive LLM pruning balancing inference speed, accuracy, and resource usage.}
}
```

## Acknowledgments

- Hugging Face Transformers and Datasets for model and data handling.
- spaCy and NLTK for NLP analysis.
- PyTorch for deep learning framework.
- lm-eval-harness for standardized evaluation.