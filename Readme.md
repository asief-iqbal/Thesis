# Runtime-Adaptive Pruning via RL

An RL-driven system for adaptive LLM pruning that balances inference speed, accuracy, and resource usage based on real-time hardware state and prompt complexity. Bridges static pruning (e.g., LLM-Pruner, SparseGPT) with dynamic runtime decisions.

## Description

This project implements a reinforcement learning (RL) controller using a Deep Q-Network (DQN) to dynamically select structured pruning actions (attention heads, FFN channels, layers) for LLaMA models. The RL agent observes hardware telemetry (CPU/GPU utilization, memory, power via NVML) and prompt complexity (analyzed via spaCy/NLTK POS tagging) to optimize trade-offs between latency, perplexity, and energy.

Key innovations:
- **Adaptive Pruning**: RL learns to prune more aggressively for complex prompts on constrained hardware.
- **Structural Speedups**: Magnitude-based slicing rebuilds model layers for real inference speedups (Phase 3).
- **Standardized Evaluation**: Integrates lm-eval-harness for ARC, HellaSwag, Winogrande, LAMBADA.
- **Dataset-Driven Training**: Trains on real prompt datasets (Dolly-15k, Alpaca).

## Features

- **RL Controller**: Double DQN with replay buffer, target network, epsilon-greedy exploration.
- **NLP Analysis**: Hybrid spaCy/NLTK for rich prompt features (POS, noun chunks, dependency span).
- **Pruning Methods**:
  - Functional masking (hooks for validation).
  - Structural slicing (rebuilds Linear layers for speedups).
- **Benchmarks**: WikiText-2 perplexity, lm-eval-harness tasks, latency/tokens/sec metrics.
- **Modes**: Separate train/test CLI modes with checkpointing.
- **Safety**: Reversible pruning, no permanent model damage.

## Installation

1. **Clone the repository** (if not already):
   ```bash
   git clone <repo-url>
   cd runtime-adaptive-pruning
   ```

2. **Create virtual environment** (all packages installed locally):
   ```bash
   python -m venv venv
   ```

3. **Activate venv and install dependencies**:
   ```bash
   venv\Scripts\activate  # On Windows
   pip install torch transformers psutil numpy accelerate pynvml spacy datasets lm-eval nltk
   ```

4. **Download data locally** (to project folder):
   ```bash
   # NLTK data
   python -c "import nltk; nltk.data.path.append('nltk_data'); nltk.download('punkt', download_dir='nltk_data')"
   
   # spaCy model
   python -c "import spacy.cli; spacy.cli.download('en_core_web_sm', data_path='spacy_data')"
   ```

5. **Set up environment**:
   - Create `.env` with Hugging Face token: `HUGGINGFACE_HUB_TOKEN=your_token_here`
   - Optional: `STRUCTURAL_PRUNING=1` to enable structural pruning.

**Note**: All caches (HF models/datasets, NLTK, spaCy) are stored in the project folder (`hf_cache/`, `nltk_data/`, `spacy_data/`). If you have global installations on C drive, uninstall them manually or ignore— the code uses local paths.

## Usage

### Training
Train the RL agent on a prompt dataset:
```bash
venv\Scripts\activate
python Adaptive_pruning.py --mode train --episodes 100 --checkpoint checkpoints/rl_policy.pt --train-dataset databricks/databricks-dolly-15k --train-samples 5000 --max-new-tokens 50
```

**Faster Training Options** (reduce time for testing/debugging):
- `--episodes 50` (half the episodes).
- `--train-samples 1000` (fewer prompts).
- `--max-new-tokens 20` (shorter generations).
- Example fast run: `python Adaptive_pruning.py --mode train --episodes 10 --checkpoint checkpoints/rl_policy.pt --train-dataset databricks/databricks-dolly-15k --train-samples 500 --max-new-tokens 20`

### Testing
Evaluate the trained agent:
```bash
venv\Scripts\activate
python Adaptive_pruning.py --mode test --checkpoint checkpoints/rl_policy.pt --max-new-tokens 50 --wikitext-samples 200 --lm-eval
```

### Options
- `--mode`: `train` or `test`.
- `--episodes`: Number of training episodes.
- `--checkpoint`: Path to save/load RL policy.
- `--train-dataset`: Dataset for training (e.g., `databricks/databricks-dolly-15k`, `tatsu-lab/alpaca`).
- `--train-samples`: Number of training prompts.
- `--max-new-tokens`: Generation length.
- `--wikitext-samples`: Samples for WikiText-2 eval.
- `--lm-eval`: Run lm-eval-harness after test.

## Architecture

### Core Components
- **RL Controller (DQN)**: State includes hardware (CPU/GPU, memory, battery) + prompt complexity score. Actions: pruning targets/intensities. Reward: 0.6 * tokens/sec - 0.4 * (perplexity/10).
- **NLP Analyzer**: Hybrid spaCy/NLTK scoring (verbs, questions, tokens, noun chunks, dep span).
- **Model Engine**: Loads LLaMA-3.2-1B from HF, applies reversible pruning, generates responses, computes PPL.
- **Benchmark System**: Measures latency (ms), tokens/sec, PPL, VRAM, power.

### Modular Structure
- `Adaptive_pruning.py`: Main script (RL agent, loops, eval runners).
- `model_loader.py`: HF loading, token auth, pruning apply/restore, save_pretrained.
- `nlp_analyzer.py`: Prompt complexity analysis.
- `pruners/`: 
  - `head_pruner.py`, `ffn_pruner.py`, `layer_skipper.py` (functional masks).
  - `structured_ffn_slicer.py`, `structured_head_slicer.py` (structural slicing).
- `.env`: Config (HF token, pruning mode).
- `checkpoints/`: RL policy files.

### Pruning Details
- **Phase 2 (Functional)**: Hooks mask outputs, reversible, for RL exploration.
- **Phase 3 (Structural)**: Rebuilds Linear layers with reduced dimensions (magnitude-based selection), real speedups. Toggle via `STRUCTURAL_PRUNING=1`.

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
   - lm-eval-harness: Export pruned model, evaluate on tasks (ARC-E, HellaSwag, etc.).

## Benchmarks

- **WikiText-2**: Perplexity and inference speed.
- **lm-eval-harness**: Standardized accuracy on ARC-E, HellaSwag, Winogrande, LAMBADA.
- **Custom Metrics**: Latency (ms), tokens/sec, perplexity in train/test.

## Roadmap

- **Phase 4**: Power/energy logging, baselines (SparseGPT, LLM-Pruner, quantization), Pareto plots.
- **Phase 5**: Full NLP feature vector in RL state, gradient-based saliency for pruning, ablations.

## Contributing

- Fork and submit PRs.
- Report issues with logs/metrics.

## License

MIT License.