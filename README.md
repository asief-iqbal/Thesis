# Structured Adaptive Pruning for LLMs

A rule-based system for adaptive LLM pruning that balances inference speed, accuracy, and resource usage based on real-time hardware state and prompt complexity. Bridges static pruning (e.g., LLM-Pruner, SparseGPT) with dynamic runtime decisions using deterministic methodologies.

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

This project implements a **structured adaptive pruning system** for LLM pruning that balances inference speed, accuracy, and resource usage based on real-time hardware state and prompt complexity. Bridges static pruning (e.g., LLM-Pruner, SparseGPT) with dynamic runtime decisions using rule-based methodologies.

### Key Improvements
- **Structured Pruning**: Deterministic rules replace random RL exploration for interpretable decisions.
- **KV Cache Pruning**: Runtime reduction of generation length to limit cache usage.
- **Enhanced Evaluation**: Separate graphs (inference time & perplexity) with outliers removed and trendlines.
- **Comprehensive Training**: Automatic 80/20 split on custom CSV datasets, training on all 80% prompts.

The system is designed for A*-level research, comparing to SparseGPT, LLM-Pruner, PAT, RAP, with real pruning effects, standardized evaluation (lm-eval-harness), and rigorous training.

## Key Innovations

- **Structured Pruning**: Rule-based decisions based on prompt complexity and hardware state for interpretable, deterministic pruning.
- **Multi-Level Pruning**: Supports attention heads, FFN neurons, transformer layers, and KV cache pruning.
- **Enhanced Visualization**: Separate scatter plots with trendlines and outlier removal for inference time and perplexity.
- **Dataset Flexibility**: Automatic 80/20 splits for custom CSV datasets, training on all prompts in the split.
- **Custom Dataset Support**: Load and train/test on custom CSV files with automatic 80/20 splits.

## Features

- **Structured Controller**: Rule-based pruning decisions using prompt complexity and hardware telemetry.
- **NLP Analysis**: Hybrid spaCy/NLTK for rich prompt features (POS, noun chunks, dependency span).
- **Pruning Methods**:
  - Functional masking (hooks for validation).
  - Structural slicing (rebuilds Linear layers for speedups).
  - Runtime KV cache pruning (reduces generation length).
- **Benchmarks**: WikiText-2 perplexity, lm-eval-harness tasks, latency/tokens/sec metrics.
- **Modes**: Separate train/test CLI modes with checkpointing.
- **Safety**: Reversible pruning, no permanent model damage.
- **Local Everything**: All caches, models, datasets stored in project folder.
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
- 1: `kv_cache` (0.3) - KV cache pruning (reduces generation length).
- 2: `attention_heads` (0.4) - Attention head pruning.
- 3: `ffn_neurons` (0.5) - FFN channel pruning.
- 4: `transformer_layers` (0.3) - Layer pruning.

### System Architecture Diagram

```mermaid
flowchart TD
    subgraph Initialization
        A1[Load Model & Pruners]
        B1[Setup NLP Analyzer & Hardware Monitor]
    end

    subgraph Runtime
        A[User Prompt] --> B[NLP Analyzer]
        B --> C[Complexity Score]
        D[Hardware State] --> E[CPU/GPU/Memory]
        C --> F[Structured Controller]
        E --> F
        F --> G{Decision Rules}
        G --> H[Prune Action: Heads/FFN/Layers/KV/None]
        H --> I[Apply Pruning]
        I --> J[Generate Response]
        J --> K[Benchmark Metrics]
        K --> L[Restore Model]
    end

    subgraph Training
        M[Dataset] --> N{For Each Prompt}
        N --> O[Runtime Workflow]
        O --> P[Aggregate Results]
        P --> Q[Reports & Graphs]
    end

    A1 --> A
    B1 --> D
    L --> N
```

This cleaner diagram provides a high-level overview of the system's architecture, focusing on the key phases and data flow without excessive detail.

## Modules

- `Adaptive_pruning.py`: Main script (RL agent, loops, eval runners).
- `model_loader.py`: HF loading, token auth, pruning apply/restore, save_pretrained.
- `nlp_analyzer.py`: Prompt complexity analysis.
- `pruners/`: 
  - `head_pruner.py`, `ffn_pruner.py`, `layer_skipper.py` (functional masks).
  - `structured_ffn_slicer.py`, `structured_head_slicer.py` (structural slicing).
  - `kv_cache_pruner.py` (runtime KV cache size reduction).
- `.env`: Config (HF token, pruning mode).
- `checkpoints/`: RL policy files.
- `training_report.txt`: Post-training report.
- `training_metrics.png`: Graphs.

## Benchmarks

- **WikiText-2**: Perplexity and inference speed.
- **lm-eval-harness**: Standardized accuracy on ARC-E, HellaSwag, Winogrande, LAMBADA.
- **Custom Metrics**: Latency (ms), tokens/sec, perplexity in train/test.

## Detailed Methodology: How the Structured Pruning Architecture Works

This section provides a comprehensive, step-by-step explanation of the system's workflow, calculations, and decision-making processes.

### 1. Initialization Phase
- **Model Loading**: Load LLaMA-3.2-1B from Hugging Face using `transformers`. Requires HF token for access.
- **Pruner Setup**: Initialize all pruners:
  - Functional pruners (hooks for masking).
  - Structural slicers (rebuild layers).
  - KV cache pruner (runtime length reduction).
- **NLP Analyzer**: Load spaCy/NLTK models for prompt analysis.
- **Hardware Monitor**: Initialize NVML (if GPU) and psutil for telemetry.

### 2. Prompt Processing (Workflow)

```mermaid
flowchart TD
    A[Input Prompt] --> B[Tokenize with NLTK/spaCy]
    B --> C[Extract Features]
    C --> D[Normalize Features]
    D --> E[Calculate Complexity Score]
    E --> F[Collect Hardware State]
    F --> G[Output Complexity + Hardware Metrics]
```

**Step-by-step breakdown**:

1. **Tokenization**:
   - Use NLTK for basic word/POS tagging.
   - Use spaCy (if available) for noun chunks, sentence segmentation, dependency parsing.
   - Count LLM tokens using model's tokenizer.

2. **Feature Extraction**:
   - LLM tokens: Total tokens from model's tokenizer.
   - Verbs: Count of POS tags starting with 'V'.
   - Questions: Count of interrogative words + '?' characters.
   - Noun chunks: Number of noun phrases.
   - Dependency span: Max distance between token and head in dependency tree.
   - Sentence length: Average sentence length.

3. **Normalization**:
   - llm_norm = min(1.0, llm_tokens / 200.0)
   - q_norm = min(1.0, interrogative_count / 2.0)
   - v_norm = min(1.0, verb_count / 5.0)
   - sent_norm = min(1.0, avg_sent_len / 30.0)
   - noun_norm = min(1.0, noun_chunks / 20.0)
   - dep_norm = min(1.0, dep_span / 20.0)

4. **Complexity Score Calculation**:
   - Complexity Score = 0.4 * llm_norm + 0.05 * q_norm + 0.20 * v_norm + 0.10 * sent_norm + 0.25 * noun_norm + 0.10 * dep_norm
   - Range: 0.0 (simple prompts) to 1.0+ (highly complex prompts).

5. **Hardware State Collection**:
   - CPU utilization (%).
   - Memory available (GB).
   - GPU utilization (%) and free VRAM (GB).
   - Battery (%) if applicable.

### 3. Pruning Decision-Making (Structured Rules) (Workflow)

```mermaid
flowchart TD
    A[Complexity Score + Hardware State] --> B{Evaluate Rules}
    B -->|High Complexity + High GPU Load| C[Prune FFN Neurons]
    B -->|Medium Complexity + Low Memory| D[Prune Layers]
    B -->|Low Complexity + High CPU| E[No Pruning]
    B -->|Other Conditions| F[Prune Attention Heads or KV Cache]
    C --> G[Apply Pruning]
    D --> G
    E --> G
    F --> G
    G --> H[Output Pruning Action]
```

**Deterministic Decision Rules**:

The system uses if-else logic based on complexity and hardware thresholds:

```python
# Decision rules based on complexity and hardware constraints
if complexity > 0.8:  # High complexity prompts
    if gpu_util > 0.8 or gpu_mem_gb < 2.0:  # High GPU load or low VRAM
        action_index = 3  # ffn_neurons (aggressive pruning for speed)
    else:
        action_index = 2  # attention_heads (balanced pruning)
elif complexity > 0.5:  # Medium complexity
    if mem_gb < 4.0:  # Low memory
        action_index = 4  # transformer_layers (memory-efficient)
    else:
        action_index = 1  # kv_cache (light pruning)
else:  # Low complexity
    if cpu_util > 0.7:  # High CPU usage
        action_index = 0  # none (no pruning needed)
    else:
        action_index = 1  # kv_cache (minimal pruning)
```

### 4. Pruning Application
Based on target and intensity:

- **Attention Heads**:
  - Calculate importance: L2 norms of Q/K/V/O projections per head.
  - Remove lowest-scoring heads (e.g., intensity 0.4 → remove 40% heads).
  - Functional: Mask with hooks; Structural: Rebuild layer without heads.

- **FFN Neurons**:
  - Importance: L2 norms of gate/up/down projections per channel.
  - Remove channels with lowest scores.
  - Functional: Mask channels; Structural: Rebuild Linear layers.

- **Transformer Layers**:
  - Skip layers from the end (e.g., intensity 0.3 → skip 30% layers).
  - Functional: Hook to bypass layers.

- **KV Cache**:
  - Reduce generation length: effective_max = base_max * (1 - intensity).
  - Simulates cache pruning by limiting output tokens.

- **None**: No changes to model.

### 5. Response Generation
- Apply pruning to model.
- Generate response: `model.generate(prompt, max_new_tokens=effective_max)`.
- For KV pruning, effective_max is reduced.

### 6. Benchmarking and Evaluation
- **Perplexity Calculation**: 
  - Full text = prompt + generated response.
  - PPL = exp(cross-entropy loss) on full text.
- **Performance Metrics**:
  - Inference time (ms): Time to generate response.
  - Tokens/sec: max_new_tokens / inference_time.
- **Reward (for evaluation)**: 0.6 * tokens/sec - 0.4 * (PPL / 10).
  - Balances speed (higher reward) vs. accuracy (lower PPL).

### 7. Model Restoration
- Restore all pruners to original state for next prompt.
- Ensures reversible pruning.

### 8. Training/Evaluation Loop
- For each prompt in dataset (80% train split):
  - Process prompt → Decide pruning → Apply → Generate → Benchmark → Log metrics.
- Generate reports: Overall stats, per-prune-type averages.
- Produce graphs: Inference time vs. episode (scatter + trendline, outliers removed), Perplexity vs. episode.

### 9. Testing on Holdout Set
- Use 20% test split from CSV.
- Same process, average metrics across test prompts.
- Optional: WikiText-2 PPL, lm-eval-harness tasks.

This methodology ensures interpretable, hardware-aware pruning without randomness, adapting to prompt complexity for optimal speed-accuracy trade-offs.

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
@misc{structured_adaptive_pruning_llm,
  title={Structured Adaptive Pruning for Large Language Models},
  author={Asief Iqbal},
  year={2025},
  howpublished={\url{https://github.com/asief-iqbal/Thesis}},
  note={A rule-based system for adaptive LLM pruning balancing inference speed, accuracy, and resource usage with deterministic methodologies.}
}
```

## Acknowledgments

- Hugging Face Transformers and Datasets for model and data handling.
- PyTorch for deep learning framework.
- lm-eval-harness for standardized evaluation.
