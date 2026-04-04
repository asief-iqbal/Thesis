# Implementation Guide: Neuro-Adaptive Inference Engine

**Target:** Q1/A\* Thesis Grade  
**Focus:** Learned Complexity Router (LCR), Dynamic KV-Cache, and Zero-Overhead RL.

---

## 1. BERT-Tiny Selection & Rationale

- **Model Path:** `prajjwal1/bert-tiny` (L=2, H=128, A=2)
- **Rationale:** At 4.4M parameters, BERT-Tiny acts as a "Gatekeeper." It is 25x smaller than BERT-Base. In a Q1 paper, you must justify this choice: _"We utilize BERT-Tiny to ensure that the semantic profiling overhead is <1% of the total prefill latency of the Llama-3.2-1B backbone."_

---

## 2. Dataset Mixture for "Oracle" Training

To ensure the LCR generalizes, you must train on a **"Sensitivity Mixture."**

| Dataset Category | Source     | Samples | Purpose                                                          |
| ---------------- | ---------- | ------: | ---------------------------------------------------------------- |
| Logic/Math       | GSM8K      |   2,500 | High sensitivity. Pruning breaks step-by-step logic.             |
| Code             | MBPP       |   2,500 | High sensitivity. Pruning breaks syntax (brackets/tabs).         |
| Narrative        | WikiText-2 |   2,500 | Low sensitivity. High linguistic redundancy allows 50%+ pruning. |
| QA / Reading     | BoolQ      |   2,500 | Context-dependent binary QA. Tests passage grounding.            |
| Reasoning        | MMLU       |   2,500 | Mixed. Teaches BERT-Tiny the gradient of difficulty.             |

---

## 3. The "Oracle" Labeling Protocol (Step-by-Step)

You must create a Ground Truth for BERT to learn.

1. **Dense Pass:** Run each prompt through uncompressed Llama-3.2-1B. Record `PPL_D`.
2. **Sparse Pass:** Apply a fixed compression (e.g., 50% Wanda Pruning). Record `PPL_S`.
3. **Calculate Label (𝑦):**

<center>
\[
y = \text{clamp}\!\left(\frac{\text{PPL}_S - \text{PPL}_D}{\text{PPL}_D},\,0,\,1\right)
\]
</center>

4. **Result:** A CSV with `[text, sensitivity_score]`.

---

## 4. Training BERT-Tiny (The SFT Pipeline)

**Loss Function:** Mean Squared Error (MSE).

**Head:** A single MLP regression head outputting a scalar `[0.0, 1.0]`.

### Key Training Configuration

```python
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.MSELoss()
# Target: Validation R-squared > 0.80
```

---

## 5. Integrating with the RL Controller (The State Vector)

The RL Agent's state vector `S_t` must be updated to replace the old heuristic:

\[
S_t = [\text{LCR\_Score}, \text{Battery\%}, \text{VRAM\_Available}, \text{Context\_Length}]
\]

### The Action Space (Joint Compression):

The RL Agent should output a tuple \((P, K, E)\):

- **P:** Pruning Ratio (e.g., 0.0, 0.3, 0.5)
- **K:** KV-Cache Budget (e.g., 0.5, 1.0)
- **E:** Early Exit Layer (e.g., Layer 12, Layer 16)

---

## 6. Ensuring Zero Latency (The Engineering Rules)

To prevent your framework from slowing down the system, follow these **Mandatory Rules:**

1. **Async Execution:** Trigger the BERT-Tiny inference in a background thread the moment the first 128 tokens are typed.
2. **Chunk-Based Decisions:** Do NOT run the RL agent for every token. Run it once at the start of the prompt.
3. **VRAM Masking:** Pre-load masks for 30% and 50% pruning into VRAM as bitmasks. Do not re-compute them at runtime.

---

## 7. Results Visualization for Thesis

Include these three plots to guarantee an A\*:

1. **LCR Accuracy:** A scatter plot of Predicted Sensitivity vs. Actual PPL Drop.
2. **The Pareto Frontier:** Plot Accuracy vs. Latency. Show that your Dynamic Framework is closer to the top-right corner than static Wanda or H2O.
3. **Ablation Study:** Show that using BERT-Tiny is 15% more accurate than your old "Word Count" heuristic.

### Summary of Action Items for the Team:

- **Member 1 (The Miner):** Script the Llama PPL comparisons to build the Oracle dataset.
- **Member 2 (The Architect):** Fine-tune BERT-Tiny regression head and save the checkpoint.
- **Member 3 (The Strategist):** Modify the RL Agent's state vector and reward function to include the LCR score.
