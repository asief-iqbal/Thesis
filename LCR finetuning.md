# **High-level pipeline**

The full pipeline has 6 stages:

1. **Choose reference LLM and pruning configuration**
2. **Build oracle-labeled sensitivity dataset**
3. **Preprocess text for BERT-Tiny**
4. **Fine-tune BERT-Tiny for regression**
5. **Evaluate and calibrate the router**
6. **Export and integrate into runtime controller**

---

# **3\. Stage 1 — Choose the reference system for labeling**

Before training BERT-Tiny, you must define the system whose sensitivity it is learning.

## **3.1 Reference dense model**

Use one reference backbone:

- **Llama-3.2-1B**

This becomes the model from which sensitivity labels are derived.

## **3.2 Reference pruning method**

Pick one pruning method for the oracle labels, and keep it fixed. For example:

- Wanda at 50%  
   or
- SparseGPT at 50%

Do not mix methods during initial label generation.

## **3.3 Why fix one pruning method?**

Because the label needs a stable meaning. The label is not “universal prompt difficulty.” It is:

**degradation under a specific pruning configuration**

That is scientifically acceptable as long as you state it clearly.

---

# **4\. Stage 2 — Build the oracle-labeled dataset**

This is the most important stage.

Your guide proposes a mixed benchmark pool:

- GSM8K
- MBPP
- WikiText-2
- MMLU
- BoolQ

This is a good choice because it creates high-, low-, and mixed-sensitivity examples.

## **4.1 Data collection**

Create one combined table with columns like:

- `id`
- `source_dataset`
- `text`
- `loss_dense` / `ppl_dense`
- `loss_sparse` / `ppl_sparse`
- `loss_gap` (recommended raw metric)
- `raw_gap` (optional legacy: PPL gap)
- `normalized_sensitivity`

Example:

id | source | text | loss_dense | loss_sparse | loss_gap | ppl_dense | ppl_sparse | raw_gap | sensitivity

## **4.2 Prompt extraction**

For each dataset:

- GSM8K: use question text
- MBPP: use problem statement
- WikiText-2: use extracted text chunk or paragraph
- MMLU: use question stem plus choices if needed
- BoolQ: use question or passage

Keep prompt formatting consistent.

## **4.3 Dense pass**

Run each sample through the dense LLM and record a quality metric.

For your current setup, the simplest choice is:

- **perplexity**

So for each text sample, compute:

- `PPL_D = perplexity under dense model`

## **4.4 Pruned pass**

Run the same sample through the pruned LLM and compute:

- `PPL_S = perplexity under pruned model`

## **4.5 Raw sensitivity label**

For stability, prefer a loss-gap (log-perplexity ratio) label rather than a raw PPL gap:

- Compute teacher-forcing losses (mean cross-entropy) $\ell_D, \ell_S$
- Define: `loss_gap = max(0, ℓ_S - ℓ_D)`

This is equivalent to `log(PPL_S / PPL_D)` (up to the non-negativity clamp), and avoids heavy-tailed behavior from exponentiating losses.

You can still log `raw_gap = PPL_S - PPL_D` for backward compatibility/inspection, but it is not recommended as the primary training target.

## **4.6 Normalize the label**

You should not feed raw PPL gaps directly without normalization, because their scale may vary a lot across datasets.

Use one of these:

### **Option A: min-max normalization**

sensitivity \= (raw_gap \- min_gap) / (max_gap \- min_gap)

This gives a value in `[0,1]`.

### **Option B: percentile clipping \+ min-max**

This is safer:

1. clip raw gaps to 5th and 95th percentile
2. min-max normalize

This reduces outlier effects.

## **4.7 Why normalize?**

Because your guide expects a scalar output around `[0,1]` and this makes training more stable.

## **4.8 Recommended dataset split**

Use stratified splits by source dataset:

- 70% train
- 10% validation
- 20% test

That prevents one dataset from dominating a split.

---

# **5\. Stage 3 — Preprocess text for BERT-Tiny**

BERT-Tiny is already pretrained. You are fine-tuning it.

## **5.1 Model choice**

Use:

- `prajjwal1/bert-tiny`

Your guide explicitly recommends this model for low-overhead semantic profiling.

## **5.2 Tokenization**

Use the matching BERT tokenizer.

Preprocessing steps:

- lowercase only if tokenizer expects it
- truncate long prompts
- pad batches dynamically

## **5.3 Sequence length**

Choose a max length such as:

- 128  
   or
- 256

Recommendation:

- **128** if you want speed and lower memory
- **256** if many prompts are longer and truncation hurts performance

Since the router is just a sensitivity estimator, 128 is often enough.

## **5.4 Input format**

Use raw prompt text only.

Do not hand-engineer token length or reasoning cues as separate inputs to BERT. Let the model infer useful structure from the text.

---

# **6\. Stage 4 — Model architecture for fine-tuning**

## **6.1 Backbone**

- pretrained BERT-Tiny encoder

## **6.2 Output head**

Replace classification head with a **regression head**:

- pooled CLS embedding
- linear layer
- optional small hidden layer
- scalar output

Simple version:

BERT encoder \-\> \[CLS\] embedding \-\> Linear \-\> scalar

Slightly stronger version:

BERT encoder \-\> \[CLS\] embedding \-\> Linear \-\> GELU/ReLU \-\> Dropout \-\> Linear \-\> scalar

## **6.3 Output range**

Two choices:

### **Option A: unbounded output**

Train on raw scalar and let regression output any real value.

### **Option B: bounded output in \[0,1\]**

Use sigmoid on final layer and train on normalized labels.

For your case, I recommend:

- **sigmoid output**
- normalized labels in `[0,1]`

This matches your desired interpretation of sensitivity as a bounded score.

---

# **7\. Stage 5 — Training setup**

Your guide specifies MSE loss and AdamW, which is appropriate.

## **7.1 Loss function**

Use:

- **Mean Squared Error (MSE)**

If labels are noisy, Huber loss is also reasonable, but MSE is fine and aligns with your guide.

## **7.2 Optimizer**

Use:

- **AdamW**
- learning rate around `5e-5`

That exact scale is already suggested in the guide.

## **7.3 Batch size**

Depending on memory:

- 16
- 32
- or 64 if feasible

## **7.4 Epochs**

Start with:

- 3 to 5 epochs

Use early stopping on validation loss/correlation.

## **7.5 Regularization**

Use:

- dropout around `0.1`
- weight decay `0.01`

## **7.6 Mixed precision**

Optional, but usually not necessary for BERT-Tiny.

---

# **8\. Stage 6 — Training loop in detail**

For each training batch:

1. Tokenize batch text
2. Feed into BERT-Tiny
3. Extract `[CLS]` representation or pooled output
4. Pass through regression head
5. Compute predicted sensitivity
6. Compare with oracle sensitivity using MSE
7. Backpropagate
8. Update parameters with AdamW

Pseudo-flow:

for batch in train_loader:  
 input_ids, attention_mask, y \= batch  
 y_pred \= model(input_ids, attention_mask)  
 loss \= MSE(y_pred, y)  
 loss.backward()  
 optimizer.step()  
 optimizer.zero_grad()

---

# **9\. Stage 7 — Validation and model selection**

Do not choose the best model using only validation loss.

Use multiple criteria.

## **9.1 Primary validation metrics**

Use:

- **MSE / MAE**
- **Spearman rank correlation**
- optionally **R²**

Why Spearman matters:  
 Your router mainly needs to rank prompts by fragility, not necessarily predict perfectly calibrated values.

## **9.2 Recommended selection rule**

Pick the checkpoint with:

- lowest validation MSE  
   and
- high Spearman correlation

If they conflict, prioritize correlation slightly, because the downstream controller cares more about ordering than exact values.

## **9.3 Domain-wise validation**

Report performance separately on:

- GSM8K
- MBPP
- WikiText-2
- MMLU

This is important for showing the router did not just learn one domain.

---

# **10\. Stage 8 — Post-training calibration**

Even if the regression model works, the raw output may be noisy.

## **10.1 Optional discretization**

For runtime use, you can bucket the score into:

- low sensitivity
- medium sensitivity
- high sensitivity

Example:

- `0.0–0.33` \-\> low
- `0.34–0.66` \-\> medium
- `0.67–1.0` \-\> high

This improves controller robustness.

## **10.2 Why discretize?**

Because RL does not need ultra-precise decimal sensitivity. It needs a stable signal that separates fragile from robust prompts.

## **10.3 Keep both versions**

Store:

- continuous score for analysis
- discretized band for controller if helpful

---

# **11\. Stage 9 — Export for runtime inference**

Your guide recommends ONNX INT8 export for low overhead.

## **11.1 Save best PyTorch checkpoint**

Save:

- model weights
- tokenizer config
- normalization parameters
- label bin thresholds if using bins

## **11.2 Export to ONNX**

Export BERT-Tiny to ONNX for CPU inference.

## **11.3 Optional INT8 quantization**

Post-training quantization to INT8 is fine here because:

- router accuracy is tolerant to small noise
- latency reduction matters

## **11.4 Runtime target**

The guide suggests keeping router latency below \~10 ms and under 1% of total prefill latency. That is a good engineering target.

---

# **12\. Stage 10 — Runtime integration with controller**

At runtime:

1. Prompt arrives
2. Tokenize prompt for BERT-Tiny
3. LCR predicts sensitivity score
4. Build controller state:
   - LCR score
   - context length
   - available memory
   - battery/power state if available
5. Controller chooses pruning level
6. Apply preloaded pruning mask
7. Run LLM inference

This matches the offline/online split described in your checklist.

---

# **13\. Recommended ablations**

To make the thesis stronger, run these:

## **13.1 LCR vs heuristic predictor**

Compare BERT-Tiny against:

- token length
- word count
- maybe entropy proxy if feasible

This proves the value of learned sensitivity.

## **13.2 Single-method bias check**

Take a subset and compare sensitivity labels generated from:

- pruning method A
- pruning method B

Then compute correlation.

This addresses the criticism that your oracle is tied to one pruning scheme.

## **13.3 Cross-domain performance**

Show LCR works reasonably across all four datasets.

---

# **14\. Failure modes to watch for**

## **14.1 Label noise**

If dense/pruned PPL computation is unstable, the router learns noise.

Fix:

- use fixed evaluation settings
- keep prompt formatting consistent
- average over repeated runs if needed

## **14.2 Domain shortcut learning**

The router may simply learn:

- “code \= sensitive”
- “WikiText \= not sensitive”

Fix:

- use mixed training
- report domain-wise errors
- analyze examples where it fails within each domain

## **14.3 Overfitting**

If train loss drops but validation correlation is weak, the dataset is too small or too noisy.

Fix:

- simplify head
- add dropout
- reduce epochs
- rebalance dataset

---

# **15\. What to say in the thesis**

A clean methodology statement would be:

We fine-tune a pretrained BERT-Tiny model as a regression-based Learned Sensitivity Router (LCR). The training labels are constructed from the degradation in Llama-3.2-1B performance under a fixed pruning configuration, measured as the gap between dense and pruned perplexity. The router is trained on a mixed corpus spanning code, reasoning, math, and general text, enabling it to predict a normalized sensitivity score for previously unseen prompts. This score is then used as part of the runtime controller state to guide adaptive pruning decisions.

---

# **16\. My recommended concrete configuration**

If you want one practical default setup, use this:

## **Data**

- 2,000 samples each from:
  - GSM8K
  - MBPP
  - WikiText-2
  - MMLU

## **Labeling**

- dense Llama-3.2-1B PPL
- pruned Llama-3.2-1B PPL using one fixed method at 50%
- normalized gap in `[0,1]`

## **BERT-Tiny training**

- max length: 128
- AdamW, lr \= `5e-5`
- batch size: 32
- epochs: 4
- loss: MSE
- dropout: 0.1
- early stopping on validation Spearman \+ MSE

## **Output**

- sigmoid scalar in `[0,1]`

## **Deployment**

- export to ONNX
- optional INT8 quantization
- CPU inference before controller decision
