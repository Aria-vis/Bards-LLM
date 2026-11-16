# Bard's LLM

A lightweight, high-performance decoder-only Transformer (GPT) built from scratch in PyTorch. Designed for both pedagogical clarity and production systems engineering, featuring **FlashAttention**, **Weight Tying**, **Pre-LayerNorm residual pathways**, **Zero-RAM memory-mapped dataloading**, and a **custom checkpoint importer for official OpenAI GPT-2 weights**.

---

## Architecture Overview

```
Input Tokens (B, T)
        │
┌───────┴───────┐
│ Token + Pos   │  wte (Vocab -> d_model) + wpe (Context -> d_model)
│  Embeddings   │
└───────┬───────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│ Transformer Block (Repeated L times)                        │
│   ├── Pre-LayerNorm (ln_1)                                  │
│   ├── Causal Self-Attention (Fused QKV + FlashAttention)    │
│   ├── Residual Connection (+)                               │
│   ├── Pre-LayerNorm (ln_2)                                  │
│   ├── Position-wise Feed-Forward (Linear -> GELU -> Linear) │
│   └── Residual Connection (+)                               │
└───────┬─────────────────────────────────────────────────────┘
        │
┌───────▼───────┐
│ Final LN      │  ln_f (d_model)
└───────┬───────┘
        │
┌───────▼───────┐
│ LM Head       │  lm_head (Tied with wte.weight) ──► Logits (B, T, Vocab)
└───────────────┘
```

---

## Key Systems Engineering Highlights

### 1. Pre-LayerNorm Residual Highway
Unlike the original 2017 Transformer paper (Post-LN), Layer Normalization is applied *before* the multi-head attention and MLP sub-layers. This preserves an unobstructed residual pathway, allowing gradients to propagate directly through the depth of the network without vanishing or exploding.

### 2. Batched QKV Linear Projection
Instead of three separate linear layers for Queries, Keys, and Values, all three projections are computed in a single batched GEMM matrix multiplication (`Linear(d_model, 3 * d_model)`). This maximizes hardware throughput and eliminates GPU kernel launch overhead.

### 3. PyTorch 2.x FlashAttention Integration
Attention computation leverages `torch.nn.functional.scaled_dot_product_attention`, fusing softmax and matrix multiplications into fast SRAM tiles to avoid the $O(T^2)$ memory bottleneck of naive attention matrices. Includes a manual triangular causal mask fallback for backward compatibility.

### 4. Parameter Tying (Press & Wolf, 2017)
The output classification matrix (`lm_head.weight`) shares physical memory with the input token embedding table (`wte.weight`). This reduces model parameter count by up to 30% and enforces unified geometric representations between token inputs and language modeling targets.

### 5. Parameter-Segregated Weight Decay in AdamW
To prevent degradation of representation capacity, weight decay (0.1) is applied strictly to 2D weight tensors (embedding matrices and linear projection kernels). 1D parameters (LayerNorm gains and biases) have weight decay disabled (`0.0`).

### 6. OpenAI GPT-2 Weight Transposition Importer
Includes a custom `@classmethod from_pretrained()` capable of downloading official OpenAI GPT-2 checkpoints (`124M`, `350M`, `774M`, `1558M`) from Hugging Face. The loader automatically converts OpenAI's legacy TensorFlow `Conv1D` weight matrices (`[fan_in, fan_out]`) into standard PyTorch `nn.Linear` matrices (`[fan_out, fan_in]`).

### 7. Zero-RAM Memory-Mapped Dataloading
The data pipeline pre-tokenizes raw text into 16-bit unsigned integer binaries (`train.bin`, `val.bin`). The batch loader queries slices on-the-fly using `numpy.memmap`, enabling training on multi-gigabyte corpora with virtually zero RAM footprint.

---

## Directory Structure

```text
Bard's LLM/
├── bard/                     # Core library package
│   ├── __init__.py
│   ├── config.py             # Hyperparameter dataclass (BardConfig)
│   └── model.py              # Decoder-only Transformer & GPT-2 weight loader
├── data/                     # Dataset preprocessing utilities
│   ├── prepare.py            # Tokenization and memory-mapped binary generator
├── scripts/                  # Standalone execution pipelines
│   ├── train.py              # Training loop with warmup & cosine decay
│   └── generate.py           # Autoregressive sampling (Temperature & Top-K)
├── checkpoints/              # Model weights & training states (.pt)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart Guide

### 1. Environment Setup
```powershell
# Clone the repository
git clone <your-repo-url>
cd "Bard's LLM"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Download the Tiny Shakespeare dataset, build vocabulary indices, and generate binary buffers:
```powershell
python data/prepare.py
```

### 3. Train from Scratch
Train an 800K-parameter model with cosine learning rate scheduling:
```powershell
python scripts/train.py
```

### 4. Text Generation & Sampling
Sample text using your trained checkpoint:
```powershell
python scripts/generate.py
```

Or switch `model_source = "gpt2"` inside `scripts/generate.py` to sample directly from OpenAI's 124M-parameter GPT-2 using Byte-Pair Encoding (`tiktoken`).

---

## Training Benchmarks (Shakespeare Character Model)

| Iteration | Train Loss | Validation Loss | Learning Rate | Observations |
|:---:|:---:|:---:|:---:|:---|
| **0** | 4.1806 | 4.1781 | 9.90e-6 | Baseline random entropy ($\ln(65) \approx 4.174$) |
| **200** | 2.4816 | 2.4715 | 9.73e-4 | Learns basic letter pairings and spaces |
| **400** | 2.3804 | 2.3901 | 7.75e-4 | Learns common English prefixes & suffixes |
| **600** | 2.2637 | 2.2851 | 4.72e-4 | Dialogue formatting and speaker colons emerge |
| **800** | 2.1989 | 2.2209 | 2.05e-4 | Shakespearean vocabulary and line rhythm |
| **1000** | 2.1622 | 2.1914 | 1.00e-4 | Coherent dramatic structure and contractions |

---

## Sample Outputs

### Trained Model (From scratch on Shakespeare, Step 1000):
```text
ROMEO:
That, go knoutad touth that I hourd he beake.

CARORTARD'd ENIIW:
Cellll my wit toor oftond wir
Thir hat an cecnd pevear an he send weith;-nd grond as:
I'llld gutr omor the heand a achong dany sind thaver.
```

### Pretrained GPT-2 (Loaded into BardLM, BPE Subword):
```text
Prompt: 'Alan Turing was a mathematician who'
Output:
Alan Turing was a mathematician who was highly respected in the world 
of mathematical linguistics. It is difficult to understand why he was 
known only as "The Turing Machine", because it is very difficult to 
describe the system of Turing's system...
```

---

## Technical Interview Cheatsheet

### Q1: Why Pre-LN instead of Post-LN?
*Answer:* The original Transformer paper used Post-LN (`x = Norm(x + Sublayer(x))`). In deep networks, Post-LN causes the gradient magnitude at early layers to be much smaller than at later layers, requiring strict warmup and small learning rates. Pre-LN (`x = x + Sublayer(Norm(x))`) keeps the residual connection untouched, forming an identity highway where gradients flow directly back from the output to early layers.

### Q2: Why do GPT-2 linear weights need to be transposed?
*Answer:* OpenAI's original TensorFlow GPT-2 code used a custom module called `Conv1D`, which stores 2D weight kernels with shape `[fan_in, fan_out]`. Standard PyTorch `nn.Linear` layers store weights with shape `[fan_out, fan_in]`. Hugging Face preserved OpenAI's tensor shapes, so when transferring weights to standard PyTorch linear layers, all projection weight matrices must be transposed (`.t()`).

### Q3: Why disable weight decay on 1D parameters?
*Answer:* Weight decay ($L_2$ regularization) penalizes large weight magnitudes. In linear layers ($y = Wx + b$), penalizing $W$ prevents overfitting by favoring smoother decision surfaces. However, biases ($b$) and LayerNorm parameters ($\gamma, \beta$) determine the baseline shift and scale of activations. Penalizing them restricts the network's capacity to normalize signals without providing meaningful regularization benefits.

---

## References

1. Vaswani et al., *"Attention Is All You Need"*, 2017.
2. Radford et al., *"Language Models are Unsupervised Multitask Learners"* (GPT-2), 2019.
3. Dao et al., *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*, 2022.
4. Press & Wolf, *"Using the Output Embedding to Improve Language Models"*, 2017.