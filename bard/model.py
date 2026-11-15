"""
Bard's Language Model (Decoder-only Transformer).
Features:
- Fused QKV linear projections
- PyTorch 2.x FlashAttention (scaled_dot_product_attention) with fallback
- Pre-LayerNorm residual architecture
- Weight tying between token embeddings and the LM head
- Parameter-specific AdamW weight decay configuration
"""

import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

from bard.config import BardConfig


class LayerNorm(nn.Module):
    """LayerNorm with optional learnable bias (PyTorch's nn.LayerNorm lacks bias=False toggle in some versions)."""

    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    Computes batched Q, K, V projections and applies causal masking so tokens
    can only attend to current and preceding tokens in the sequence.
    """

    def __init__(self, config: BardConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # 1. Batched QKV projection: projects n_embd -> 3 * n_embd in one matmul
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 2. Output projection back to residual stream
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # PyTorch 2.0+ FlashAttention check
        self.has_flash = hasattr(F, "scaled_dot_product_attention")
        if not self.has_flash:
            # Fallback causal mask for older PyTorch versions
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension (n_embd)

        # Calculate Query, Key, Values for all heads simultaneously
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (Batch, Heads, Time, Head_Dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal Attention calculation
        if self.has_flash:
            # Efficient attention kernel (FlashAttention-style)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual scaled dot-product attention fallback
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            out = weights @ v

        # Re-assemble all attention head outputs side by side -> (Batch, Time, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection into residual stream
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP).
    Projects n_embd -> 4 * n_embd -> GELU -> n_embd.
    """

    def __init__(self, config: BardConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer Block: Communication (Attention) followed by Computation (MLP).
    Uses Pre-LayerNorm residual connections.
    """

    def __init__(self, config: BardConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual connections: x + Sublayer(LN(x))
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BardLM(nn.Module):
    """
    The Full Bard Decoder-Only Language Model.
    """

    def __init__(self, config: BardConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            "ln_f": LayerNorm(config.n_embd, bias=config.bias),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with final classifier
        self.transformer.wte.weight = self.lm_head.weight

        # Weight initialization
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (per GPT-2 paper)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        """Standard Gaussian initialization for weights, zero initialization for biases."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Count trainable parameters.
        By default, subtracts position embeddings (wpe) since they don't scale with sequence length.
        Token embeddings are kept in the count because they are tied with lm_head.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence length {t}, block_size is {self.config.block_size}"
        )

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward through embeddings
        tok_emb = self.transformer.wte(idx)       # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)       # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training: compute cross-entropy loss across all positions
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: only project the last time step [-1]
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.Optimizer:
        """
        Separates parameters into:
        1. 2D parameters (weights in Linear/Embeddings) -> Regularized with weight decay.
        2. 1D parameters (biases, LayerNorm weights) -> No weight decay (prevents capacity loss).
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW if available on CUDA
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> torch.Tensor:
        """
        Autoregressively generates new tokens by conditioning on existing context.
        Applies temperature scaling and top-k filtering.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds the model's block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward the model
            logits, _ = self(idx_cond)
            
            # Pluck the last token's logits and scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering: clamp logits below the top-k threshold to -inf
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx