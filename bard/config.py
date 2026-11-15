"""
Configuration dataclass for Bard's Language Model.
"""

from dataclasses import dataclass

@dataclass
class BardConfig:
    # Context & Vocabulary
    block_size: int = 128     # Maximum context length (tokens)
    vocab_size: int = 65      # Number of tokens in vocabulary (default: character-level Shakespeare)
    
    # Model Architecture
    n_layer: int = 4          # Number of Transformer blocks (stack depth)
    n_head: int = 4           # Number of attention heads
    n_embd: int = 128         # Embedding dimension (must be divisible by n_head)
    
    # Regularization & Details
    dropout: float = 0.1      # Dropout rate
    bias: bool = False        # True: GPT-2 style bias; False: LLaMA-style faster/cleaner matmuls
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )