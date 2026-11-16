"""
Text generation script for Bard's Language Model.
Supports:
1. Custom trained checkpoints on Tiny Shakespeare (character-level)
2. Official OpenAI GPT-2 pretrained weights (subword BPE with tiktoken)
"""

import os
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import tiktoken

from bard.config import BardConfig
from bard.model import BardLM

# -----------------------------------------------------------------------------
# Generation Settings
# -----------------------------------------------------------------------------
# Set model_source to 'checkpoint' (your Shakespeare model) or 'gpt2' (OpenAI 124M weights)
model_source = "gpt2"   # Options: 'checkpoint' or 'gpt2'
checkpoint_path = os.path.join("checkpoints", "best_model.pt")
meta_path = os.path.join("data", "meta.pkl")
device = "cuda" if torch.cuda.is_available() else "cpu"

num_samples = 2          # Number of samples to generate
max_new_tokens = 60      # Number of tokens to generate
temperature = 0.8        # Higher = creative, Lower = deterministic
top_k = 40               # Top-k probability sampling threshold
prompt = "Alan Turing was a mathematician who"
# -----------------------------------------------------------------------------


def main():
    print(f"Running generation on device: {device}")

    # 1. Load model and tokenizer depending on source
    if model_source == "gpt2":
        # Load official OpenAI GPT-2 weights into our custom BardLM architecture
        model = BardLM.from_pretrained("gpt2").to(device)
        model.eval()

        # OpenAI BPE Tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        encode = lambda s: tokenizer.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: tokenizer.decode(l)

    elif model_source == "checkpoint":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Train the model first!")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config: BardConfig = checkpoint["config"]
        model = BardLM(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(f"Loaded checkpoint from iter {checkpoint['iter_num']} (Val Loss: {checkpoint['best_val_loss']:.4f})")

        # Load character-level tokenizer
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])

    else:
        raise ValueError(f"Unknown model_source: {model_source}")

    # 2. Encode prompt
    prompt_ids = encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    print("\n" + "=" * 65)
    print(f"MODEL SOURCE: {model_source.upper()}")
    print(f"PROMPT: {repr(prompt)}")
    print(f"SAMPLING CONFIG: temperature={temperature}, top_k={top_k}, max_new_tokens={max_new_tokens}")
    print("=" * 65 + "\n")

    # 3. Generate text
    with torch.no_grad():
        for sample_idx in range(1, num_samples + 1):
            print(f"--- [Sample {sample_idx}/{num_samples}] ---")
            generated_ids = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            output_text = decode(generated_ids[0].tolist())
            print(output_text)
            print("-" * 65 + "\n")


if __name__ == "__main__":
    main()