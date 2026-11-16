"""
Text generation script for Bard's Language Model.
Loads a trained checkpoint and samples tokens autoregressively.
"""

import os
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bard.config import BardConfig
from bard.model import BardLM

# -----------------------------------------------------------------------------
# Generation Settings
# -----------------------------------------------------------------------------
checkpoint_path = os.path.join("checkpoints", "best_model.pt")
meta_path = os.path.join("data", "meta.pkl")
device = "cuda" if torch.cuda.is_available() else "cpu"

num_samples = 3          # How many independent samples to generate
max_new_tokens = 400     # Number of characters to generate per sample
temperature = 0.8        # Higher = more creative/random, Lower = more deterministic
top_k = 20               # Only sample from the top_k most likely characters
prompt = "ROMEO:\n"      # Prompt to condition the model on (use "\n" for unconditional)
# -----------------------------------------------------------------------------


def main():
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Train the model first!")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 1. Reconstruct model from checkpoint config
    config: BardConfig = checkpoint["config"]
    model = BardLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded model from iteration {checkpoint['iter_num']} (Best Val Loss: {checkpoint['best_val_loss']:.4f})")

    # 2. Load tokenizer metadata
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    itos = meta["itos"]

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    def decode(ids: list[int]) -> str:
        return "".join([itos[i] for i in ids])

    # 3. Encode prompt
    prompt_ids = encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    print("\n" + "=" * 65)
    print(f"PROMPT: {repr(prompt)}")
    print(f"SAMPLING CONFIG: temperature={temperature}, top_k={top_k}, max_new_tokens={max_new_tokens}")
    print("=" * 65 + "\n")

    # 4. Generate text
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