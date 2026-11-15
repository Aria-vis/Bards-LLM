"""
Data preparation script for character-level language modeling.
Downloads the Tiny Shakespeare dataset, builds character-to-integer
vocabulary mappings, and exports train/validation token arrays to binary.
"""

import os
import pickle
import requests
import numpy as np

# 1. Dataset source
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.dirname(__file__)
INPUT_FILE_PATH = os.path.join(DATA_DIR, "input.txt")

def download_dataset():
    """Download the raw text if it does not already exist locally."""
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Downloading Tiny Shakespeare dataset from {DATA_URL}...")
        response = requests.get(DATA_URL)
        with open(INPUT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Saved raw text to {INPUT_FILE_PATH}")
    else:
        print(f"Dataset already exists at {INPUT_FILE_PATH}")

def main():
    download_dataset()

    # Read raw text
    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    total_chars = len(text)
    unique_chars = sorted(list(set(text)))
    vocab_size = len(unique_chars)

    print(f"Total dataset length: {total_chars:,} characters")
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Vocabulary: {''.join(unique_chars)}")

    # 2. Build vocabulary lookup tables
    stoi = {ch: i for i, ch in enumerate(unique_chars)}
    itos = {i: ch for i, ch in enumerate(unique_chars)}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    # 3. Train / Validation split (90% / 10%)
    split_idx = int(0.9 * total_chars)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_ids = np.array(encode(train_text), dtype=np.uint16)
    val_ids = np.array(encode(val_text), dtype=np.uint16)

    print(f"Train split tokens: {len(train_ids):,}")
    print(f"Validation split tokens: {len(val_ids):,}")

    # 4. Save to binary files
    train_bin_path = os.path.join(DATA_DIR, "train.bin")
    val_bin_path = os.path.join(DATA_DIR, "val.bin")
    meta_path = os.path.join(DATA_DIR, "meta.pkl")

    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)

    # 5. Save metadata (vocab mappings) for inference
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved: {train_bin_path}")
    print(f"Saved: {val_bin_path}")
    print(f"Saved: {meta_path}")

if __name__ == "__main__":
    main()