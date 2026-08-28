"""
Week 2, Day 1 (prep step): Prepare Training Data
Converts the flattened CUAD data (from load_data.py) into a train/val split
suitable for clause classification: given a chunk of contract text and a
clause category, predict whether that clause is present (binary).

Usage:
    python src/prepare_training_data.py
Produces:
    data/processed/train.jsonl
    data/processed/val.jsonl
"""

import json
import random

IN_PATH = "data/processed/cuad_flat.jsonl"
TRAIN_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"
VAL_FRACTION = 0.15
SEED = 42


def load_rows(path: str = IN_PATH) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_examples(rows: list[dict]) -> list[dict]:
    """One example per (contract, category): text + category name -> label."""
    examples = []
    for r in rows:
        # Skip rows with no usable context text
        if not r.get("context_preview"):
            continue
        examples.append({
            "text": r["context_preview"],
            "category": r["category"],
            "label": int(r["has_clause"]),
        })
    return examples


def split_by_contract(rows: list[dict], val_fraction: float = VAL_FRACTION, seed: int = SEED):
    """Split at the CONTRACT level (not row level) to avoid leakage --
    all clause rows for a given contract stay together in train or val."""
    contract_titles = sorted({r["contract_title"] for r in rows})
    rnd = random.Random(seed)
    rnd.shuffle(contract_titles)

    n_val = max(1, int(len(contract_titles) * val_fraction))
    val_titles = set(contract_titles[:n_val])

    train_rows = [r for r in rows if r["contract_title"] not in val_titles]
    val_rows = [r for r in rows if r["contract_title"] in val_titles]
    return train_rows, val_rows


def main():
    rows = load_rows()
    train_rows, val_rows = split_by_contract(rows)

    train_examples = build_examples(train_rows)
    val_examples = build_examples(val_rows)

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Contracts total: {len({r['contract_title'] for r in rows})}")
    print(f"Train examples: {len(train_examples)} (from {len({r['contract_title'] for r in train_rows})} contracts)")
    print(f"Val examples:   {len(val_examples)} (from {len({r['contract_title'] for r in val_rows})} contracts)")
    print(f"Saved: {TRAIN_PATH}, {VAL_PATH}")


if __name__ == "__main__":
    main()
