"""
Week 2/3 fix: Prepare training data using the ACTUAL clause text
(from CUAD's answer spans) instead of a fixed 300-char document preview.
This fixes a real data quality issue: the original approach used the
first 300 characters of each contract regardless of category, which is
often just boilerplate (title, addresses) unrelated to the clause being
checked, hurting classifier accuracy and demo reliability.

For positive examples: use the actual answer span text.
For negative examples: sample a random paragraph-length chunk of the
contract that is NOT part of any positive answer span for that category.

Usage:
    python src/prepare_training_data_v2.py
Produces:
    data/processed/train_v2.jsonl
    data/processed/val_v2.jsonl
"""

import json
import random

IN_PATH = "data/processed/cuad_flat.jsonl"
CUAD_RAW_PATH = "data/cuad/extracted/CUADv1.json"
TRAIN_PATH = "data/processed/train_v2.jsonl"
VAL_PATH = "data/processed/val_v2.jsonl"
VAL_FRACTION = 0.15
SEED = 42


def load_raw_contexts() -> dict[str, str]:
    """Map contract title -> full context text, from the original CUAD file."""
    with open(CUAD_RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    contexts = {}
    for contract in raw["data"]:
        # CUAD stores one paragraph per contract in this dataset version
        contexts[contract["title"]] = contract["paragraphs"][0]["context"]
    return contexts


def load_flat_rows(path: str = IN_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sample_negative_chunk(context: str, rnd: random.Random, chunk_len: int = 300) -> str:
    """Grab a random chunk of the contract text as a negative example."""
    if len(context) <= chunk_len:
        return context
    start = rnd.randint(0, len(context) - chunk_len)
    return context[start:start + chunk_len]


def build_examples(rows: list[dict], contexts: dict[str, str], rnd: random.Random) -> list[dict]:
    examples = []
    for r in rows:
        title = r["contract_title"]
        if r["has_clause"] and r["answers"]:
            # Positive: use the actual clause text (concatenate all spans for this category)
            text = " ".join(a["text"] for a in r["answers"])
        else:
            # Negative: random chunk of the contract, unrelated to this category
            full_context = contexts.get(title, r.get("context_preview", ""))
            text = sample_negative_chunk(full_context, rnd)

        if not text.strip():
            continue

        examples.append({
            "text": text,
            "category": r["category"],
            "label": int(r["has_clause"]),
        })
    return examples


def split_by_contract(rows: list[dict], val_fraction: float = VAL_FRACTION, seed: int = SEED):
    contract_titles = sorted({r["contract_title"] for r in rows})
    rnd = random.Random(seed)
    rnd.shuffle(contract_titles)
    n_val = max(1, int(len(contract_titles) * val_fraction))
    val_titles = set(contract_titles[:n_val])
    train_rows = [r for r in rows if r["contract_title"] not in val_titles]
    val_rows = [r for r in rows if r["contract_title"] in val_titles]
    return train_rows, val_rows


def main():
    print("Loading raw CUAD contexts (full contract text)...")
    contexts = load_raw_contexts()
    rows = load_flat_rows()
    train_rows, val_rows = split_by_contract(rows)

    rnd = random.Random(SEED)
    train_examples = build_examples(train_rows, contexts, rnd)
    val_examples = build_examples(val_rows, contexts, rnd)

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples:   {len(val_examples)}")
    print(f"Saved: {TRAIN_PATH}, {VAL_PATH}")


if __name__ == "__main__":
    main()
