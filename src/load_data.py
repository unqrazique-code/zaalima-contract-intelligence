"""
Week 1, Day 1-2: Data Parsing
Loads the raw CUAD dataset (SQuAD-style JSON) and converts it into a clean,
training-ready format: one row per (contract, clause_category) with the
paragraph text, the question/category, and the answer span (if any).

Usage:
    python src/load_data.py
Produces:
    data/processed/cuad_flat.jsonl
"""

import json
import os

RAW_PATH = "data/cuad/extracted/CUADv1.json"
OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "cuad_flat.jsonl")


def flatten_cuad(raw_path: str = RAW_PATH) -> list[dict]:
    """Flatten CUAD's nested SQuAD-style JSON into a flat list of records."""
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for contract in raw["data"]:
        contract_title = contract["title"]
        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                # Each "question" encodes the clause category being checked.
                category = qa["question"].split('"')[1] if '"' in qa["question"] else qa["question"]
                has_answer = len(qa["answers"]) > 0
                records.append({
                    "contract_title": contract_title,
                    "category": category,
                    "question": qa["question"],
                    "has_clause": has_answer,
                    "answers": [
                        {"text": a["text"], "start": a["answer_start"]}
                        for a in qa["answers"]
                    ],
                    "context_preview": context[:300],  # keep file size sane
                })
    return records


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = flatten_cuad()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_contracts = len({r["contract_title"] for r in records})
    n_with_clause = sum(1 for r in records if r["has_clause"])
    print(f"Contracts processed: {n_contracts}")
    print(f"Total (contract, category) rows: {len(records)}")
    print(f"Rows where the clause IS present: {n_with_clause}")
    print(f"Rows where the clause is absent: {len(records) - n_with_clause}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
