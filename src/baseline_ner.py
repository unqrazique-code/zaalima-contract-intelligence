"""
Week 1, Day 6-7: Baseline NER Model
Uses spaCy's pretrained pipeline to extract basic entities (organizations,
dates, monetary values) from contract text. This is the "quick baseline"
before Week 2's fine-tuned transformer.

Usage:
    python src/baseline_ner.py
"""

import json
import spacy
from collections import Counter

DATA_PATH = "data/processed/cuad_flat.jsonl"
ENTITY_TYPES_OF_INTEREST = {"ORG", "DATE", "MONEY", "GPE", "LAW", "PERSON"}


def load_sample_contexts(path: str = DATA_PATH, n: int = 25) -> list[str]:
    """Grab a sample of unique contract text previews to run NER on."""
    seen_titles = set()
    contexts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["contract_title"] not in seen_titles:
                seen_titles.add(row["contract_title"])
                contexts.append(row["context_preview"])
            if len(contexts) >= n:
                break
    return contexts


def run_baseline_ner(contexts: list[str]):
    nlp = spacy.load("en_core_web_sm")
    entity_counter = Counter()
    examples = []

    for doc_text in nlp.pipe(contexts):
        found = [(ent.text, ent.label_) for ent in doc_text.ents
                 if ent.label_ in ENTITY_TYPES_OF_INTEREST]
        entity_counter.update(label for _, label in found)
        if found:
            examples.append({"text": doc_text.text[:150], "entities": found})

    return entity_counter, examples


def main():
    contexts = load_sample_contexts()
    print(f"Running baseline NER on {len(contexts)} sample contract excerpts...\n")

    counts, examples = run_baseline_ner(contexts)

    print("Entity type counts across sample:")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")

    print("\nSample extractions:")
    for ex in examples[:5]:
        print(f"\n  Text: {ex['text']}...")
        print(f"  Entities: {ex['entities']}")


if __name__ == "__main__":
    main()
