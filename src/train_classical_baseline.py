"""
Week 2, Day 5-7 (baseline for comparison): Classical Clause Classifier
A TF-IDF + Logistic Regression baseline that predicts whether a given clause
category is present in a contract excerpt. Runs on CPU in seconds, giving a
real, working benchmark to compare the fine-tuned transformer against
(Week 2, Day 1-4 -- see train_transformer_kaggle.ipynb for that piece).

Usage:
    python src/train_classical_baseline.py
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.pipeline import Pipeline

TRAIN_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"

# Focus on the categories most relevant to risk scoring -- mirrors the kind
# of clauses a compliance team cares about most (see project brief).
TARGET_CATEGORIES = [
    "Termination For Convenience",
    "Anti-Assignment",
    "Governing Law",
    "Cap On Liability",
    "Non-Compete",
    "Confidentiality",
]


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_input_text(row: dict) -> str:
    """Combine the category name with the contract excerpt -- gives the
    model the clause it's checking for, alongside the text to check."""
    return f"[CATEGORY: {row['category']}] {row['text']}"


def train_and_evaluate_category(train_rows, val_rows, category: str) -> dict:
    train_cat = [r for r in train_rows if category.lower() in r["category"].lower()]
    val_cat = [r for r in val_rows if category.lower() in r["category"].lower()]

    if len(train_cat) < 10 or len(val_cat) < 5:
        return None  # not enough data for this category

    X_train = [build_input_text(r) for r in train_cat]
    y_train = [r["label"] for r in train_cat]
    X_val = [build_input_text(r) for r in val_cat]
    y_val = [r["label"] for r in val_cat]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(y_val, preds)

    return {
        "category": category,
        "train_size": len(train_cat),
        "val_size": len(val_cat),
        "accuracy": round(acc, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def main():
    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)

    print(f"Loaded {len(train_rows)} train / {len(val_rows)} val examples\n")
    print("Training classical baseline (TF-IDF + Logistic Regression) per category...\n")

    results = []
    for cat in TARGET_CATEGORIES:
        result = train_and_evaluate_category(train_rows, val_rows, cat)
        if result:
            results.append(result)
            print(f"{result['category']:35s} | "
                  f"acc={result['accuracy']:.3f}  "
                  f"prec={result['precision']:.3f}  "
                  f"recall={result['recall']:.3f}  "
                  f"f1={result['f1']:.3f}  "
                  f"(n_train={result['train_size']}, n_val={result['val_size']})")
        else:
            print(f"{cat:35s} | skipped (insufficient data)")

    if results:
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        print(f"\nAverage F1 across {len(results)} categories: {avg_f1:.3f}")

    with open("data/processed/classical_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to data/processed/classical_baseline_results.json")


if __name__ == "__main__":
    main()
