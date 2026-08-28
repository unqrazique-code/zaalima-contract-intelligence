# Week 2: Model Comparison — Classical Baseline vs. Fine-Tuned Transformer

## Task
Binary classification: given a contract excerpt and a clause category
(e.g. "Termination For Convenience"), predict whether that clause is
present. Evaluated on the same 5 CUAD categories for both models:
Termination For Convenience, Anti-Assignment, Governing Law,
Cap On Liability, Non-Compete.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Classical baseline** (TF-IDF + Logistic Regression, trained per-category) | 0.716 | 0.653 | 0.686 | 0.668 |
| **Fine-tuned Transformer** (`roberta-base`, jointly trained across categories, 3 epochs) | **0.755** | **0.823** | 0.728 | **0.773** |

## Analysis

**The transformer outperforms the classical baseline on every metric**,
with the biggest gain in **precision** (+0.17) — meaning when the
transformer flags a clause as present, it's right more often. This
matters directly for the project's use case: a legal/compliance tool
that cries wolf too often (low precision) gets ignored by its users,
even if it never misses a real clause.

**F1 improved from 0.668 → 0.773 (+0.105)**, roughly a 16% relative
improvement — a meaningful gain given the classical baseline was already
a reasonably strong, well-regularized model (balanced class weights,
bigram TF-IDF features).

**Why the transformer wins here:** TF-IDF treats text as a bag of
weighted word/bigram counts with no understanding of word order, context,
or legal-domain semantics. RoBERTa's pretraining gives it contextual
understanding — e.g. it can distinguish "the Agreement **shall not**
be assigned" from "the Agreement **may** be assigned" even though both
share almost identical surface vocabulary, whereas TF-IDF sees these as
nearly the same bag of words.

**Trade-off worth noting:** the transformer needed a GPU and ~15-20
minutes to fine-tune, versus the classical baseline running in seconds
on CPU. For a real production system, this justifies the transformer's
extra cost given the precision gain — but the classical baseline
remains a useful fast fallback / sanity-check model.

## Next steps (Week 3)
Use the fine-tuned transformer as the primary clause classifier, served
via the FastAPI application, with the classical baseline kept as a
lightweight secondary signal for the risk-scoring logic.
