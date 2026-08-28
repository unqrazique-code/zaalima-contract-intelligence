# AI-Powered Contract Intelligence & Risk Scoring (NLP)

Zaalima Development — Data Science & ML Internship, Project 1 (Month 1)

## What this is
An NLP system that ingests legal contracts, extracts key entities (dates,
parties, jurisdictions), classifies clauses (termination, confidentiality,
etc.), and flags high-risk language.

## Dataset
[CUAD (Contract Understanding Atticus Dataset)](https://github.com/TheAtticusProject/cuad)
— 510 commercial contracts, 41 annotated clause categories, 13,000+ labels.

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Get the data (already gitignored — don't commit the raw dataset)
git clone --depth 1 https://github.com/TheAtticusProject/cuad.git data/cuad
cd data/cuad && unzip data.zip -d extracted && cd ../..
```

## Progress log

### Week 1 — Data Parsing & Baseline Modeling
- [x] Day 1-2: `src/load_data.py` — flattens CUAD's nested SQuAD-style JSON
  into `data/processed/cuad_flat.jsonl` (20,910 contract/category rows across
  510 contracts).
- [x] Day 3-5: `src/ocr_pipeline.py` — hybrid pipeline: native text
  extraction via pdfplumber first, falls back to Tesseract OCR per-page for
  scanned/image-only PDFs. Tested on both native-text and scanned samples.
- [x] Day 6-7: `src/baseline_ner.py` — spaCy `en_core_web_sm` baseline entity
  extraction (ORG, DATE, MONEY, GPE, PERSON). Rough but functional; fine-tuning
  in Week 2 will sharpen this significantly.

### Week 2 — Advanced NLP & Fine-Tuning
- [x] Day 1 (prep): `src/prepare_training_data.py` — splits the flattened CUAD
  data into train/val at the *contract* level (no leakage): 17,794 train /
  3,116 val examples across 510 contracts.
- [x] Day 1-4: `notebooks/train_transformer_kaggle.ipynb` — fine-tunes
  `roberta-base` on 5 key clause categories (Termination For Convenience,
  Anti-Assignment, Governing Law, Cap On Liability, Non-Compete). Runs on
  Kaggle (GPU required, not runnable in this sandbox due to no Hugging Face
  Hub access here).
- [x] Day 5-7: `src/train_classical_baseline.py` — TF-IDF + Logistic
  Regression baseline for the same categories, for comparison against the
  fine-tuned transformer. **Result: 0.668 average F1** across 5 categories
  (see `data/processed/classical_baseline_results.json`).

### Week 3 — Vector Search & API (not started)
### Week 4 — Integration & Productionization (not started)

## Project structure
```
contract-intelligence/
├── data/
│   ├── cuad/            # cloned dataset (gitignored)
│   └── processed/       # flattened + train/val split data, results
├── notebooks/
│   └── train_transformer_kaggle.ipynb   # Week 2 Day 1-4, run on Kaggle
├── src/
│   ├── load_data.py               # Day 1-2
│   ├── ocr_pipeline.py            # Day 3-5
│   ├── baseline_ner.py            # Day 6-7
│   ├── prepare_training_data.py   # Week 2 prep
│   └── train_classical_baseline.py # Week 2 Day 5-7
└── requirements.txt
```
