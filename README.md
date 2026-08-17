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
- [ ] Day 3-5: OCR pipeline (Tesseract/pdf2image) for raw PDF ingestion —
  needed since CUAD ships pre-extracted text, but real-world contracts will
  come in as scanned/native PDFs.
- [x] Day 6-7: `src/baseline_ner.py` — spaCy `en_core_web_sm` baseline entity
  extraction (ORG, DATE, MONEY, GPE, PERSON). Rough but functional; fine-tuning
  in Week 2 will sharpen this significantly.

### Week 2 — Advanced NLP & Fine-Tuning (not started)
### Week 3 — Vector Search & API (not started)
### Week 4 — Integration & Productionization (not started)

## Project structure
```
contract-intelligence/
├── data/
│   ├── cuad/            # cloned dataset (gitignored)
│   └── processed/       # flattened, training-ready data
├── src/
│   ├── load_data.py     # Day 1-2
│   └── baseline_ner.py  # Day 6-7
└── requirements.txt
```
