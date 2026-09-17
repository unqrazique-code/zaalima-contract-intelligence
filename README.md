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

### Week 3 — Vector Search & API
- [x] Day 1-3: `src/semantic_search.py` — TF-IDF + cosine similarity search
  over known clause examples. **Note:** the original plan specified
  Pinecone/Milvus (hosted vector DBs); this uses a local, dependency-free
  equivalent since hosted services need external accounts/API keys not
  practical for this project's scope. Same interface — could be swapped
  in later without changing the API layer.
- [x] Day 4-7: `src/api.py` — FastAPI app with `/predict` (single clause
  check), `/search` (semantic search), and `/analyze_pdf` (full pipeline:
  upload a PDF → OCR → check all 5 clause categories). All three endpoints
  tested end-to-end.
- [x] Data quality fix: `src/prepare_training_data_v2.py` — retrains on
  actual clause text (from CUAD answer spans) instead of a fixed
  300-character document preview. Classical baseline F1 improved from
  0.668 → 0.961 on the corrected task (see note in code: this is a
  different, more realistic task framing — classifying clause-like chunks
  rather than guessing from a document's opening lines).

### Week 4 — Integration & Productionization
- [x] Day 1-3: `Dockerfile` + `requirements-api.txt` — containerizes the
  API with Tesseract/Poppler system deps. Kept separate from the full
  `requirements.txt` (which includes training-only libs like torch/
  transformers) to keep the image lean.
- [x] Day 4-5: FastAPI's built-in Swagger UI (`/docs`) serves as the
  interactive demo frontend — no separate UI built given time constraints.
- [ ] Day 6-7: Load testing not yet performed; basic error handling and
  input validation are in place (see `api.py`).

## Running the API
```bash
pip install -r requirements-api.txt
uvicorn src.api:app --reload --port 8000
# visit http://localhost:8000/docs
```

Or with Docker:
```bash
docker build -t contract-intelligence-api .
docker run -p 8000:8000 contract-intelligence-api
```

## Project structure
```
contract-intelligence/
├── Dockerfile
├── requirements.txt          # full deps (training + API)
├── requirements-api.txt      # API-only deps (used by Docker)
├── data/
│   ├── cuad/                       # cloned dataset (gitignored)
│   └── processed/                  # flattened + train/val split data, results
├── notebooks/
│   └── train_transformer_kaggle.ipynb   # Week 2 Day 1-4, run on Kaggle
├── docs/
│   └── week2_model_comparison.md   # classical vs. transformer results
└── src/
    ├── load_data.py                  # Day 1-2
    ├── ocr_pipeline.py                # Day 3-5
    ├── baseline_ner.py                # Day 6-7
    ├── prepare_training_data.py       # Week 2 prep (v1)
    ├── prepare_training_data_v2.py    # Week 3 fix: real clause text
    ├── train_classical_baseline.py    # Week 2 Day 5-7
    ├── semantic_search.py             # Week 3 Day 1-3
    └── api.py                         # Week 3 Day 4-7, Week 4 serving layer
```

## Known limitations / honest notes
- Semantic search uses TF-IDF, not a hosted vector DB (see Week 3 note above).
- The corrected classifier (v2) is trained on isolated clause-like text
  chunks; real-world use would need a clause-segmentation step upstream
  (splitting a full contract into candidate chunks) before classification
  — not yet built.
- No automated test suite; testing was done manually via FastAPI's
  TestClient during development (see commit history).
