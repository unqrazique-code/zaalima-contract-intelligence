"""
Week 3, Day 4-7: FastAPI Serving Layer
Wraps the clause classifier (TF-IDF + Logistic Regression, trained on
actual clause text) behind a REST API. Includes a PDF upload endpoint
that runs the Week 1 OCR pipeline, then checks the extracted text against
all clause categories.

Run locally:
    uvicorn src.api:app --reload --port 8000
Then visit http://localhost:8000/docs for interactive Swagger UI.
"""

import json
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ocr_pipeline import process_pdf
from src.semantic_search import ClauseSearchIndex

TRAIN_PATH = "data/processed/train_v2.jsonl"
TARGET_CATEGORIES = [
    "Termination For Convenience",
    "Anti-Assignment",
    "Governing Law",
    "Cap On Liability",
    "Non-Compete",
]

models: dict[str, Pipeline] = {}
search_index: ClauseSearchIndex | None = None


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_input_text(category: str, text: str) -> str:
    return f"[CATEGORY: {category}] {text}"


def train_models():
    global search_index
    rows = load_jsonl(TRAIN_PATH)
    for category in TARGET_CATEGORIES:
        cat_rows = [r for r in rows if category.lower() in r["category"].lower()]
        if len(cat_rows) < 10:
            continue
        X = [build_input_text(r["category"], r["text"]) for r in cat_rows]
        y = [r["label"] for r in cat_rows]
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipeline.fit(X, y)
        models[category] = pipeline
    print(f"Trained models for {len(models)} categories: {list(models.keys())}")

    # Build the semantic search index over positive clause examples
    positive_texts = [r["text"] for r in rows if r["label"] == 1]
    search_index = ClauseSearchIndex(positive_texts)
    print(f"Built semantic search index over {len(positive_texts)} known clause examples")


@asynccontextmanager
async def lifespan(app: FastAPI):
    train_models()
    yield


app = FastAPI(
    title="Contract Intelligence API",
    description="Detects clause categories and flags risk language in contracts.",
    version="0.2.0",
    lifespan=lifespan,
)


class ClauseCheckRequest(BaseModel):
    text: str
    category: str


class ClauseCheckResponse(BaseModel):
    category: str
    clause_present: bool
    confidence: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/")
def root():
    return {
        "message": "Contract Intelligence API is running.",
        "available_categories": list(models.keys()),
        "docs": "/docs",
    }


@app.get("/categories")
def get_categories():
    return {"categories": list(models.keys())}


@app.post("/predict", response_model=ClauseCheckResponse)
def predict(req: ClauseCheckRequest):
    matched_category = next((c for c in models if c.lower() == req.category.lower()), None)
    if matched_category is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{req.category}'. Available: {list(models.keys())}",
        )
    model = models[matched_category]
    input_text = build_input_text(matched_category, req.text)
    pred = model.predict([input_text])[0]
    proba = model.predict_proba([input_text])[0][pred]
    return ClauseCheckResponse(
        category=matched_category,
        clause_present=bool(pred),
        confidence=round(float(proba), 3),
    )


@app.post("/search")
def search_similar_clauses(req: SearchRequest):
    """Week 3: semantic search over known clause examples using TF-IDF cosine similarity."""
    if search_index is None:
        raise HTTPException(status_code=503, detail="Search index not ready")
    results = search_index.search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}


@app.post("/analyze_pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    """Week 1 (OCR) + Week 2/3 (classification) combined: upload a contract
    PDF, extract its text, and check it against every clause category."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        ocr_result = process_pdf(tmp_path)
        full_text = ocr_result["full_text"]
        if not full_text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from this PDF")

        findings = {}
        for category, model in models.items():
            input_text = build_input_text(category, full_text[:2000])  # cap length for speed
            pred = model.predict([input_text])[0]
            proba = model.predict_proba([input_text])[0][pred]
            findings[category] = {
                "clause_present": bool(pred),
                "confidence": round(float(proba), 3),
            }

        return {
            "filename": file.filename,
            "pages": ocr_result["total_pages"],
            "pages_requiring_ocr": ocr_result["pages_requiring_ocr"],
            "clause_findings": findings,
        }
    finally:
        os.unlink(tmp_path)
