"""
Contract Intelligence — Streamlit Demo App
Zaalima Development · Data Science & ML Internship · Project 1 (Month 1)

Run locally:
    pip install streamlit scikit-learn pandas plotly
    streamlit run streamlit_app.py

Deploy free on Streamlit Community Cloud:
    https://streamlit.io/cloud  →  connect repo → set main file: streamlit_app.py
"""

import json
import time
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Intelligence · Zaalima Dev",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main-title {
    font-size: 2.2rem;
    font-weight: 300;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #7d8590;
    font-size: 0.95rem;
    font-weight: 300;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.stat-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #58a6ff;
}
.stat-label {
    font-size: 0.75rem;
    color: #7d8590;
    margin-top: 2px;
}
.verdict-present {
    font-size: 1.3rem;
    font-weight: 600;
    color: #f85149;
}
.verdict-absent {
    font-size: 1.3rem;
    font-weight: 600;
    color: #3fb950;
}
.clause-tag-present {
    display: inline-block;
    background: rgba(248,81,73,0.15);
    color: #f85149;
    border: 1px solid rgba(248,81,73,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
}
.clause-tag-absent {
    display: inline-block;
    background: rgba(63,185,80,0.1);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
}
.mono { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #7d8590; }
.search-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.search-sim { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #bc8cff; margin-bottom: 6px; }
.search-text { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_CATEGORIES = [
    "Termination For Convenience",
    "Anti-Assignment",
    "Governing Law",
    "Cap On Liability",
    "Non-Compete",
]

TRAIN_PATH = Path("data/processed/train_v2.jsonl")
RESULTS_PATH = Path("data/processed/classical_baseline_results.json")

SAMPLE_CLAUSES = {
    "Termination For Convenience": "Either party may terminate this Agreement for convenience upon thirty (30) days prior written notice to the other party. Upon termination, all licenses granted hereunder shall immediately cease and each party shall promptly return or destroy any Confidential Information of the other party.",
    "Anti-Assignment": "Neither party may assign, transfer, sublicense, or otherwise delegate any of its rights or obligations under this Agreement without the prior written consent of the other party, which consent shall not be unreasonably withheld.",
    "Governing Law": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles. Any disputes arising hereunder shall be subject to the exclusive jurisdiction of courts in Wilmington, Delaware.",
    "Cap On Liability": "In no event shall either party's aggregate liability arising out of or related to this Agreement exceed the total amounts paid or payable by Customer to Vendor during the twelve (12) month period immediately preceding the event giving rise to such liability.",
    "Non-Compete": "During the term of this Agreement and for a period of two (2) years following its termination, Employee shall not directly or indirectly engage in any business that competes with the Company's products or services, nor shall Employee solicit any customer or employee of the Company.",
    "Neutral (no clause)": "The parties agree to collaborate in good faith on the joint product roadmap. Regular steering committee meetings shall be held quarterly, with minutes distributed to all stakeholders within five business days of each meeting.",
}

SAMPLE_CONTRACTS = {
    "SaaS Agreement": """SOFTWARE-AS-A-SERVICE AGREEMENT

This Agreement is entered into between TechVendor Inc. ("Vendor") and Enterprise Corp. ("Customer").

LICENSE: Vendor grants Customer a non-exclusive, non-transferable license to use the Software. Customer may not assign or transfer this Agreement without Vendor's prior written consent.

TERM & TERMINATION: Either party may terminate this Agreement for convenience upon thirty (30) days written notice. Vendor may terminate immediately upon Customer's material breach.

LIABILITY: In no event shall Vendor's aggregate liability exceed the total fees paid by Customer in the preceding twelve (12) months.

GOVERNING LAW: This Agreement is governed by the laws of the State of California, and the parties consent to jurisdiction in San Francisco County.""",

    "NDA": """MUTUAL NON-DISCLOSURE AGREEMENT

The parties agree to keep all Proprietary Information confidential. Neither party shall disclose the other's Confidential Information without prior written consent.

NON-SOLICITATION: During the term and for two (2) years thereafter, neither party shall directly or indirectly solicit the other's employees or clients.

ASSIGNMENT: Neither party may assign its rights under this Agreement without written consent of the other party.

GOVERNING LAW: This Agreement shall be governed by the laws of the State of New York.""",

    "Employment Contract": """EMPLOYMENT AGREEMENT

Role: The Company employs Employee as Senior Software Engineer.

NON-COMPETE: For eighteen (18) months following termination, Employee shall not engage in any business activity that competes with the Company within the geographic region in which the Company operates.

TERMINATION: Either party may terminate this Agreement upon fourteen (14) days written notice. The Company may terminate immediately for cause.

LIABILITY CAP: Employee's total liability for any claims shall not exceed six (6) months of base salary.

GOVERNING LAW: This Agreement is governed by the laws of the State of Texas.""",
}

# ─── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Train TF-IDF + LogReg models from the CUAD dataset."""
    if not TRAIN_PATH.exists():
        return None, None

    rows = [json.loads(l) for l in TRAIN_PATH.read_text().splitlines() if l.strip()]

    models = {}
    for cat in TARGET_CATEGORIES:
        cat_rows = [r for r in rows if cat.lower() in r["category"].lower()]
        if len(cat_rows) < 10:
            continue
        X = [f"[CATEGORY: {r['category']}] {r['text']}" for r in cat_rows]
        y = [r["label"] for r in cat_rows]
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")),
            ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipe.fit(X, y)
        models[cat] = pipe

    # Semantic search index
    positive = [r["text"] for r in rows if r["label"] == 1][:2000]
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    mat = vec.fit_transform(positive)

    return models, {"vectorizer": vec, "matrix": mat, "texts": positive}


@st.cache_data
def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return []


# ─── Prediction helpers ───────────────────────────────────────────────────────
def predict(models, text, category):
    model = models.get(category)
    if not model:
        return None
    inp = f"[CATEGORY: {category}] {text}"
    pred = model.predict([inp])[0]
    proba = model.predict_proba([inp])[0][pred]
    return {"clause_present": bool(pred), "confidence": round(float(proba), 3)}


def scan_all(models, text):
    return [{"category": cat, **predict(models, text, cat)} for cat in TARGET_CATEGORIES if predict(models, text, cat)]


def semantic_search(index, query, top_k=3):
    qv = index["vectorizer"].transform([query])
    sims = cosine_similarity(qv, index["matrix"])[0]
    top_idx = sims.argsort()[::-1][:top_k]
    return [{"text": index["texts"][i][:350], "similarity": round(float(sims[i]), 3)} for i in top_idx]


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ Contract Intelligence")
    st.markdown('<div class="mono">Zaalima Dev · Month 1 · NLP</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**Model**")
    st.markdown('<div class="mono">TF-IDF + Logistic Regression<br>Trained on CUAD dataset<br>510 contracts · 41 categories</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**Performance**")
    results = load_results()
    if results:
        for r in results:
            st.markdown(f'<div class="mono">{r["category"][:22]}<br>F1: {r["f1"]:.3f}</div>', unsafe_allow_html=True)
            st.progress(r["f1"])
    st.divider()

    st.markdown("**Links**")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/unqrazique-code/zaalima-contract-intelligence)")
    st.markdown('[CUAD Dataset](https://github.com/TheAtticusProject/cuad)')

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">AI-Powered Contract Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clause detection · Risk scoring · Semantic search · Zaalima Development Internship Project 1</div>', unsafe_allow_html=True)

# Stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-num">0.961</div><div class="stat-label">F1 (classical, best cat)</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-num">0.773</div><div class="stat-label">F1 (RoBERTa)</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-num">596 req/s</div><div class="stat-label">API throughput</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-num">21.5 ms</div><div class="stat-label">p95 latency</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load models
with st.spinner("Loading models from CUAD training data…"):
    models, search_index = load_models()

if models is None:
    st.error("⚠️ Training data not found at `data/processed/train_v2.jsonl`. Run `src/prepare_training_data_v2.py` first.")
    st.stop()

st.success(f"✅ Models loaded for {len(models)} clause categories", icon="🤖")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Clause Detection", "📄 Full Contract Scan", "🔎 Semantic Search", "📊 Model Performance"])

# ── Tab 1: Clause Detection ───────────────────────────────────────────────────
with tab1:
    st.markdown("#### Detect a specific clause in contract text")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        sample_key = st.selectbox("Load a sample clause", ["— paste your own —"] + list(SAMPLE_CLAUSES.keys()))
        default_text = SAMPLE_CLAUSES.get(sample_key, "")
        text_input = st.text_area("Contract excerpt", value=default_text, height=200,
                                  placeholder="Paste a contract clause or excerpt here…")
        category = st.selectbox("Clause category to check", TARGET_CATEGORIES)
        run_btn = st.button("Analyse Clause", type="primary", use_container_width=True)

    with col_right:
        if run_btn and text_input.strip():
            with st.spinner("Running inference…"):
                time.sleep(0.3)
                result = predict(models, text_input, category)

            if result:
                st.markdown("**Result**")
                if result["clause_present"]:
                    st.markdown(f'<div class="verdict-present">⚠️ Clause detected</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="clause-tag-present">PRESENT</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="verdict-absent">✅ Not detected</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="clause-tag-absent">ABSENT</span>', unsafe_allow_html=True)

                st.markdown(f"**Category:** `{category}`")
                conf_pct = int(result["confidence"] * 100)
                st.markdown(f"**Confidence:** {conf_pct}%")
                st.progress(result["confidence"])

                st.markdown("---")
                st.markdown(f'<div class="mono">POST /predict<br>category: {category}<br>clause_present: {result["clause_present"]}<br>confidence: {result["confidence"]}</div>', unsafe_allow_html=True)
        elif run_btn:
            st.warning("Please enter some contract text first.")
        else:
            st.info("👈 Paste contract text and click **Analyse Clause**")

# ── Tab 2: Full Contract Scan ─────────────────────────────────────────────────
with tab2:
    st.markdown("#### Scan a full contract across all clause categories")

    sample_contract = st.selectbox("Load a sample contract", ["— paste your own —"] + list(SAMPLE_CONTRACTS.keys()))
    default_contract = SAMPLE_CONTRACTS.get(sample_contract, "")
    contract_text = st.text_area("Full contract text", value=default_contract, height=250,
                                  placeholder="Paste a full contract or multi-paragraph excerpt…")

    if st.button("Scan All Categories", type="primary"):
        if not contract_text.strip():
            st.warning("Please enter contract text first.")
        else:
            with st.spinner("Scanning all clause categories…"):
                time.sleep(0.4)
                scan_results = scan_all(models, contract_text)

            st.markdown("#### Findings")
            cols = st.columns(len(scan_results))
            for i, r in enumerate(scan_results):
                with cols[i]:
                    tag = "clause-tag-present" if r["clause_present"] else "clause-tag-absent"
                    label = "DETECTED" if r["clause_present"] else "ABSENT"
                    st.markdown(f'<span class="{tag}">{label}</span>', unsafe_allow_html=True)
                    st.markdown(f"**{r['category']}**")
                    st.progress(r["confidence"])
                    st.markdown(f'<div class="mono">{int(r["confidence"]*100)}% confidence</div>', unsafe_allow_html=True)

            # Risk score
            present = [r for r in scan_results if r["clause_present"]]
            risk_score = len(present) / len(scan_results)
            risk_pct = int(risk_score * 100)

            st.markdown("---")
            st.markdown("#### Overall Risk Score")
            col_score, col_desc = st.columns([1, 3])
            with col_score:
                color = "#f85149" if risk_pct >= 60 else "#d29922" if risk_pct >= 30 else "#3fb950"
                st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:3rem;font-weight:500;color:{color}">{risk_pct}%</div>', unsafe_allow_html=True)
                st.progress(risk_score)
            with col_desc:
                present_cats = [r["category"] for r in present]
                if risk_pct >= 60:
                    st.error(f"**High risk** — {len(present)} of {len(scan_results)} clause types detected: {', '.join(present_cats)}. Legal review strongly recommended before signing.")
                elif risk_pct >= 30:
                    st.warning(f"**Moderate risk** — {len(present)} clause type(s) detected: {', '.join(present_cats)}. Review flagged sections carefully.")
                else:
                    if present_cats:
                        st.success(f"**Low risk** — Only {', '.join(present_cats)} detected. Standard review recommended.")
                    else:
                        st.success("**Low risk** — No high-risk clauses detected. Standard review recommended.")

# ── Tab 3: Semantic Search ────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Find similar clauses from the CUAD dataset")
    st.markdown('<div class="mono">TF-IDF cosine similarity over 2,000+ positive clause examples</div>', unsafe_allow_html=True)
    st.markdown("")

    query = st.text_input("Ask in natural language",
        placeholder='e.g. "can either party exit the contract early?" or "what happens if I assign this agreement?"')
    top_k = st.slider("Number of results", 1, 5, 3)

    if st.button("Find Similar Clauses", type="primary"):
        if not query.strip():
            st.warning("Please enter a search query.")
        elif search_index is None:
            st.error("Search index not available.")
        else:
            with st.spinner("Searching…"):
                time.sleep(0.2)
                results = semantic_search(search_index, query, top_k)

            st.markdown(f"**Top {top_k} matches for:** *{query}*")
            for r in results:
                sim_color = "#3fb950" if r["similarity"] > 0.15 else "#d29922" if r["similarity"] > 0.05 else "#7d8590"
                st.markdown(f"""
                <div class="search-card">
                    <div class="search-sim">cosine similarity: {r['similarity']} {'▓' * int(r['similarity']*20)}</div>
                    <div class="search-text">{r['text']}</div>
                </div>
                """, unsafe_allow_html=True)

# ── Tab 4: Model Performance ──────────────────────────────────────────────────
with tab4:
    st.markdown("#### Classical Baseline Performance (TF-IDF + Logistic Regression)")
    st.markdown('<div class="mono">Trained on CUAD · data/processed/train_v2.jsonl</div>', unsafe_allow_html=True)
    st.markdown("")

    results = load_results()
    if results:
        df = pd.DataFrame(results)[["category", "accuracy", "precision", "recall", "f1", "train_size", "val_size"]]
        df.columns = ["Category", "Accuracy", "Precision", "Recall", "F1", "Train Size", "Val Size"]
        st.dataframe(df.set_index("Category").style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1"], color="#1e3a1e"
        ).format("{:.3f}", subset=["Accuracy", "Precision", "Recall", "F1"]), use_container_width=True)

        # Bar chart
        fig = go.Figure()
        cats = [r["category"] for r in results]
        for metric, color in [("precision", "#58a6ff"), ("recall", "#3fb950"), ("f1", "#bc8cff")]:
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=cats,
                y=[r[metric] for r in results],
                marker_color=color,
            ))
        fig.update_layout(
            barmode="group",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font=dict(color="#e6edf3", family="IBM Plex Mono"),
            xaxis=dict(gridcolor="#21262d", tickangle=-20),
            yaxis=dict(gridcolor="#21262d", range=[0, 1]),
            legend=dict(bgcolor="#161b22", bordercolor="#21262d"),
            margin=dict(t=20, b=20),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RoBERTa (Transformer)**")
            st.markdown('<div class="mono">F1: 0.773 · Run on Kaggle<br>Model: roberta-base<br>Notebook: notebooks/train_transformer_kaggle.ipynb</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**Load Test Results**")
            st.markdown('<div class="mono">Requests: 100 · Success: 100%<br>Throughput: ~596 req/s<br>p95 latency: 21.5 ms</div>', unsafe_allow_html=True)
    else:
        st.info("Run `src/train_classical_baseline.py` to generate performance results.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="mono" style="text-align:center">Built by Razique · Zaalima Development Internship · Project 1/3 · <a href="https://github.com/unqrazique-code/zaalima-contract-intelligence" style="color:#58a6ff">GitHub</a></div>', unsafe_allow_html=True)
