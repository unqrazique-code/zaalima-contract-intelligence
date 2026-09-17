"""
Week 3, Day 1-3: Semantic Search
Builds a searchable index over known clause examples so a user can query
in natural language (e.g. "can either party end the contract early?")
and retrieve the most similar real clauses from the dataset.

Note on implementation choice: the official project plan suggests Pinecone
or Milvus (hosted vector databases). Those require external network
services and API keys that aren't practical to provision for an internship
project demo. This uses scikit-learn's TF-IDF + cosine similarity as a
lightweight, dependency-free vector search that runs entirely locally --
same core idea (embed text, find nearest neighbors), simpler infrastructure.
Swapping in a real vector DB later just means replacing the `search()`
internals; the API layer (`/search` endpoint) would not need to change.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ClauseSearchIndex:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_vectors)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        return [
            {
                "text": self.documents[i][:300],
                "similarity": round(float(sims[i]), 3),
            }
            for i in top_indices
        ]
