"""Step 2 and 3: find the chunks most relevant to a question (TF-IDF).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.config import TOP_K

class Retriever:
    """A lightweight retriever using TF-IDF + cosine similarity.
    (No external vector DB needed -- good enough for a single document
    and easy to explain in an interview: no black-box embeddings.)"""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.chunk_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]