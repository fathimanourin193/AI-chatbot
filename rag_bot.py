"""
Mini RAG (Retrieval-Augmented Generation) Q&A Bot
---------------------------------------------------
Upload a document (txt or pdf), ask questions about it, and get
answers grounded in that document's content using the Gemini API.

Pipeline:
  1. Load & chunk document        -> load_document(), chunk_text()
  2. Build a retriever (TF-IDF)   -> build_retriever()
  3. Retrieve relevant chunks     -> retrieve()
  4. Apply guardrails on input    -> validate_query()
  5. Build prompt & call Gemini   -> ask()
  6. Log every interaction        -> log_event()
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import google.generativeai as genai

# ---------- CONFIG ----------
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between chunks so context isn't cut off
TOP_K = 3                 # number of chunks to retrieve per query
MAX_QUERY_LENGTH = 300    # guardrail: reject very long / spammy queries
LOG_FILE = "logs/events.jsonl"

MODEL_NAME = "gemini-3.5-flash"  # fast + free-tier friendly


# ---------- STEP 1: LOAD & CHUNK DOCUMENT ----------

def load_document(path: str) -> str:
    """Reads a .txt or .pdf file and returns its raw text."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    return text


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """Splits text into overlapping chunks so retrieval can find
    relevant sections without losing context at chunk boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ---------- STEP 2 & 3: RETRIEVER (TF-IDF based) ----------

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


# ---------- STEP 4: GUARDRAILS ----------

def validate_query(query: str) -> tuple[bool, str]:
    """Basic input guardrails before sending anything to the LLM."""
    if not query or not query.strip():
        return False, "Query cannot be empty."
    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Query too long (max {MAX_QUERY_LENGTH} characters)."
    # crude prompt-injection guard: block attempts to override instructions
    lowered = query.lower()
    injection_markers = ["ignore previous instructions", "system prompt", "you are now"]
    if any(marker in lowered for marker in injection_markers):
        return False, "Query rejected by guardrail (possible prompt injection)."
    return True, ""


def validate_response(response_text: str) -> str:
    """Basic output guardrail: cap response length, strip empty responses."""
    if not response_text or not response_text.strip():
        return "The model returned an empty response. Please try rephrasing your question."
    MAX_RESPONSE_CHARS = 2000
    if len(response_text) > MAX_RESPONSE_CHARS:
        response_text = response_text[:MAX_RESPONSE_CHARS] + "... [truncated]"
    return response_text


# ---------- STEP 5: PROMPT + LLM CALL ----------

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant that answers questions ONLY using the
provided context below. If the answer is not contained in the context,
say "I don't have enough information in the document to answer that."
Do not make up information.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def ask(retriever: Retriever, query: str, model) -> str:
    ok, error_msg = validate_query(query)
    if not ok:
        log_event(query, error_msg, blocked=True)
        return f"[Blocked by guardrail] {error_msg}"

    relevant_chunks = retriever.retrieve(query)
    if not relevant_chunks:
        answer = "I couldn't find anything relevant in the document for that question."
        log_event(query, answer, blocked=False, chunks_used=0)
        return answer

    prompt = build_prompt(query, relevant_chunks)

    start_time = time.time()
    response = model.generate_content(prompt)
    latency_ms = round((time.time() - start_time) * 1000, 1)

    answer = validate_response(response.text)
    log_event(query, answer, blocked=False, chunks_used=len(relevant_chunks), latency_ms=latency_ms)
    return answer


# ---------- STEP 6: OBSERVABILITY / LOGGING ----------

def log_event(query: str, answer: str, blocked: bool, chunks_used: int = 0, latency_ms: float = None):
    """Logs every request as a JSON line -- this is a simple version of
    the 'observability' practice mentioned in LLMOps."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer_preview": answer[:150],
        "blocked_by_guardrail": blocked,
        "chunks_used": chunks_used,
        "latency_ms": latency_ms,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ---------- MAIN ----------

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set your GEMINI_API_KEY environment variable first.")
        print('  Windows (PowerShell): $env:GEMINI_API_KEY="your_key_here"')
        print('  Mac/Linux:            export GEMINI_API_KEY="your_key_here"')
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    if len(sys.argv) < 2:
        print("Usage: python rag_bot.py <path_to_document.txt or .pdf>")
        sys.exit(1)

    doc_path = sys.argv[1]
    print(f"Loading document: {doc_path}")
    text = load_document(doc_path)
    chunks = chunk_text(text)
    print(f"Document split into {len(chunks)} chunks.")

    retriever = Retriever(chunks)
    print("\nRAG bot ready. Type your questions (or 'exit' to quit).\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        answer = ask(retriever, query, model)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
