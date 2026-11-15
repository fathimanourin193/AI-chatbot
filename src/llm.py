# ---------- STEP 5: PROMPT + LLM CALL ----------
from src.config import MODEL_NAME
 

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


def generate_answer(client, prompt: str) -> str:
    """Sends the prompt to Gemini and returns the raw text.
    May raise an exception; the caller (pipeline.py) handles it."""
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text