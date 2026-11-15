"""Ties all the steps together: validate -> retrieve -> prompt -> generate -> log."""

import time
from src.Guardrails import validate_query, validate_response
from src.llm import build_prompt, generate_answer
from src.logger import log_event
from src.retriever import Retriever



def ask(retriever: Retriever, query: str, client) -> str:
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
    try:
        raw_answer = generate_answer(client, prompt)
    except Exception as e:
        answer = f"The model call failed: {e}"
        log_event(query, answer, blocked=False, chunks_used=len(relevant_chunks))
        return answer
    latency_ms = round((time.time() - start_time) * 1000, 1)

    answer = validate_response(raw_answer)
    log_event(query, answer, blocked=False, chunks_used=len(relevant_chunks), latency_ms=latency_ms)
    return answer