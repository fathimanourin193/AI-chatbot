"""Step 4: input and output safety checks."""

from src.config import MAX_QUERY_LENGTH,MAX_RESPONSE_CHARS

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

    if len(response_text) > MAX_RESPONSE_CHARS:
        response_text = response_text[:MAX_RESPONSE_CHARS] + "... [truncated]"
    return response_text