
"""Step 6: append every request to a JSON-lines log file."""

import os
import json
from datetime import datetime, timezone
from src.config import LOG_FILE
 
 
def log_event(query: str, answer: str, blocked: bool, chunks_used: int = 0, latency_ms: float = None):
    """Logs every request as a JSON line (a simple form of observability)."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer_preview": answer[:150],
        "blocked_by_guardrail": blocked,
        "chunks_used": chunks_used,
        "latency_ms": latency_ms,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
 