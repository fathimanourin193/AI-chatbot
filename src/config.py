CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between chunks so context isn't cut off
TOP_K = 3                 # number of chunks to retrieve per query
MAX_QUERY_LENGTH = 300    # guardrail: reject very long / spammy queries
MAX_RESPONSE_CHARS = 2000 # guardrail: cap the length of model answers
LOG_FILE = "logs/events.jsonl"

MODEL_NAME = "gemini-3.5-flash"  # update if you get a "model not found" error