"""Step 1b: split long text into overlapping chunks."""

from src.config import CHUNK_SIZE, CHUNK_OVERLAP      # overlap between chunks so context isn't cut off

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
