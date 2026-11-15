"""Step 1a: read a .txt or .pdf file into plain text."""

from pypdf import PdfReader


def load_document(path: str) -> str:
    """Reads a .txt or .pdf file and returns its raw text."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    return text
