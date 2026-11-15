import os
import sys

from google import genai

from src.loader import load_document
from src.chunking import chunk_text
from src.retriever import Retriever
from src.pipeline import ask

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set your GEMINI_API_KEY environment variable first.")
        print('  Windows (PowerShell): $env:GEMINI_API_KEY="your_key_here"')
        print('  Mac/Linux:            export GEMINI_API_KEY="your_key_here"')
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    if len(sys.argv) < 2:
        print("Usage: python cli.py <path_to_document.txt or .pdf>")
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
        answer = ask(retriever, query, client)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
