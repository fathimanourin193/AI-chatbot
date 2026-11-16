"""
Evaluation runner for the RAG bot.

Usage (from the project root, with .venv activated):

  # Retrieval only: free, instant, needs no API key
  python run_eval.py evaluation/sample_doc_questions.json --tag v1-tfidf

  # Retrieval + answers: calls Gemini once per question
  python run_eval.py evaluation/sample_doc_questions.json --answers --tag v1-tfidf

Metrics:
  retrieval hit : did any of the TOP_K retrieved chunks contain the text
                  from "source_snippet"?
  answer correct: does the answer contain every word/phrase in "answer_keywords"
                  (or, for unanswerable questions, does it say it doesn't know)?
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

from src.loader import load_document
from src.chunking import chunk_text
from src.retriever import Retriever
from src.config import TOP_K

REFUSAL_MARKERS = [
    "don't have enough information",
    "do not have enough information",
    "couldn't find anything relevant",
]


def normalize(text: str) -> str:
    return text.lower().replace("\u2019", "'")


def retrieval_hit(chunks: list[str], snippet: str) -> bool:
    s = normalize(snippet)
    return any(s in normalize(c) for c in chunks)


def answer_correct(answer: str, q: dict) -> bool:
    a = normalize(answer)
    if q.get("expect_refusal"):
        return any(marker in a for marker in REFUSAL_MARKERS)
    return all(normalize(k) in a for k in q.get("answer_keywords", []))


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:5.1f}%" if den else "   n/a"


def summarize(results: list[dict], use_answers: bool) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[r["category"]].append(r)
    groups["ALL"] = results

    summary = {}
    print(f"\n{'category':<14}{'n':>4}{'retrieval hit':>16}", end="")
    print(f"{'answer correct':>18}" if use_answers else "")
    for cat, rows in groups.items():
        with_snip = [r for r in rows if "retrieval_hit" in r]
        hits = sum(r["retrieval_hit"] for r in with_snip)
        entry = {"n": len(rows), "retrieval_hits": hits, "retrieval_total": len(with_snip)}
        line = f"{cat:<14}{len(rows):>4}{pct(hits, len(with_snip)):>16}"
        if use_answers:
            correct = sum(r["answer_correct"] for r in rows)
            entry["answers_correct"] = correct
            line += f"{pct(correct, len(rows)):>18}"
        summary[cat] = entry
        print(line)
    return summary


def run(questions_path: str, use_answers: bool, tag: str, delay: float):
    with open(questions_path, encoding="utf-8") as f:
        spec = json.load(f)

    doc_path = spec["document"]
    chunks = chunk_text(load_document(doc_path))
    retriever = Retriever(chunks)
    print(f"Document: {doc_path} ({len(chunks)} chunks), TOP_K={TOP_K}, "
          f"{len(spec['questions'])} questions")

    client = None
    if use_answers:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: set GEMINI_API_KEY to use --answers.")
            sys.exit(1)
        from google import genai
        from src.llm import build_prompt, generate_answer
        client = genai.Client(api_key=api_key)

    results = []
    for q in spec["questions"]:
        retrieved = retriever.retrieve(q["question"], TOP_K)
        row = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "n_chunks": len(retrieved),
        }
        if q.get("source_snippet"):
            row["retrieval_hit"] = retrieval_hit(retrieved, q["source_snippet"])

        if use_answers:
            if not retrieved:
                # same behaviour as the real bot: no chunks -> no LLM call
                answer = "I couldn't find anything relevant in the document for that question."
            else:
                try:
                    answer = generate_answer(client, build_prompt(q["question"], retrieved)) or ""
                except Exception as e:
                    answer = f"ERROR: {e}"
                time.sleep(delay)  # stay under free-tier rate limits
            row["answer"] = answer
            row["answer_correct"] = (not answer.startswith("ERROR")) and answer_correct(answer, q)

        status = ""
        if "retrieval_hit" in row:
            status += "hit " if row["retrieval_hit"] else "MISS"
        if use_answers:
            status += " ok " if row["answer_correct"] else " BAD"
        print(f"  [{q['id']}] {status:<10} {q['question']}")
        results.append(row)

    summary = summarize(results, use_answers)

    os.makedirs("evaluation/results", exist_ok=True)
    suffix = "answers" if use_answers else "retrieval"
    out_path = f"evaluation/results/{tag}_{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "document": doc_path, "top_k": TOP_K,
                   "n_chunks": len(chunks), "summary": summary, "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\nSaved details to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG bot.")
    parser.add_argument("questions", help="path to a questions JSON file")
    parser.add_argument("--answers", action="store_true", help="also generate and score answers (calls Gemini)")
    parser.add_argument("--tag", default="run", help="label for this run, e.g. v1-tfidf")
    parser.add_argument("--delay", type=float, default=5.0, help="seconds to wait between API calls")
    args = parser.parse_args()
    run(args.questions, args.answers, args.tag, args.delay)