# app/main.py
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query

# Base URL for the public messages API (override via env var if needed)
MEMBER_API_BASE = os.getenv(
    "MEMBER_API_BASE",
    "https://november7-730026606190.europe-west1.run.app",
)

MESSAGES_ENDPOINT = f"{MEMBER_API_BASE}/messages"

# Minimal stopword set to remove very common tokens for scoring
STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on",
    "for", "with", "at", "is", "are", "was", "were", "do",
    "does", "did", "be", "have", "has", "had", "how", "what",
    "when", "where", "who", "why", "which", "i", "you", "my",
    "me", "we", "they", "it", "this", "that", "your", "our",
}

app = FastAPI(
    title="Member QA Service",
    description="Simple question-answering over member messages.",
    version="0.1.0",
)


def tokenize(text: str) -> List[str]:
    """
    Lowercase, keep words, remove stopwords.
    """
    if not text:
        return []
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def extract_message_text(msg: Dict[str, Any]) -> str:
    """
    Try common message fields; fallback to stringifying the dict.
    """
    for key in ("message", "text", "content", "body"):
        if key in msg and isinstance(msg[key], str):
            return msg[key].strip()
    # Some messages include nested payloads
    if "data" in msg and isinstance(msg["data"], dict):
        for key in ("message", "text", "content", "body"):
            if key in msg["data"] and isinstance(msg["data"][key], str):
                return msg["data"][key].strip()
    # Fallback
    return str(msg)


def normalize_messages_payload(raw: Any) -> List[Dict[str, Any]]:
    """
    Normalize possible /messages responses:
      - { "items": [...] }  (Swagger shows items)
      - { "messages": [...] }
      - [...]  (list directly)
    """
    if isinstance(raw, dict):
        if "items" in raw and isinstance(raw["items"], list):
            messages = raw["items"]
        elif "messages" in raw and isinstance(raw["messages"], list):
            messages = raw["messages"]
        else:
            # If dict but not expected keys, try to find a list value
            # pick the first list we find
            for v in raw.values():
                if isinstance(v, list):
                    messages = v
                    break
            else:
                raise ValueError("Unexpected /messages API format (no list found)")
    elif isinstance(raw, list):
        messages = raw
    else:
        raise ValueError("Unexpected /messages API format")

    # Ensure elements are dicts
    messages = [m for m in messages if isinstance(m, dict)]
    return messages


def score_question_vs_message(question_tokens: List[str], msg_tokens: List[str]) -> float:
    """
    Very simple relevance score: Jaccard overlap.
    """
    if not question_tokens or not msg_tokens:
        return 0.0
    q_set = set(question_tokens)
    m_set = set(msg_tokens)
    intersection = q_set & m_set
    union = q_set | m_set
    return len(intersection) / len(union)


def find_best_message(question: str, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q_tokens = tokenize(question)
    if not q_tokens:
        return None

    best_msg = None
    best_score = 0.0

    for msg in messages:
        text = extract_message_text(msg)
        tokens = tokenize(text)
        score = score_question_vs_message(q_tokens, tokens)
        if score > best_score:
            best_score = score
            best_msg = msg

    # Threshold: requires some overlap; tune if necessary
    if best_score < 0.05:
        return None

    return best_msg


@app.get("/ask")
async def ask(question: str = Query(..., min_length=3)) -> Dict[str, str]:
    """
    GET /ask?question=...
    Returns:
      { "answer": "..." }
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(MESSAGES_ENDPOINT)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Error calling member API: {str(e)}",
            )

    try:
        raw = resp.json()
        messages = normalize_messages_payload(raw)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse /messages response: {e}",
        )

    best_msg = find_best_message(question, messages)

    if not best_msg:
        return {"answer": "I could not find an answer to that question in the member messages."}

    answer_text = extract_message_text(best_msg)
    return {"answer": answer_text}
