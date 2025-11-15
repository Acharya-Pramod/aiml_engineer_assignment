import requests
import pandas as pd

BASE_URL = "https://november7-730026606190.europe-west1.run.app/messages/"

def fetch_messages(skip=0, limit=100):
    resp = requests.get(BASE_URL, params={"skip": skip, "limit": limit})
    resp.raise_for_status()
    return resp.json()

data = fetch_messages()
df = pd.DataFrame(data["items"])
df.head()

!pip install sentence-transformers fastapi uvicorn nest_asyncio pyngrok


from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

messages = df["message"].tolist()
embeddings = model.encode(messages, convert_to_numpy=True)


from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(payload: Query):
    q = payload.question
    q_emb = model.encode([q])[0]

    # compute cosine similarity
    sims = embeddings @ q_emb / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    idx = np.argmax(sims)

    best_message = messages[idx]

    return {"answer": best_message}


import os
import re
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query

# Base URL for the public messages API.
# Override via env var if needed.
MEMBER_API_BASE = os.getenv(
    "MEMBER_API_BASE",
    "https://november7-730026606190.europe-west1.run.app",
)

MESSAGES_ENDPOINT = f"{MEMBER_API_BASE}/messages"

# Very small stopword list for scoring
STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on",
    "for", "with", "at", "is", "are", "was", "were", "do",
    "does", "did", "be", "have", "has", "had", "how", "what",
    "when", "where", "who", "why", "which",
}

app = FastAPI(
    title="Member QA Service",
    description="Simple question-answering over member messages.",
    version="0.1.0",
)


def tokenize(text: str) -> List[str]:
    """Lowercase, remove non-word chars, split, drop stopwords."""
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def extract_message_text(msg: Dict[str, Any]) -> str:
    """
    Try a few common field names to get the actual message text.
    Adjust this if your schema differs.
    """
    for key in ("text", "message", "content", "body"):
        if key in msg and isinstance(msg[key], str):
            return msg[key]
    # Fallback: stringify whole object (not ideal, but robust)
    return str(msg)


def normalize_messages_payload(raw: Any) -> List[Dict[str, Any]]:
    """
    Handle either:
      - {"messages": [...]} or
      - [...]
    """
    if isinstance(raw, dict) and "messages" in raw:
        messages = raw["messages"]
    else:
        messages = raw

    if not isinstance(messages, list):
        raise ValueError("Unexpected /messages API format: not a list")

    return [m for m in messages if isinstance(m, dict)]


def score_question_vs_message(question_tokens: List[str], msg_tokens: List[str]) -> float:
    """
    Very simple relevance score: Jaccard-like overlap.
    """
    if not msg_tokens:
        return 0.0
    q_set = set(question_tokens)
    m_set = set(msg_tokens)
    intersection = q_set & m_set
    union = q_set | m_set
    if not union:
        return 0.0
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

    # You can tune this threshold; below it we say "I don't know"
    if best_score < 0.05:
        return None

    return best_msg


@app.get("/ask")
async def ask(question: str = Query(..., min_length=3)) -> Dict[str, str]:
    """
    Example:
      GET /ask?question=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F

    Response:
      { "answer": "Layla is planning her trip to London on 2025-04-12." }
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(MESSAGES_ENDPOINT)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Error calling member API: {e}",
            )

    try:
        raw = resp.json()
        messages = normalize_messages_payload(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse /messages response: {e}",
        )

    best_msg = find_best_message(question, messages)

    if not best_msg:
        return {"answer": "I could not find an answer to that question in the member messages."}

    answer_text = extract_message_text(best_msg)
    return {"answer": answer_text}

from fastapi import FastAPI
import requests
import re
import spacy

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

API_URL = "https://november7-730026606190.europe-west1.run.app/messages"

# Extract person names from question
def extract_name(question):
    doc = nlp(question)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

# Detect intent of the question
def detect_intent(question):
    q = question.lower()
    if "when" in q:
        return "date"
    if "how many" in q or "number of" in q:
        return "count"
    if "favorite" in q or "favourite" in q:
        return "favorite"
    if "trip" in q or "travel" in q:
        return "travel"
    return "general"

# Fetch all messages from API
def fetch_all_messages():
    messages = []
    skip = 0
    limit = 100
    while True:
        res = requests.get(API_URL, params={"skip": skip, "limit": limit}).json()
        items = res.get("items", [])
        messages.extend(items)
        if len(items) < limit:
            break
        skip += limit
    return messages

# Main API endpoint
@app.get("/ask")
def ask(question: str):
    name = extract_name(question)
    if not name:
        return {"answer": "I could not identify the member name in the question."}

    intent = detect_intent(question)
    messages = fetch_all_messages()

    # Filter messages by user name
    user_msgs = [m["message"] for m in messages if m["user_name"].lower().startswith(name.lower())]

    if not user_msgs:
        return {"answer": f"No messages found for {name}."}

    text = " ".join(user_msgs)

    # Extract answer based on intent
    if intent == "date":
        date = re.search(r"(\b\w+ \d{1,2}\b|\d{1,2}/\d{1,2}/\d{2,4})", text)
        if date:
            return {"answer": f"{name} mentioned: {date.group(0)}"}
    
    if intent == "count":
        count = re.search(r"\b\d+\b", text)
        if count:
            return {"answer": f"{name} has {count.group(0)}."}
    
    if intent == "favorite":
        m = re.search(r"(favorite.*?\.|favourite.*?\.|I love .*?\.|my favorite.*?\.)", text, re.I)
        if m:
            return {"answer": m.group(0)}
    
    if intent == "travel":
        m = re.search(r"(trip.*?\.|travel.*?\.|going to .*?\. )", text, re.I)
        if m:
            return {"answer": m.group(0)}

    # Fallback
    return {"answer": text[:300] + "..."}
