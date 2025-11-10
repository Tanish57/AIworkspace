import os
import requests
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Optional, List
from datetime import datetime, timezone

from session_store import (
    create_session,
    list_sessions,
    get_session,
    touch_session,
    delete_session_metadata
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
LLAMA_SERVER = "http://127.0.0.1:8080/v1/chat/completions"
CHROMA_PATH = "../tanish_memory_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# -------------------------------------------------
# FASTAPI SETUP
# -------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# CHROMA COLLECTIONS
# -------------------------------------------------
client = chromadb.Client(Settings(
    persist_directory=CHROMA_PATH,
    anonymized_telemetry=False
))
session_collection = client.get_or_create_collection("session_messages")
global_memory = client.get_or_create_collection("global_memory")
embedder = SentenceTransformer(EMBED_MODEL)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def embed(text: str):
    """Return a single embedding vector."""
    return embedder.encode([text]).tolist()[0]

def next_turn_index(session_id: str):
    results = session_collection.get(where={"session_id": session_id}, include=["metadatas"])
    if not results["metadatas"]:
        return 0
    return max(meta["turn_index"] for meta in results["metadatas"]) + 1

def should_store_memory(user_msg: str, ai_reply: str) -> bool:
    IMPORTANT_KEYWORDS = [
        "my name is", "i live in", "i am from", "my birthday", "my age", "i study",
        "i work as", "my job", "my email", "my phone", "my preference", "i like",
        "my goal", "i prefer", "remember this", "save this", "note this",
        "i want you to remember", "from now on", "let's assume", "you should know"
    ]
    text = (user_msg + " " + ai_reply).lower()
    for key in IMPORTANT_KEYWORDS:
        if key in text:
            return True
    if len(user_msg) < 5:
        return False
    BAD_PATTERNS = ["hi", "hello", "thanks", "thank you", "ok", "cool", "lol"]
    if any(p == user_msg.lower().strip() for p in BAD_PATTERNS):
        return False
    return False

def recall_global(query, n=3):
    res = global_memory.query(query_texts=[query], n_results=n)
    return res.get("documents", [[]])[0]

def save_message(session_id: str, role: str, text: str):
    turn_idx = next_turn_index(session_id)
    session_collection.add(
        ids=[f"{session_id}_{role}_{turn_idx}"],
        documents=[text],
        embeddings=[embed(text)],
        metadatas=[{
            "session_id": session_id,
            "role": role,
            "turn_index": turn_idx,
            "ts": datetime.now(timezone.utc).isoformat()
        }]
    )

def recall_session(session_id: str, query: str, n=5):
    res = session_collection.query(query_texts=[query], n_results=n, where={"session_id": session_id})
    return res.get("documents", [[]])[0]

def call_llama(messages):
    payload = {
        "model": "tanish-local",
        "messages": messages,
        "temperature": 0.2,
        "stream": False
    }
    r = requests.post(LLAMA_SERVER, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------
class ChatReq(BaseModel):
    session_id: Optional[str] = None
    message: str
    top_n: int = 5

class ChatResp(BaseModel):
    session_id: str
    reply: str

class SessionInfo(BaseModel):
    id: str
    title: str
    created_at: int
    last_active: int

# -------------------------------------------------
# ENDPOINTS
# -------------------------------------------------
@app.post("/sessions/new", response_model=SessionInfo)
def new_session():
    return create_session()

@app.get("/sessions", response_model=List[SessionInfo])
def sessions():
    return list_sessions()

@app.get("/sessions/{session_id}", response_model=SessionInfo)
def session_detail(session_id: str):
    sess = get_session(session_id)
    if not sess:
        return {"error": "Session not found"}
    return sess

@app.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    res = session_collection.get(where={"session_id": session_id}, include=["documents", "metadatas"])
    if not res["documents"] or len(res["documents"]) == 0:
        return []
    items = list(zip(res["documents"], res["metadatas"]))
    items.sort(key=lambda x: x[1]["turn_index"])
    messages = [{
        "role": meta["role"],
        "content": doc,
        "turn_index": meta["turn_index"],
        "ts": meta["ts"]
    } for doc, meta in items]
    return messages

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    delete_session_metadata(session_id)
    res = session_collection.get(where={"session_id": session_id}, include=["ids"])
    ids = res.get("ids", [])
    if ids:
        session_collection.delete(ids=ids)
    return {"status": "deleted", "session_id": session_id}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    if req.session_id:
        session_id = req.session_id
    else:
        new_sess = create_session()
        session_id = new_sess["id"]

    touch_session(session_id)

    session_memories = recall_session(session_id, req.message, n=req.top_n)
    session_block = "\n".join(session_memories)
    global_memories = recall_global(req.message, n=3)
    global_block = "\n".join(global_memories)

    messages = [{
        "role": "system",
        "content": f"Relevant global memory:\n{global_block}\n\nRelevant session memory:\n{session_block}"
    }, {"role": "user", "content": req.message}]

    reply = call_llama(messages)

    save_message(session_id, "user", req.message)
    save_message(session_id, "assistant", reply)

    if should_store_memory(req.message, reply):
        combined = req.message + " " + reply
        global_memory.add(
            ids=[f"gm_{os.urandom(4).hex()}"],
            documents=[f"User: {req.message}\nAI: {reply}"],
            embeddings=[embed(combined)],
            metadatas=[{"source": "global_memory"}]
        )

    return ChatResp(session_id=session_id, reply=reply)

@app.get("/health")
def health():
    return {"status": "ok"}
