import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List

import requests
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from session_store import (
    create_session,
    list_sessions,
    get_session,
    touch_session,
    delete_session_metadata
)

# New Modules
from document_processor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, chunk_with_metadata
from graph_rag import GraphBuilder, GraphRetriever

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
LLAMA_SERVER = "http://127.0.0.1:8080/v1/chat/completions"
CHROMA_PATH = "../tanish_memory_db"
DATA_DIR = Path("./data")
GRAPH_DIR = Path("./graphs")
EMBED_MODEL = "all-MiniLM-L6-v2"

DATA_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

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
# Document collection will be separate or shared? Let's use a shared one for simplicity for now, 
# or per-user. Since this is single user, we use one "documents" collection.
doc_collection = client.get_or_create_collection("documents")

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

def recall_documents(query: str, n=3):
    """Retrieve relevant chunks from uploaded documents."""
    res = doc_collection.query(query_texts=[query], n_results=n)
    return res.get("documents", [[]])[0]

def format_memory(mem_list):
    if not mem_list:
        return "None."
    formatted_lines = []
    for m in mem_list:
        line = m.strip()
        if not line:
            continue
        if line.lower().startswith("user:") or line.lower().startswith("ai:"):
            formatted_lines.append(f"- {line}")
        else:
            formatted_lines.append(f"- {line}")
    return "\n".join(formatted_lines)

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
# BACKGROUND TASKS
# -------------------------------------------------
def process_document_background(file_path: Path, doc_id: str):
    """Extracts text, chunks it, indexes in Vector DB, and builds Graph."""
    print(f"Processing {file_path}...")
    
    # 1. Extract
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        text = extract_text_from_txt(file_path)
        
    # 2. Chunk
    chunks_data = chunk_with_metadata(text)
    texts = [c["text"] for c in chunks_data]
    metadatas = [c["metadata"] for c in chunks_data]
    
    # Add doc_id to metadata
    for m in metadatas:
        m["doc_id"] = doc_id
        
    # 3. Vector Index
    ids = [f"{doc_id}_{i}" for i in range(len(texts))]
    doc_collection.add(
        ids=ids,
        documents=texts,
        embeddings=[embed(t) for t in texts],
        metadatas=metadatas
    )
    print(f"Indexed {len(texts)} chunks for {doc_id}")

    # 4. Graph Build
    graph_path = GRAPH_DIR / "knowledge_graph.json"
    builder = GraphBuilder(graph_path)
    builder.build_graph(texts) # This uses LLM, might take time
    print(f"Graph updated for {doc_id}")

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------
class ChatReq(BaseModel):
    session_id: Optional[str] = None
    message: str
    top_n: int = 5
    deep_search: bool = False # New flag

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
@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    file_path = DATA_DIR / f"{doc_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(process_document_background, file_path, doc_id)
    
    return {"status": "queued", "doc_id": doc_id, "message": "Document is being processed in background."}

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
    res = session_collection.get(where={"session_id": session_id})
    ids = res.get("ids", [])
    if ids:
        session_collection.delete(ids=ids)
    return {"status": "deleted", "session_id": session_id}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    if req.session_id:
        session_id = req.session_id
    else:
        title = req.message.strip()
        if len(title) > 50:
            title = title[:47] + "..."
        new_sess = create_session(title=title)
        session_id = new_sess["id"]

    touch_session(session_id)

    # 1. Recall Session & Global Memory
    session_memories = recall_session(session_id, req.message, n=8)
    global_memories = recall_global(req.message, n=5)
    
    # 2. Recall Documents (Vector Search)
    doc_memories = recall_documents(req.message, n=5)
    
    # 3. Graph Search (Deep Search)
    graph_context = ""
    if req.deep_search:
        graph_path = GRAPH_DIR / "knowledge_graph.json"
        retriever = GraphRetriever(graph_path)
        graph_context = retriever.get_relevant_subgraph_text(req.message)

    # Format blocks
    session_block = format_memory(session_memories)
    global_block = format_memory(global_memories)
    doc_block = format_memory(doc_memories)

    system_prompt = f"""
You are TanishGPT, a memory-augmented personal assistant for Tanish Solanki.

You DO have memory of previous facts, preferences, and personal details shared with you.
You also have access to a Knowledge Base of uploaded documents.

### LONG-TERM (global) MEMORY
{global_block}

### CONVERSATION (session) MEMORY
{session_block}

### DOCUMENT KNOWLEDGE BASE
{doc_block}
"""

    if req.deep_search and graph_context:
        system_prompt += f"\n### DEEP SEARCH (GRAPH) CONTEXT\n{graph_context}\n"

    assistant_primer = "Note: Use the memory and knowledge base above when answering. If a fact is present, use it directly."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": assistant_primer},
        {"role": "user", "content": req.message}
    ]

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