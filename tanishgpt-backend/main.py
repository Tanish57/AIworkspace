import os
import requests
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# -------------------------
#  CONFIG
# -------------------------
LLAMA_SERVER = "http://127.0.0.1:8080/v1/chat/completions"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "../tanish_memory_db"

USER_NAME = "Tanish Solanki"
USER_EMAIL = "solanki.tanish57@gmail.com"

# -------------------------
#  INIT FASTAPI
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
#  INIT CHROMADB
# -------------------------
client = chromadb.Client(Settings(
    persist_directory=CHROMA_PATH,
    anonymized_telemetry=False
))

collection = client.get_or_create_collection("tanish_memory")

embedder = SentenceTransformer(EMBED_MODEL)

# -------------------------
#   MEMORY HELPERS
# -------------------------
def embed(text):
    return embedder.encode([text]).tolist()

def add_memory(text):
    emb = embed(text)
    current = collection.get()
    mem_id = f"mem_{len(current.get('ids', [])) + 1}"

    collection.add(
        ids=[mem_id],
        embeddings=emb,
        documents=[text],
        metadatas=[{"source": "memory"}]
    )

def recall(query, n=5):
    emb = embed(query)
    res = collection.query(query_embeddings=emb, n_results=n)
    return res.get("documents", [[]])[0]

# -------------------------
#   LLaMA CALLER
# -------------------------
def ask_llama(history, memory_block):
    """Call llama.cpp chat-completion endpoint with memory injected"""

    # prepend memory block as a system message
    final_messages = [
        {
            "role": "system",
            "content": f"""
You are TanishGPT, a helpful assistant.

User name: {USER_NAME}
User email: {USER_EMAIL}

NEVER hallucinate; if unsure say "I don't know".

Relevant memory:
{memory_block}
"""}
    ] + history

    payload = {
        "model": "tanish-local",
        "messages": final_messages,
        "temperature": 0.2,
        "max_tokens": 400,
        "stream": False
    }

    r = requests.post(LLAMA_SERVER, json=payload)
    data = r.json()

    # llama.cpp response format
    return data["choices"][0]["message"]["content"]

# -------------------------
#    API ROUTES
# -------------------------
class ChatInput(BaseModel):
    session_id: str
    message: str
    history: list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(input: ChatInput):
    user_msg = input.message

    # 1. Retrieve relevant memories
    memories = recall(user_msg)
    memory_text = "\n".join(memories) if memories else "None."

    # 2. Convert frontend history → llama.cpp format
    formatted_history = []
    for msg in input.history:
        formatted_history.append({
            "role": "user" if msg["sender"] == "user" else "assistant",
            "content": msg["text"]
        })

    # 3. Add the new user message
    formatted_history.append({
        "role": "user",
        "content": user_msg
    })

    # 4. Query llama.cpp
    reply = ask_llama(formatted_history, memory_text)

    # 5. Save memory
    add_memory(f"User: {user_msg}")
    add_memory(f"AI: {reply}")

    return {"reply": reply}
