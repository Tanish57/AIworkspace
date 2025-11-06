import os
import json
import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ───────────────────────────────────────────────
#  1️⃣  Configuration
# ───────────────────────────────────────────────
USER_NAME = "Tanish Solanki"
USER_EMAIL = "solanki.tanish57@gmail.com"
CHROMA_PATH = "tanish_memory_db"
LLAMA_SERVER = "http://127.0.0.1:8080/completion"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ───────────────────────────────────────────────
#  2️⃣  Initialize ChromaDB + Embedder
# ───────────────────────────────────────────────
client = chromadb.Client(Settings(
    persist_directory=CHROMA_PATH,
    anonymized_telemetry=False
))
collection = client.get_or_create_collection("tanish_memory")
embedder = SentenceTransformer(EMBED_MODEL)

print(f"✅ Memory system ready for {USER_NAME}")
print(f"📍 DB Path: {os.path.abspath(CHROMA_PATH)}\n")

# ───────────────────────────────────────────────
#  3️⃣  Memory helpers
# ───────────────────────────────────────────────
def add_memory(text: str, metadata: dict | None = None):
    """Store text + optional metadata in Chroma (safe, no duplicates)."""
    existing = collection.query(query_texts=[text], n_results=1)
    if existing.get("documents") and text in existing["documents"][0]:
        return  # Skip duplicates

    emb = embedder.encode([text]).tolist()
    count = len(existing.get('ids', []))
    mem_id = f"mem_{count + 1}"

    # Ensure metadata is non-empty
    safe_metadata = metadata if metadata else {"source": "memory"}

    collection.add(
        ids=[mem_id],
        embeddings=emb,
        documents=[text],
        metadatas=[safe_metadata]
    )
    print(f"🧠 Saved memory: {text[:60]}")



def recall_memory(query: str, n_results: int = 5):
    """
    Retrieve the top-n most relevant memories for the query.
    Includes relevance scoring for clarity.
    """
    emb = embedder.encode([query]).tolist()
    res = collection.query(query_embeddings=emb, n_results=n_results)
    
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # Convert distances to simple relevance scores
    pairs = list(zip(docs, dists))
    pairs = [p for p in pairs if p[0] is not None]
    pairs.sort(key=lambda x: x[1])  # smaller distance = more relevant

    top_memories = [f"- {p[0]}" for p in pairs[:n_results]]
    return "\n".join(top_memories) if top_memories else "(no relevant memories found)"


# ───────────────────────────────────────────────
#  4️⃣  LLaMA server helper
# ───────────────────────────────────────────────
def ask_llama(prompt: str):
    """Send a prompt to your local llama-server with tuned parameters."""
    data = {
        "prompt": prompt,
        "n_predict": min(8192, 512 + len(prompt) // 3),  # Increased for more complete answers
        "temperature": 0.15,  # Lowered for factual accuracy
        "top_p": 0.9,
        "repeat_penalty": 1.15,
        "stop": ["</s>", "User:", "Me:", "🧑 You:"],  # stop when switching roles
    }
    for attempt in range(3):
        try:
            response = requests.post(LLAMA_SERVER, json=data, timeout=30)
            if response.ok:
                content = response.json().get("content", "")
                # clean up hallucinated logs if model drifts
                if "User:" in content and "Me:" in content:
                    content = content.split("Me:")[-1].strip()
                return content.strip() if content else "(no response received from model)"
            else:
                print(f"⚠️ Attempt {attempt+1}: LLaMA error {response.status_code}")
        except Exception as e:
            if attempt ==2:
                return f"⚠️ Connection error after 3 retries: {e}"


# ───────────────────────────────────────────────
#  5️⃣  Conversation loop
# ───────────────────────────────────────────────
def chat():
    print(f"🤖  Tanish GPT connected to local LLaMA at 8080")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        # ① recall related memories
        memories = recall_memory(user_input)
        memory_context = memories if memories else "None."

        full_prompt = f"""
You are TanishGPT — a local, factual AI assistant with memory, running for Tanish Solanki.

Known facts (do not contradict these):
• User name: {USER_NAME}
• User email: {USER_EMAIL}
• You never invent details. If you don't know, say so plainly.
• Speak as a conversational assistant — no role labels or meta commentary.

Here are relevant memories from past chats (treat them as trusted context):
{memory_context}

Now the user says: "{user_input}"

Reply as if continuing a natural, memory-aware conversation with Tanish.
Be concise, warm, and factual — no instructions or explanations, just the answer.
"""
        print(f"Injected memories:\n{memory_context}\n")
        # ③ send to local LLaMA
        answer = ask_llama(full_prompt)
        print(f"🤖 Tanish GPT: {answer}\n")

        # ④ persist conversation
        add_memory(f"User: {user_input}")
        add_memory(f"AI: {answer}")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    chat()
