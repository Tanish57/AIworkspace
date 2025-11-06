# tanish_rag_memory.py

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
import os

# --- Setup persistent storage ---
CHROMA_PATH = "./tanish_memory_db"
os.makedirs(CHROMA_PATH, exist_ok=True)

# Initialize Chroma client
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create or get the collection
collection = client.get_or_create_collection(name="tanish_memory")

# --- Embedding model for text similarity ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Add user identity (default memory) ---
user_info = {
    "name": "Tanish Solanki",
    "email": "tanish.solanki@example.com"
}

# Store user identity if not already stored
existing = collection.get(ids=["user_identity"])
if not existing["ids"]:
    collection.add(
        ids=["user_identity"],
        documents=[json.dumps(user_info)],
        metadatas=[{"type": "identity"}]
    )

print("✅ Memory system initialized for:", user_info["name"])
print("📍Database stored at:", os.path.abspath(CHROMA_PATH))


# --- Add memory ---
def remember(key, value):
    """Store a new fact or update existing memory."""
    docs = [f"{key}: {value}"]
    ids = [key]
    collection.upsert(ids=ids, documents=docs)
    print(f"🧠 Remembered: {key} -> {value}")


# --- Recall memory ---
def recall(query, n_results=2):
    """Search memory for related info."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0] if results["documents"] else []


# --- Example Usage ---
if __name__ == "__main__":
    remember("favourite_language", "Python")
    print("🔍 Recall test:", recall("What is my favourite language?"))
