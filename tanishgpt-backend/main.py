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
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

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
from mcp_servers.web_search import web_search

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
client = chromadb.PersistentClient(path=CHROMA_PATH)
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

def save_message(session_id: str, role: str, text: str, **kwargs):
    turn_idx = next_turn_index(session_id)
    
    metadata = {
        "session_id": session_id,
        "role": role,
        "turn_index": turn_idx,
        "ts": datetime.now(timezone.utc).isoformat()
    }
    metadata.update(kwargs)

    session_collection.add(
        ids=[f"{session_id}_{role}_{turn_idx}"],
        documents=[text],
        embeddings=[embed(text)],
        metadatas=[metadata]
    )

def recall_session(session_id: str, query: str, n=5):
    res = session_collection.query(query_texts=[query], n_results=n, where={"session_id": session_id})
    return res.get("documents", [[]])[0]

def recall_documents(query: str, n=5):
    """Retrieve relevant chunks using Advanced Retrieval Pipeline."""
    results, confidence = retrieval_pipeline.search(query, n_final=n)
    
    # Extract docs and metas
    docs = [r["text"] for r in results]
    metas = [r["metadata"] for r in results]
    
    return list(zip(docs, metas)), confidence

def format_memory(mem_list):
    if not mem_list:
        return "None."
    formatted_lines = []
    for item in mem_list:
        # Handle tuple (doc, meta) from recall_documents
        if isinstance(item, tuple):
            text, meta = item
            citation = ""
            if meta:
                page = meta.get("page_label", "?")
                chapter = meta.get("chapter_title", "")
                citation = f" [Page: {page}, Chapter: {chapter}]"
            formatted_lines.append(f"-{citation} {text.strip()}")
        # Handle simple string (session/global memory)
        elif isinstance(item, str):
            line = item.strip()
            if not line:
                continue
            formatted_lines.append(f"- {line}")
    return "\n".join(formatted_lines)

# -------------------------------------------------
# WEB SEARCH HELPERS
# -------------------------------------------------
FRESHNESS_KEYWORDS = [
    "latest", "today", "recent", "current",
    "news", "update", "announcement",
    "this week", "this month"
]

WEB_CACHE = {}

def build_news_query(user_query: str) -> str:
    """Rewrite generic queries for better news results."""
    uq = user_query.lower()
    if "ai" in uq and "news" in uq:
        return "artificial intelligence latest technology news"
    return user_query

def filter_relevant_results(results, original_query):
    """Filter out irrelevant results based on simple keyword matching if needed."""
    # For now, we trust DDG News relative to the rewritten query, but we ensure we have snippets.
    filtered = [r for r in results if r.get('snippet')]
    return filtered

def cached_web_search(query: str):
    # 1. Expand Query
    search_query = build_news_query(query)
    
    # Check cache with original query to avoid re-searching for same user intent
    if query in WEB_CACHE:
        print(f"[Web Search] Returning cached results for: {query}")
        return WEB_CACHE[query]
    
    print(f"[Web Search] Searching for: {search_query} (Original: {query})")
    raw_results = web_search(search_query)
    
    # 2. Filter
    filtered_results = filter_relevant_results(raw_results, query)
    
    WEB_CACHE[query] = filtered_results
    return filtered_results

def needs_web_search(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in FRESHNESS_KEYWORDS)

def build_web_prompt(query: str, search_results: list) -> str:
    results_text = ""
    for i, res in enumerate(search_results):
        results_text += f"{i+1}. Title: {res.get('title')}\n   Snippet: {res.get('snippet')}\n   URL: {res.get('url')}\n\n"

    return f"""You are TanishGPT. You are given recent web search results.

Use ONLY the information from the results below.
Do not add facts that are not present.
If the information is insufficient, say so clearly.

User question:
{query}

Web search results:
{results_text}

Summarize the key points concisely. Include sources (URLs) at the end.
"""

def call_llama(messages):
    payload = {
        "model": "tanish-local",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 8098,
        "stream": False
    }
    r = requests.post(LLAMA_SERVER, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

import hashlib

def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# -------------------------------------------------
# RETRIEVAL PIPELINE
# -------------------------------------------------
class RetrievalPipeline:
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder
        print("Loading Reranker Model (Cross-Encoder)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Reranker Loaded.")

    def search(self, query: str, n_final: int = 5):
        # 1. Hybrid Recall (Top 40 -> BM25 -> Top 20)
        candidates = self.hybrid_recall(query, n_initial=40, n_top=20)
        print(f"[Retrieval] Hybrid Recall found {len(candidates)} candidates.")
        
        if not candidates:
            return [], 0.0
        
        # 2. Rerank (Top 20 -> Reranker)
        reranked = self.rerank(query, candidates)
        
        # 3. Metadata Boost
        final_results = self.apply_metadata_boost(reranked, query)
        
        # 4. Confidence Score
        confidence = self.calculate_confidence(final_results)
        print(f"[Retrieval] Final Confidence: {confidence:.4f}")
        for i, res in enumerate(final_results[:n_final]):
             print(f"  Rank {i+1}: {res['final_score']:.4f} | {res['metadata'].get('source', '?')}")
        
        return final_results[:n_final], confidence

    def hybrid_recall(self, query: str, n_initial=40, n_top=20):
        # Vector Search
        res = self.collection.query(
            query_texts=[query], 
            n_results=n_initial,
            include=["documents", "metadatas"]
        )
        
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        
        if not docs:
            return []

        # BM25 Rescoring
        tokenized_corpus = [doc.split(" ") for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        
        candidates = []
        for i, (doc, meta, id) in enumerate(zip(docs, metas, ids)):
            # Normalize Vector Rank (Simple 1/rank approximation)
            vector_score = 1.0 / (i + 1)
            
            # Normalize BM25
            bm25_norm = bm25_scores[i] / max_bm25
            
            # Hybrid Score
            hybrid_score = (0.7 * vector_score) + (0.3 * bm25_norm)
            
            candidates.append({
                "id": id,
                "text": doc,
                "metadata": meta,
                "hybrid_score": hybrid_score
            })
            
        # Select Top N
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return candidates[:n_top]

    def rerank(self, query, candidates):
        if not candidates:
            return []
            
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        
        # Apply Sigmoid to get 0-1 range
        scores = 1 / (1 + np.exp(-scores))
        
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
            
        return candidates

    def apply_metadata_boost(self, candidates, query):
        for c in candidates:
            score = c["rerank_score"]
            meta = c["metadata"]
            
            # Boost Code
            if "def " in c["text"] or "class " in c["text"] or "import " in c["text"]:
                 score *= 1.1
            
            # Boost Chapter Match
            chapter_title = meta.get("chapter_title", "")
            if chapter_title and isinstance(chapter_title, str) and chapter_title.lower() in query.lower():
                score *= 1.15
            
            c["final_score"] = min(score, 1.0)
            
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates

    def calculate_confidence(self, candidates):
        if not candidates:
            return 0.0
        
        scores = [c["final_score"] for c in candidates]
        top_1 = scores[0]
        top_2 = scores[1] if len(scores) > 1 else 0.0
        
        top_5 = scores[:5]
        avg_top_5 = sum(top_5) / len(top_5)
        
        confidence = (0.5 * top_1) + (0.3 * (top_1 - top_2)) + (0.2 * avg_top_5)
        return confidence

# Initialize Pipeline
retrieval_pipeline = RetrievalPipeline(doc_collection, embedder)

# -------------------------------------------------
# BACKGROUND TASKS
# -------------------------------------------------
def process_document_background(file_path: str, doc_id: str, file_hash: str):
    print(f"Processing document: {file_path} (ID: {doc_id})")
    
    # 1. Extract Text & Chunk
    # document_processor returns a list of dictionaries (TextSegment objects)
    # The original document_processor functions are extract_text_from_pdf, etc.
    # Assuming a unified `document_processor.process_document` now exists.
    # If not, the logic needs to be adapted to call the correct extractor based on file extension.
    
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        text, page_map = extract_text_from_pdf(Path(file_path))
    elif ext == ".docx":
        text, page_map = extract_text_from_docx(Path(file_path))
    else: # .txt or other
        text, page_map = extract_text_from_txt(Path(file_path))
        
    chunks = chunk_with_metadata(text, page_map)

    if not chunks:
        print(f"No text extracted from {file_path}")
        return

    print(f"Extracted {len(chunks)} chunks. Generating embeddings...")
    
    # 2. Generate Embeddings (BATCH PROCESSING)
    texts = [chunk["text"] for chunk in chunks]
    
    # OPTIMIZATION: Use batch encoding instead of loop
    # This runs all embeddings in parallel (or vectorized)
    embeddings = embedder.encode(texts).tolist()
    
    # 3. Store in ChromaDB
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        meta = {
            "doc_id": doc_id,
            "source": str(file_path),
            "file_hash": file_hash, # Store hash for duplicate detection
            "page_label": chunk["metadata"].get("page_label", "N/A"), # Use page_label from original chunk_with_metadata
            "chapter_title": chunk["metadata"].get("chapter_title", "N/A"), # Use chapter_title from original chunk_with_metadata
            "start_char": chunk["metadata"].get("start_char", 0),
            "end_char": chunk["metadata"].get("end_char", 0),
            "paragraph_index": chunk["metadata"].get("paragraph_index", i) # Fallback if not present
        }
        metadatas.append(meta)
        
    doc_collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Stored {len(chunks)} chunks in Vector DB.")
    
    # 4. Build Knowledge Graph (Async/Background)
    # We pass the full text for graph extraction
    full_text = "\n".join(texts)
    print("Building Knowledge Graph...")
    graph_builder.build_graph_from_text(full_text, doc_id)
    print(f"Knowledge Graph built for {doc_id}.")

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
    filename = file.filename
    # 1. Generate unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
    file_path = DATA_DIR / unique_filename
    
    # Save file temporarily to calculate hash
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    # Calculate Hash
    file_hash = calculate_file_hash(str(file_path))
    
    # Check for duplicates in ChromaDB
    # We query for any chunk that has this file_hash
    existing_docs = doc_collection.get(where={"file_hash": file_hash}, limit=1)
    
    if existing_docs and existing_docs['ids']:
        print(f"Duplicate file detected: {filename} (Hash: {file_hash})")
        
        # Clean up duplicate file from disk
        file_path.unlink(missing_ok=True) # No error if missing
        
        # If duplicate, we can return the existing doc_id if we had a way to map hash -> doc_id easily.
        # The metadata of the first chunk will have 'doc_id'.
        existing_metadata = existing_docs['metadatas'][0]
        doc_id = existing_metadata.get('doc_id')
        
        return {
            "status": "success", 
            "message": f"File '{filename}' already exists (Duplicate Content). Skipping processing.",
            "doc_id": doc_id
        }

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    
    # Add background task to process the file
    background_tasks.add_task(process_document_background, str(file_path), doc_id, file_hash)
    
    return {"status": "success", "message": f"File '{filename}' uploaded successfully. Processing started.", "doc_id": doc_id}

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

# -------------------------------------------------
# MCP CLIENT (TOOL MANAGER)
# -------------------------------------------------
import subprocess
import json
import threading

class ToolManager:
    def __init__(self):
        self.server_process = None
        self.lock = threading.Lock()
        self.start_server()

    def start_server(self):
        """Starts the filesystem MCP server as a subprocess."""
        server_path = Path(__file__).parent / "mcp_servers" / "filesystem.py"
        self.server_process = subprocess.Popen(
            ["python3", str(server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        print(f"MCP Server started at {server_path}")

    def call_tool(self, tool_name: str, args: dict) -> str:
        """Calls a tool on the MCP server via JSON-RPC."""
        with self.lock:
            if self.server_process.poll() is not None:
                print("MCP Server died, restarting...")
                self.start_server()

            req_id = str(uuid.uuid4())
            request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": args},
                "id": req_id
            }
            
            try:
                self.server_process.stdin.write(json.dumps(request) + "\n")
                self.server_process.stdin.flush()
                
                response_line = self.server_process.stdout.readline()
                if not response_line:
                    return "Error: No response from MCP server."
                
                response = json.loads(response_line)
                if "error" in response:
                    return f"Tool Error: {response['error']['message']}"
                
                # Extract text content from result
                content_list = response.get("result", {}).get("content", [])
                text_content = "".join([c["text"] for c in content_list if c["type"] == "text"])
                return text_content
            except Exception as e:
                return f"RPC Error: {str(e)}"

    def get_tool_definitions(self) -> str:
        """Returns the tool definitions for the system prompt."""
        # Hardcoded for now, but ideally fetched via tools/list
        return """
- write_file(relative_path: str, content: str): Writes code or text to a file in the workspace.
  Example: write_file("hello.py", "print('Hello World')")
"""

tool_manager = ToolManager()

# -------------------------------------------------
# ENDPOINTS
# -------------------------------------------------
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

    # 1. Freshness Check / Web Search Branch
    if needs_web_search(req.message):
        print(f"[Web Search] Detected freshness intent for: {req.message}")
        try:
            results = cached_web_search(req.message)
            
            # Check if results are sufficient (at least 2 valid results)
            if results and len(results) >= 2:
                print(f"[Web Search] Found {len(results)} valid results.")
                
                # Build Prompt
                web_prompt = build_web_prompt(req.message, results)
                
                # Call LLM directly
                messages = [{"role": "user", "content": web_prompt}]
                reply = call_llama(messages)
                
                # Save & Return
                save_message(session_id, "user", req.message)
                save_message(session_id, "assistant", reply, source="web")
                return ChatResp(session_id=session_id, reply=reply)
            
            else:
                print("[Web Search] Insufficient results found. Returning polite fallback.")
                return ChatResp(
                    session_id=session_id, 
                    reply="I couldn't find reliable recent news from web search at the moment. This may be due to search source limitations. Try rephrasing your query or asking about a specific topic."
                )
                
        except Exception as e:
            print(f"[Web Search] Error during search: {e}. Falling back.")
                
        except Exception as e:
            print(f"[Web Search] Error during search: {e}. Falling back.")

    # 2. Recall Memories
    session_memories = recall_session(session_id, req.message, n=8)
    global_memories = recall_global(req.message, n=5)
    doc_memories, confidence = recall_documents(req.message, n=5)
    
    # 2. Graph Context
    graph_context = ""
    if req.deep_search:
        graph_path = GRAPH_DIR / "knowledge_graph.json"
        retriever = GraphRetriever(graph_path)
        graph_context = retriever.get_relevant_subgraph_text(req.message)

    # 3. Format Context
    session_block = format_memory(session_memories)
    global_block = format_memory(global_memories)
    doc_block = format_memory(doc_memories)
    
    # 4. Construct System Prompt with Tools
    tools_block = tool_manager.get_tool_definitions()
    
    confidence_warning = ""
    if confidence < 0.5 and doc_memories:
         confidence_warning = "\nSYSTEM: The retrieved context has LOW CONFIDENCE. It may be incomplete or weak. If the answer cannot be determined directly from the sources, say you are unsure instead of guessing.\n"
    
    system_prompt = f"""
You are TanishGPT, a memory-augmented personal assistant.

### TOOLS AVAILABLE
You have access to the following tools. 
- To use a tool, you MUST format your response as JSON: {{ "tool": "tool_name", "arguments": {{ ... }} }}
- To reply to the user normally, just write PLAIN TEXT. DO NOT use JSON.

{tools_block}

### MEMORY & KNOWLEDGE
Global: {global_block}
Session: {session_block}
Documents: {doc_block}
{confidence_warning}

INSTRUCTIONS:
1. If the user EXPLICITLY asks to "create", "save", or "write" a file, use the `write_file` tool (JSON).
   - Example 1: "Create a python script named hello.py" -> {{ "tool": "write_file", ... }}
   - Example 2: "How do I install homebrew?" -> [PLAIN TEXT] (Do NOT use tool)
2. If answering from documents, cite [Page: X, Chapter: Y].
3. For all other responses, answer normally in PLAIN TEXT. Do NOT invent new tools like "confirm".
"""

    if req.deep_search and graph_context:
        system_prompt += f"\n### DEEP SEARCH CONTEXT\n{graph_context}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.message}
    ]

    # 5. LLM Call Loop (Simple 1-turn tool use)
    reply = call_llama(messages)
    
    # Check for Tool Call (Robust parsing)
    # We look for a JSON object structure: { "tool": ... }
    import re
    json_match = re.search(r"\{.*\}", reply, re.DOTALL)
    
    if json_match:
        try:
            potential_json = json_match.group(0)
            # Try to parse
            tool_call = json.loads(potential_json)
            
            if "tool" in tool_call and "arguments" in tool_call:
                tool_name = tool_call.get("tool")
                args = tool_call.get("arguments", {})
                
                print(f"Executing Tool: {tool_name} with {args}")
                tool_result = tool_manager.call_tool(tool_name, args)
                
                # Feed result back to LLM
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Tool Output: {tool_result}\nNow confirm to the user."})
                reply = call_llama(messages)
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            
            # Attempt to repair JSON (LLM often forgets closing braces)
            repaired = False
            for append_str in ["}", "}}"]:
                try:
                    tool_call = json.loads(potential_json + append_str)
                    if "tool" in tool_call and "arguments" in tool_call:
                        tool_name = tool_call.get("tool")
                        args = tool_call.get("arguments", {})
                        
                        print(f"Executing Tool (Repaired): {tool_name} with {args}")
                        tool_result = tool_manager.call_tool(tool_name, args)
                        
                        messages.append({"role": "assistant", "content": reply})
                        messages.append({"role": "user", "content": f"Tool Output: {tool_result}\nNow confirm to the user."})
                        reply = call_llama(messages)
                        repaired = True
                        break
                except json.JSONDecodeError:
                    continue
            
            if not repaired:
                print("Failed to repair JSON.")
                pass

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

@app.get("/")
def root():
    return {"status": "ok", "message": "TanishGPT Backend is running. Go to /docs for API documentation."}