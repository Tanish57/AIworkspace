import os
import json
import re
import time
import networkx as nx
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
from filelock import FileLock
from itertools import combinations

# -----------------------
# CONFIG
# -----------------------
LLAMA_SERVER = "http://127.0.0.1:8080/v1/chat/completions"

def call_llm_json(messages: List[Dict[str, str]]) -> Any:
    """Helper to call local LLM and expect JSON response."""
    payload = {
        "model": "tanish-local",
        "messages": messages,
        "temperature": 0.1,
        "stream": False,
        "response_format": {"type": "json_object"} 
    }
    try:
        r = requests.post(LLAMA_SERVER, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        
        # Robust JSON extraction using regex
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            # Fallback: try to parse the whole content if regex fails
            return json.loads(content)
            
    except Exception as e:
        print(f"LLM JSON Error: {e}")
        return {}

# -----------------------
# GRAPH BUILDER
# -----------------------
class GraphBuilder:
    def __init__(self, graph_path: Path):
        self.graph_path = graph_path
        # We do NOT load the graph here to avoid stale data.
        # We will load it only when we are ready to write (under lock).

    def extract_entities_relations(self, text: str) -> List[Dict[str, str]]:
        """Extracts entities and relations from a text chunk."""
        prompt = f"""
        Analyze the following text and extract key entities (Concepts, People, Places, Events) and their relationships.
        Return a JSON object with a key "triples" containing a list of objects with "subject", "predicate", "object".
        
        Text:
        {text[:2000]}
        
        JSON Format:
        {{
            "triples": [
                {{"subject": "Entity1", "predicate": "relation", "object": "Entity2"}}
            ]
        }}
        """
        messages = [{"role": "user", "content": prompt}]
        response = call_llm_json(messages)
        return response.get("triples", [])

    def build_graph(self, chunks: List[str]):
        """
        1. Process chunks to extract triples (Slow, Parallelizable).
        2. Acquire Lock.
        3. Load Global Graph.
        4. Merge Triples.
        5. Save Global Graph.
        6. Release Lock.
        """
        print(f"Extracting knowledge from {len(chunks)} chunks...")
        all_triples = []
        
        # 1. Extraction Phase (No Lock needed)
        for i, chunk in enumerate(chunks):
            time.sleep(0.1) # Rate limit
            triples = self.extract_entities_relations(chunk)
            all_triples.extend(triples)
            
            if i % 5 == 0:
                print(f"Processed {i+1}/{len(chunks)} chunks")

        # 2. Merge Phase (Critical Section)
        lock_path = self.graph_path.with_suffix(".lock")
        print(f"Acquiring lock on {lock_path}...")
        
        with FileLock(lock_path):
            # Load latest state
            if self.graph_path.exists():
                try:
                    graph = nx.node_link_graph(json.loads(self.graph_path.read_text()))
                except Exception as e:
                    print(f"Error loading graph, starting fresh: {e}")
                    graph = nx.Graph()
            else:
                graph = nx.Graph()

            # Merge new triples
            print(f"Merging {len(all_triples)} triples into graph...")
            for t in all_triples:
                subj = t.get("subject", "").strip().lower()
                obj = t.get("object", "").strip().lower()
                pred = t.get("predicate", "").strip().lower()
                
                if subj and obj and pred:
                    graph.add_node(subj)
                    graph.add_node(obj)
                    
                    if graph.has_edge(subj, obj):
                        existing_rels = graph[subj][obj].get("relations", [])
                        if pred not in existing_rels:
                            existing_rels.append(pred)
                            graph[subj][obj]["relations"] = existing_rels
                    else:
                        graph.add_edge(subj, obj, relations=[pred])

            # Save
            data = nx.node_link_data(graph)
            self.graph_path.write_text(json.dumps(data, indent=2))
            print(f"Graph saved to {self.graph_path}")

# -----------------------
# GRAPH RETRIEVER
# -----------------------
class GraphRetriever:
    def __init__(self, graph_path: Path):
        self.graph_path = graph_path
        if self.graph_path.exists():
            self.graph = nx.node_link_graph(json.loads(self.graph_path.read_text()))
        else:
            self.graph = nx.Graph()

    def get_relevant_subgraph_text(self, query: str, depth=1) -> str:
        """
        Advanced Graph RAG:
        1. Extract entities.
        2. Multi-hop Inference: Find shortest paths between entities.
        3. Path Scoring: Use PageRank to find most important neighbors.
        4. Return rich context.
        """
        # 1. Extract Entities
        prompt = f"""
        Extract the main entities (keywords) from this query.
        Return a JSON object with a key "entities" containing a list of strings.
        
        Query: {query}
        
        JSON Format:
        {{
            "entities": ["entity1", "entity2"]
        }}
        """
        messages = [{"role": "user", "content": prompt}]
        resp = call_llm_json(messages)
        query_entities = resp.get("entities", [])
        
        query_entities = [e.lower().strip() for e in query_entities]
        if not query_entities:
            query_entities = [w.lower() for w in query.split() if len(w) > 3]

        found_nodes = [n for n in query_entities if n in self.graph.nodes]
        if not found_nodes:
            return ""

        context_lines = []
        subgraph_nodes = set(found_nodes)

        # 2. Multi-hop Inference (Shortest Paths)
        # If we have multiple entities, try to find how they are connected
        if len(found_nodes) > 1:
            context_lines.append("--- Multi-hop Connections ---")
            pairs = list(combinations(found_nodes, 2))
            for u, v in pairs:
                try:
                    # Limit path length to avoid irrelevant long chains
                    path = nx.shortest_path(self.graph, source=u, target=v)
                    if len(path) <= 4: # Only close connections
                        # Format path: A --[rel]--> B --[rel]--> C
                        path_str = path[0]
                        for i in range(len(path)-1):
                            n1, n2 = path[i], path[i+1]
                            rels = self.graph[n1][n2].get("relations", ["related"])
                            rel_str = "|".join(rels)
                            path_str += f" --[{rel_str}]--> {n2}"
                        context_lines.append(path_str)
                        subgraph_nodes.update(path)
                except nx.NetworkXNoPath:
                    continue

        # 3. Expand Neighborhood & PageRank
        # Get neighbors up to depth
        for node in found_nodes:
            neighbors = nx.single_source_shortest_path_length(self.graph, node, cutoff=depth)
            subgraph_nodes.update(neighbors.keys())

        # Create induced subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # Calculate PageRank on this subgraph to find "VIP" nodes
        # We personalize it to bias towards our query entities
        personalization = {n: 1.0 for n in found_nodes if n in subgraph}
        # Distribute remaining weight evenly or just 0
        try:
            scores = nx.pagerank(subgraph, personalization=personalization if personalization else None)
            # Get top 10 most important nodes
            top_nodes = sorted(scores, key=scores.get, reverse=True)[:15]
        except:
            # Fallback if pagerank fails (e.g. too small)
            top_nodes = list(subgraph.nodes)[:15]

        # 4. Format Context from Top Nodes
        context_lines.append("\n--- Key Concepts & Relations ---")
        final_subgraph = subgraph.subgraph(top_nodes)
        
        for u, v, data in final_subgraph.edges(data=True):
            rels = ", ".join(data.get("relations", []))
            context_lines.append(f"{u} --[{rels}]--> {v}")
            
        return "\n".join(context_lines)
