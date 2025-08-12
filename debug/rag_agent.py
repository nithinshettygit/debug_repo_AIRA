# rag_agent.py
import json
import os
from typing import List, Dict, Any, Optional, Callable, Tuple
import math

import numpy as np

# try to import faiss, fall back to sklearn if unavailable
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors

from sentence_transformers import SentenceTransformer

# -------------------------
# Configurable parameters
# -------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast; change if you want better embeddings
CHUNK_SIZE = 500        # characters per chunk (adjust as needed)
CHUNK_OVERLAP = 50      # overlap between chunks
TOP_K = 5               # how many passages to retrieve by default

# -------------------------
# Helpers
# -------------------------
def load_json_kb(path: str) -> Dict[str, Any]:
    """Load the provided JSON knowledgebase."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledgebase file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_kb(kb: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten the JSON structure into a list of {text, metadata}.
    Assumes kb is like {"1 CHAPTER": { "1.1 INTRO": "...", "1.2 ...": "..." }, "2 CHAPTER": {...}}
    """
    docs = []
    for chapter_key, chapter_val in kb.items():
        if isinstance(chapter_val, dict):
            # chapter contains sections
            for sec_key, sec_text in chapter_val.items():
                if not isinstance(sec_text, str):
                    continue
                text = sec_text.strip()
                if not text:
                    continue
                metadata = {"chapter": chapter_key, "section": sec_key}
                docs.append({"text": text, "meta": metadata})
        elif isinstance(chapter_val, str):
            text = chapter_val.strip()
            if text:
                docs.append({"text": text, "meta": {"chapter": chapter_key, "section": None}})
    return docs

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split long text into overlapping chunks by characters (simple, robust)."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

# -------------------------
# RAG Index class
# -------------------------
class SimpleRAG:
    def __init__(self,
                 kb_path: str,
                 embedding_model_name: str = EMBEDDING_MODEL_NAME,
                 use_faiss: bool = True):
        self.kb_path = kb_path
        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        self.docs: List[Dict[str, Any]] = []   # list of dicts: {text, meta, chunk_id}
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.use_faiss = use_faiss and _HAS_FAISS
        self._build_index()

    def _build_index(self):
        kb = load_json_kb(self.kb_path)
        flat = flatten_kb(kb)
        expanded = []
        chunk_id = 0
        for d in flat:
            chunks = chunk_text(d["text"])
            for c in chunks:
                expanded.append({
                    "text": c,
                    "meta": d["meta"],
                    "chunk_id": f"chunk_{chunk_id}"
                })
                chunk_id += 1
        self.docs = expanded

        texts = [d["text"] for d in self.docs]
        if len(texts) == 0:
            raise ValueError("No textual content found in the knowledgebase.")

        # compute embeddings
        emb = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # normalize (helps with cosine similarity via dot-product)
        emb = emb.astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms

        self.embeddings = emb

        # build index
        if self.use_faiss:
            dim = emb.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors ≈ cosine
            self.index.add(emb)
        else:
            # sklearn fallback: use NearestNeighbors (cosine)
            self.index = NearestNeighbors(n_neighbors=min(10, len(texts)), metric="cosine")
            self.index.fit(emb)

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
        """
        Return list of (doc, score) pairs.
        Score is cosine similarity (higher better).
        """
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

        if self.use_faiss:
            D, I = self.index.search(q_emb, top_k)
            scores = D[0].tolist()
            idxs = I[0].tolist()
            results = []
            for idx, score in zip(idxs, scores):
                if idx < 0 or idx >= len(self.docs):
                    continue
                results.append((self.docs[idx], float(score)))
            return results
        else:
            dist, idxs = self.index.kneighbors(q_emb, n_neighbors=min(top_k, len(self.docs)))
            results = []
            for d, idx in zip(dist[0], idxs[0]):
                score = 1.0 - float(d)   # sklearn cosine distance -> similarity approx
                results.append((self.docs[idx], score))
            return results

    def build_prompt(self, query: str, retrieved: List[Tuple[Dict[str, Any], float]]) -> str:
        """
        Assemble a RAG prompt: instructions + retrieved passages (with metadata).
        The prompt is designed to force the LLM to use only the provided passages.
        """
        header = (
            "You are a helpful tutor. Answer the user's question using ONLY the passages "
            "provided below. Cite the passage (chapter/section) when relevant. "
            "If the information is not present in the provided passages, say \"I don't know\".\n\n"
        )
        context_blocks = []
        for i, (doc, score) in enumerate(retrieved, start=1):
            meta = doc.get("meta", {})
            chap = meta.get("chapter", "unknown chapter")
            sec = meta.get("section", "unknown section")
            block = f"=== Passage {i} (score={score:.4f}) | {chap} :: {sec} ===\n{doc['text'].strip()}\n"
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)
        footer = (
            "\n\nUser Question:\n" + query.strip() + "\n\n"
            "Answer concisely and only using the passages above. If you must speculate, say so explicitly."
        )
        return header + context + footer

# -------------------------
# Public function
# -------------------------
def rag_answer(query: str,
               kb_path: str,
               llm_fn: Optional[Callable[[str], str]] = None,
               top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Run a RAG retrieval and (optionally) call an LLM to answer.
    - query: user query
    - kb_path: path to knowledgebase.json
    - llm_fn: optional function that accepts a prompt string and returns model's text
              If None, the function returns the assembled prompt and retrieved passages.
    - top_k: number of retrieved passages
    Returns a dict with keys:
      - 'retrieved': list of {text, meta, score}
      - 'prompt': assembled prompt string
      - 'answer': the LLM answer string if llm_fn provided, else None
    """
    rag = SimpleRAG(kb_path)
    retrieved = rag.retrieve(query, top_k=top_k)
    prompt = rag.build_prompt(query, retrieved)

    result = {
        "retrieved": [
            {"text": d["text"], "meta": d["meta"], "score": score}
            for (d, score) in retrieved
        ],
        "prompt": prompt,
        "answer": None
    }

    if llm_fn:
        # call user's LLM function
        answer = llm_fn(prompt)
        result["answer"] = answer

    return result

# -------------------------
# Example usage (commented)
# -------------------------
if __name__ == "__main__":
    KB_PATH = "knowledgebase.json"   # change if needed
    sample_q = "What is photosynthesis and where does it occur in plants?"

    # Example 1: just build prompt + inspect retrieved docs
    out = rag_answer(sample_q, KB_PATH, llm_fn=None, top_k=4)
    print("=== Retrieved Passages ===")
    for i, r in enumerate(out["retrieved"], start=1):
        print(f"[{i}] {r['meta']} (score={r['score']:.3f})")
        print(r['text'][:300].replace("\n", " ") + ("..." if len(r['text'])>300 else ""))
        print("-"*80)
    print("\n=== Assembled Prompt (preview) ===")
    print(out["prompt"][:2000])  # preview only

    # Example 2: if you have an LLM wrapper function, pass it like this:
    # def my_llm(prompt_text: str) -> str:
    #     # Use your configured Gemini or other LLM here.
    #     # Example with your gemini model from config.py:
    #     # from config import model, extract_text
    #     # resp = model.generate_content(prompt_text)
    #     # return extract_text(resp)
    #     raise NotImplementedError("Implement your LLM call here.")
    #
    # out2 = rag_answer(sample_q, KB_PATH, llm_fn=my_llm, top_k=4)
    # print("LLM answer:", out2['answer'])
