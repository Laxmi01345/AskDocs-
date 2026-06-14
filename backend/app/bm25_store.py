import pickle
import os
from pathlib import Path
from typing import Dict, List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

BM25_CACHE_DIR = os.getenv("BM25_CACHE_DIR", "./bm25_cache")
_cache: Dict[str, BM25Retriever] = {}


def _ensure_cache_dir():
    os.makedirs(BM25_CACHE_DIR, exist_ok=True)


def build_bm25_retriever(doc_id: str, chunks: List[str], k: int = 10):
    _ensure_cache_dir()
    documents = [
        Document(page_content=chunk, metadata={"chunk_index": i, "doc_id": doc_id})
        for i, chunk in enumerate(chunks)
    ]
    retriever = BM25Retriever.from_documents(documents, k=k)
    _cache[doc_id] = retriever
    cache_path = Path(BM25_CACHE_DIR) / f"{doc_id}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(retriever, f)
    return retriever


def get_bm25_retriever(doc_id: str):
    if doc_id in _cache:
        return _cache[doc_id]
    cache_path = Path(BM25_CACHE_DIR) / f"{doc_id}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            retriever = pickle.load(f)
        _cache[doc_id] = retriever
        return retriever
    return None
