import os
import json
import pickle
from pathlib import Path
from typing import List, Optional
from rank_bm25 import BM25Okapi

BM25_DIR = os.getenv("BM25_CACHE_DIR", "./bm25_cache")


def _ensure_dir():
    os.makedirs(BM25_DIR, exist_ok=True)


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_bm25_index(chunks: List[str]) -> BM25Okapi:
    tokenized_chunks = [_tokenize(chunk) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def save_bm25_index(doc_id: str, bm25: BM25Okapi, chunks: List[str]):
    _ensure_dir()
    path = Path(BM25_DIR) / f"{doc_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)


def load_bm25_index(doc_id: str) -> Optional[dict]:
    path = Path(BM25_DIR) / f"{doc_id}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def bm25_search(query: str, bm25: BM25Okapi, chunks: List[str], top_k: int = 10) -> List[dict]:
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked_indices[:top_k]:
        if scores[idx] > 0:
            results.append({
                "text": chunks[idx],
                "score": float(scores[idx]),
                "index": idx,
            })
    return results
