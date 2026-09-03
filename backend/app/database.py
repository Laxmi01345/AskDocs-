import os
import json
from typing import List
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

from app.embeddings import get_embeddings

load_dotenv()


# Config
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "./chunks_store")


def _ensure_persist_dir():
    os.makedirs(PERSIST_DIR, exist_ok=True)


def init_db():
    """Initialize local storage for Chroma-backed document data."""
    _ensure_persist_dir()
    os.makedirs(CHUNKS_DIR, exist_ok=True)


def _vectorstore(doc_id: str):
    _ensure_persist_dir()
    return Chroma(
        collection_name=f"doc_{doc_id}",
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def get_vectorstore(doc_id: str):
    return _vectorstore(doc_id)


def store_document(doc_id: str, filename: str, chunks: List[str] | None = None, embeddings=None):
    """Store document chunks in a LangChain Chroma vector store."""
    if not chunks:
        return

    vectorstore = _vectorstore(doc_id)
    metadatas = [{"chunk_index": i, "filename": filename, "method": "simple"} for i in range(len(chunks))]
    vectorstore.add_texts(texts=chunks, metadatas=metadatas)

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    chunks_path = Path(CHUNKS_DIR) / f"{doc_id}.json"
    with open(chunks_path, "w") as f:
        json.dump({"filename": filename, "chunks": chunks}, f)

    from app.bm25_store import build_bm25_index, save_bm25_index
    bm25 = build_bm25_index(chunks)
    save_bm25_index(doc_id, bm25, chunks)


def store_semantic_chunks(doc_id: str, filename: str, chunks: List[str]):
    """Store semantic chunks in a separate collection."""
    if not chunks:
        return

    vectorstore = Chroma(
        collection_name=f"doc_{doc_id}_semantic",
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )
    metadatas = [{"chunk_index": i, "filename": filename, "method": "semantic"} for i in range(len(chunks))]
    vectorstore.add_texts(texts=chunks, metadatas=metadatas)

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    chunks_path = Path(CHUNKS_DIR) / f"{doc_id}_semantic.json"
    with open(chunks_path, "w") as f:
        json.dump({"filename": filename, "chunks": chunks}, f)


def get_semantic_vectorstore(doc_id: str):
    """Get the semantic chunk vector store."""
    return Chroma(
        collection_name=f"doc_{doc_id}_semantic",
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )
