import os
import json
from typing import List
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

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
    metadatas = [{"chunk_index": i, "filename": filename} for i in range(len(chunks))]
    vectorstore.add_texts(texts=chunks, metadatas=metadatas)
    vectorstore.persist()

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    chunks_path = Path(CHUNKS_DIR) / f"{doc_id}.json"
    with open(chunks_path, "w") as f:
        json.dump({"filename": filename, "chunks": chunks}, f)
