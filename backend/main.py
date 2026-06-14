import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Doc Q&A with RAG")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_URL",
        "*",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    from app.database import init_db
    init_db()
    # Lazy BM25 rebuild - only if chunks exist
    try:
        _rebuild_bm25_indices()
    except Exception:
        pass
    # Load router after startup so imports don't block port binding
    from app.api import router
    app.include_router(router)


def _rebuild_bm25_indices():
    from app.database import load_chunks
    from app.bm25_store import build_bm25_retriever
    chunks_dir = os.getenv("CHUNKS_DIR", "./chunks_store")
    if not os.path.exists(chunks_dir):
        return
    for filename in os.listdir(chunks_dir):
        if filename.endswith(".json"):
            doc_id = filename.replace(".json", "")
            chunks = load_chunks(doc_id)
            if chunks:
                build_bm25_retriever(doc_id, chunks)
