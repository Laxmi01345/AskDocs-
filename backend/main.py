import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Doc Q&A with RAG")

# CORS must be added BEFORE any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include router at module level (not in startup)
from app.api import router
app.include_router(router)


@app.on_event("startup")
def startup():
    from app.database import init_db
    init_db()
    try:
        _rebuild_bm25_indices()
    except Exception:
        pass


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
