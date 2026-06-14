import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from app.database import init_db, load_chunks
from app.bm25_store import build_bm25_retriever

app = FastAPI(title="Doc Q&A with PostgreSQL")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_URL",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _rebuild_bm25_indices():
    chunks_dir = os.getenv("CHUNKS_DIR", "./chunks_store")
    if not os.path.exists(chunks_dir):
        return
    for filename in os.listdir(chunks_dir):
        if filename.endswith(".json"):
            doc_id = filename.replace(".json", "")
            chunks = load_chunks(doc_id)
            if chunks:
                build_bm25_retriever(doc_id, chunks)


@app.on_event("startup")
def startup():
    init_db()
    _rebuild_bm25_indices()

app.include_router(router)