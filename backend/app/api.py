from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from app.chunking import chunk_text
from app.parsing import load_document
from app.database import store_document
from app.utils.logger import log_rag_interaction
from app.retrieval import answer_with_rag
from app.session import session_manager
from app.context_builder import should_summarize, build_summarization_prompt

router = APIRouter()


class AskRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 3
    session_id: Optional[str] = None


@router.get("/documents")
def list_documents():
    """List all uploaded documents with their chunk counts."""
    import json
    from pathlib import Path
    chunks_dir = os.getenv("CHUNKS_DIR", "./chunks_store")
    documents = []
    if os.path.exists(chunks_dir):
        for f in os.listdir(chunks_dir):
            if f.endswith(".json"):
                doc_id = f.replace(".json", "")
                with open(Path(chunks_dir) / f) as fh:
                    data = json.load(fh)
                documents.append({
                    "doc_id": doc_id,
                    "filename": data.get("filename", "unknown"),
                    "chunks": len(data.get("chunks", [])),
                })
    return {"documents": documents}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    data = await file.read()
    ext = (file.filename or "").lower().split(".")[-1]

    if ext not in {"txt", "pdf", "docx"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    text = load_document(data, file.filename or "document")

    doc_id = str(uuid.uuid4())

    chunks = chunk_text(text)
    store_document(doc_id, file.filename or "document", chunks)

    return {"doc_id": doc_id, "filename": file.filename, "chunks": len(chunks)}


@router.post("/ask")
def ask(req: AskRequest):
    from app.retrieval import answer_with_rag_with_history

    # 1. Get or create session
    if req.session_id:
        session = session_manager.get_session(req.session_id)
        if session is None or session.doc_id != req.doc_id:
            session_id = session_manager.create_session(req.doc_id)
        else:
            session_id = req.session_id
    else:
        session_id = session_manager.create_session(req.doc_id)

    # 2. Summarize if needed
    session = session_manager.get_session(session_id)
    if session and session.turn_count > 5 and should_summarize(session):
        from app.llm import CerebrasLLM
        summary_prompt = build_summarization_prompt(session)
        summary_llm = CerebrasLLM(max_tokens=200)
        summary = summary_llm.invoke(summary_prompt)
        session_manager.set_summary(session_id, summary)
        session = session_manager.get_session(session_id)
        if session and len(session.turns) > 8:
            session.turns = session.turns[-5:]

    # 3. Retrieve and answer with history
    try:
        session = session_manager.get_session(session_id)
        answer, _context, retrieved_chunks = answer_with_rag_with_history(
            req.doc_id, req.question, req.top_k, session
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Save turn
    session_manager.add_turn(session_id, req.question, answer)

    return {
        "doc_id": req.doc_id,
        "question": req.question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "session_id": session_id,
    }
