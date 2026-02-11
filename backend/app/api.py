from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import uuid
from app.chunking import chunk_text
from app.embeddings import embed_text, embed_chunks
from app.llm import answer_question
from app.parsing import parse_txt, parse_pdf, parse_docx
from app.database import store_document, retrieve_chunks

router = APIRouter()


class AskRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 3


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    data = await file.read()
    ext = (file.filename or "").lower().split(".")[-1]

    if ext == "txt":
        text = parse_txt(data)
    elif ext == "pdf":
        text = parse_pdf(data)
    elif ext == "docx":
        text = parse_docx(data)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Generate doc_id
    doc_id = str(uuid.uuid4())

    # Chunk and embed
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    # Store in DB
    store_document(doc_id, file.filename, chunks, embeddings)

    return {"doc_id": doc_id, "filename": file.filename, "chunks": len(chunks)}


@router.post("/ask")
def ask(req: AskRequest):
    # Embed question
    q_emb = embed_text(req.question).tolist()

    # Retrieve top-k chunks
    results = retrieve_chunks(req.doc_id, q_emb, top_k=req.top_k)

    if not results:
        raise HTTPException(status_code=404, detail="Document not found")

    # Build context
    context = "\n\n".join([r["content"] for r in results])

    # Get answer
    answer = answer_question(context, req.question)

    return {"answer": answer, "context": context, "similarity_scores": results}