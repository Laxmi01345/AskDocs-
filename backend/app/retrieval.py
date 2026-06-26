from fastapi import HTTPException
from app.database import get_vectorstore
from app.reranker import get_cross_encoder
from app.llm import CerebrasLLM
from app.context_builder import build_rag_prompt
from app.session import Session
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


def _retrieve_and_rerank(doc_id: str, question: str, top_k: int = 3):
    vectordb = get_vectorstore(doc_id)
    initial_k = top_k * 4
    docs = vectordb.similarity_search(question, k=initial_k)

    cross_encoder = get_cross_encoder()
    if cross_encoder and len(docs) > top_k:
        pairs = [[question, doc.page_content] for doc in docs]
        scores = cross_encoder.score(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in ranked[:top_k]]

    return docs[:top_k]


def answer_with_rag(doc_id: str, question: str, top_k: int = 3):
    retrieved_docs = _retrieve_and_rerank(doc_id, question, top_k)

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer the question using ONLY the context below.\n"
            "If the answer is not present in the context, say \"I don't know\".\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}"
        ),
    )

    llm = CerebrasLLM()
    chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    answer = chain.invoke({"context": context, "question": question})

    retrieved_chunks = []
    for rank, doc in enumerate(retrieved_docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        retrieved_chunks.append({
            "rank": rank,
            "chunk_id": metadata.get("chunk_index", rank - 1),
            "text": doc.page_content,
            "source": "vector",
        })

    return answer, context, retrieved_chunks


def answer_with_rag_with_history(doc_id: str, question: str, top_k: int, session: Session):
    retrieved_docs = _retrieve_and_rerank(doc_id, question, top_k)

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    if session and session.turn_count > 0:
        full_prompt = build_rag_prompt(session, question, context)
    else:
        full_prompt = (
            "Answer the question using ONLY the context below.\n"
            "If the answer is not present in the context, say \"I don't know\".\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )

    llm = CerebrasLLM()
    answer = llm.invoke(full_prompt)

    retrieved_chunks = []
    for rank, doc in enumerate(retrieved_docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        retrieved_chunks.append({
            "rank": rank,
            "chunk_id": metadata.get("chunk_index", rank - 1),
            "text": doc.page_content,
        })

    return answer, context, retrieved_chunks
