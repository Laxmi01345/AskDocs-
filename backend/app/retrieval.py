from fastapi import HTTPException
from app.database import get_vectorstore, load_chunks
from app.bm25_store import get_bm25_retriever, build_bm25_retriever
from app.reranker import get_cross_encoder
from app.llm import CerebrasLLM
from app.context_builder import build_rag_prompt
from app.session import Session
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


def _format_score(distance):
    if distance is None:
        return None
    return round(1 / (1 + float(distance)), 4)


def _rrf_fusion(bm25_results, vector_results, k=60):
    """Reciprocal Rank Fusion: merge two ranked lists."""
    scores = {}
    for rank, doc in enumerate(bm25_results, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
    for rank, doc in enumerate(vector_results, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank)

    doc_map = {}
    for doc in bm25_results + vector_results:
        doc_map[doc.page_content] = doc

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[k] for k in sorted_keys]


def _build_hybrid_retriever(doc_id: str, top_k: int = 3):
    vectordb = get_vectorstore(doc_id)

    bm25_retriever = get_bm25_retriever(doc_id)
    if bm25_retriever is None:
        chunks = load_chunks(doc_id)
        if chunks:
            bm25_retriever = build_bm25_retriever(doc_id, chunks, k=top_k * 3)
        else:
            return vectordb.as_retriever(search_kwargs={"k": top_k})

    bm25_retriever.k = top_k * 3

    def hybrid_retrieve(query):
        bm25_docs = bm25_retriever.invoke(query)
        vector_docs = vectordb.similarity_search(query, k=top_k * 3)
        fused = _rrf_fusion(bm25_docs, vector_docs)
        return fused[:top_k]

    return hybrid_retrieve


def _rerank(query, docs, top_k):
    cross_encoder = get_cross_encoder()
    if cross_encoder is None:
        return docs[:top_k]
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.score(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]


def answer_with_rag(doc_id: str, question: str, top_k: int = 3):
    try:
        vectordb = get_vectorstore(doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    hybrid_retriever = _build_hybrid_retriever(doc_id, top_k)
    retrieved_docs = hybrid_retriever(question)

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    retrieved_docs = _rerank(question, retrieved_docs, top_k)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    context = format_docs(retrieved_docs)

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
            "source": metadata.get("source", "hybrid"),
        })

    return answer, context, retrieved_chunks


def answer_with_rag_with_history(doc_id: str, question: str, top_k: int, session: Session):
    vectordb = get_vectorstore(doc_id)
    hybrid_retriever = _build_hybrid_retriever(doc_id, top_k)
    retrieved_docs = hybrid_retriever(question)

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    retrieved_docs = _rerank(question, retrieved_docs, top_k)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    context = format_docs(retrieved_docs)

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
