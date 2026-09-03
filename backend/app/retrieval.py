from fastapi import HTTPException
from app.database import get_vectorstore, get_semantic_vectorstore
from app.reranker import get_cross_encoder
from app.llm import CerebrasLLM
from app.context_builder import build_rag_prompt
from app.session import Session
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


def retrieve_simple(doc_id: str, question: str, top_k: int = 3):
    vectordb = get_vectorstore(doc_id)
    docs = vectordb.similarity_search(question, k=top_k)
    return docs


def retrieve_semantic(doc_id: str, question: str, top_k: int = 3):
    vectordb = get_semantic_vectorstore(doc_id)
    docs = vectordb.similarity_search(question, k=top_k)
    return docs


def retrieve_hybrid(doc_id: str, question: str, top_k: int = 3):
    import numpy as np
    from app.bm25_store import load_bm25_index, bm25_search

    vectordb = get_vectorstore(doc_id)
    dense_docs = vectordb.similarity_search(question, k=top_k * 4)

    bm25_data = load_bm25_index(doc_id)
    bm25_results = []
    if bm25_data:
        bm25_results = bm25_search(question, bm25_data["bm25"], bm25_data["chunks"], top_k=top_k * 4)

    fused_scores = {}
    k = 60

    for rank, doc in enumerate(dense_docs):
        text = doc.page_content
        if text not in fused_scores:
            fused_scores[text] = {"text": text, "score": 0, "doc": doc}
        fused_scores[text]["score"] += 1.0 / (k + rank + 1)

    for rank, result in enumerate(bm25_results):
        text = result["text"]
        if text not in fused_scores:
            fused_scores[text] = {"text": text, "score": 0, "doc": None}
        fused_scores[text]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

    docs = []
    for item in ranked[:top_k]:
        if item["doc"] is not None:
            docs.append(item["doc"])
        else:
            from langchain_core.documents import Document
            docs.append(Document(page_content=item["text"], metadata={"source": "bm25"}))

    return docs


def retrieve_reranked(doc_id: str, question: str, top_k: int = 3):
    # Step 1: Get Semantic search results with similarity scores
    vectordb = get_semantic_vectorstore(doc_id)
    initial_k = top_k * 4
    
    # Get docs with similarity scores
    results = vectordb.similarity_search_with_relevance_scores(question, k=initial_k)
    
    if not results:
        return []
    
    docs = [doc for doc, _ in results]
    vector_scores = [score for _, score in results]

    # Step 2: Rerank with cross-encoder, but combine with vector scores
    cross_encoder = get_cross_encoder()
    if cross_encoder and len(docs) > top_k:
        pairs = [[question, doc.page_content] for doc in docs]
        ce_scores = cross_encoder.score(pairs)
        
        # Normalize cross-encoder scores to 0-1 range
        import numpy as np
        ce_arr = np.array(ce_scores)
        if ce_arr.max() != ce_arr.min():
            ce_normalized = (ce_arr - ce_arr.min()) / (ce_arr.max() - ce_arr.min())
        else:
            ce_normalized = np.ones_like(ce_arr) * 0.5
        
        # Normalize vector scores to 0-1 range
        vec_arr = np.array(vector_scores)
        if vec_arr.max() != vec_arr.min():
            vec_normalized = (vec_arr - vec_arr.min()) / (vec_arr.max() - vec_arr.min())
        else:
            vec_normalized = np.ones_like(vec_arr) * 0.5
        
        # Combine scores: 60% cross-encoder + 40% vector similarity
        combined_scores = 0.6 * ce_normalized + 0.4 * vec_normalized
        
        ranked = sorted(zip(docs, combined_scores), key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in ranked[:top_k]]

    return docs[:top_k]


RETRIEVAL_METHODS = {
    "simple": retrieve_simple,
    "semantic": retrieve_semantic,
    "hybrid": retrieve_hybrid,
    "reranked": retrieve_reranked,
}


def retrieve(doc_id: str, question: str, top_k: int = 3, method: str = "simple"):
    fn = RETRIEVAL_METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown retrieval method: {method}. Choose from: {list(RETRIEVAL_METHODS.keys())}")
    return fn(doc_id, question, top_k)


def _retrieve_and_rerank(doc_id: str, question: str, top_k: int = 3):
    return retrieve_reranked(doc_id, question, top_k)


def answer_with_rag(doc_id: str, question: str, top_k: int = 3, method: str = "simple"):
    retrieved_docs = retrieve(doc_id, question, top_k, method)

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant answering questions about an employee handbook.\n"
            "Use the context below to answer the question directly and concisely.\n"
            "If the context contains multiple pieces of information, prioritize the one that most directly answers the question.\n"
            "If the answer is clearly stated in the context, provide it.\n"
            "Only say \"I don't know\" if you have carefully checked the entire context and the answer is truly not there.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
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


def answer_with_rag_with_history(doc_id: str, question: str, top_k: int, session: Session, method: str = "simple"):
    retrieved_docs = retrieve(doc_id, question, top_k, method)

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    if session and session.turn_count > 0:
        full_prompt = build_rag_prompt(session, question, context)
    else:
        full_prompt = (
            "You are a helpful assistant answering questions about an employee handbook.\n"
            "Use the context below to answer the question directly and concisely.\n"
            "If the context contains multiple pieces of information, prioritize the one that most directly answers the question.\n"
            "If the answer is clearly stated in the context, provide it.\n"
            "Only say \"I don't know\" if you have carefully checked the entire context and the answer is truly not there.\n\n"
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
