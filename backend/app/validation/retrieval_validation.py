"""
Retrieval Validation - checks if the right chunks were retrieved.

All metrics in this module are deterministic and local:
they use embedding cosine similarity only, with NO LLM/API calls.

Metric overview:
    - Semantic Recall@5 : fraction of ground-truth chunks semantically matched
    - MRR               : reciprocal rank of the first matched chunk
    - NDCG@5            : ranking quality of the retrieved list
    - Precision/Recall/F1 : set overlap with ground truth (when provided)
    - Average Relevance : mean cosine similarity between question and chunks
    - Diversity         : 1 - mean pairwise similarity among retrieved chunks

Important: "Semantic Recall@5" does NOT use exact chunk-ID matching.
Retrieved and ground-truth chunks are considered a semantic match when
their embedding cosine similarity is above RETRIEVAL_MATCH_THRESHOLD.
"""
import numpy as np
from typing import List, Dict, Optional
from app.embeddings import get_embeddings

# --- Configuration constants ---

# Cosine similarity threshold for considering a retrieved chunk a semantic
# match with a ground-truth chunk (used by Semantic Recall@5, MRR, NDCG@5,
# Precision/Recall/F1). Not changed from the original implementation.
RETRIEVAL_MATCH_THRESHOLD = 0.30

# Cosine similarity threshold above which a question-chunk pair counts
# as a "hit" for the hit_rate metric.
HIT_THRESHOLD = 0.35

# Cutoff k for NDCG@k (spec requires NDCG@5).
NDCG_K = 5


def calculate_ndcg_at_k(relevance_gains: List[float], k: int = NDCG_K) -> float:
    """
    Calculate NDCG@k from binary relevance gains.

    Deterministic and local - no LLM call.

    Args:
        relevance_gains: one value per retrieved chunk, in rank order.
            1.0 if the chunk semantically matched a ground-truth chunk
            (cosine similarity >= RETRIEVAL_MATCH_THRESHOLD), else 0.0.
        k: cutoff (default NDCG_K = 5).

    Returns:
        NDCG@k in [0.0, 1.0]. Returns 0.0 when no gain exists in the
        top-k window (IDCG == 0).

    Formula:
        DCG  = sum(gain_i / log2(i + 2))          for i in 0..k-1
        IDCG = same, computed over gains sorted descending
        NDCG = DCG / IDCG   (0.0 if IDCG == 0)
    """
    gains = list(relevance_gains[:k])
    if not gains:
        return 0.0

    dcg = sum(gain / np.log2(i + 2) for i, gain in enumerate(gains))

    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(gain / np.log2(i + 2) for i, gain in enumerate(ideal_gains))

    if idcg == 0:
        return 0.0
    return round(float(dcg / idcg), 4)


def validate_retrieval(
    question: str,
    retrieved_chunks: List[str],
    ground_truth_chunks: Optional[List[str]] = None,
    ground_truth_answer: Optional[str] = None,
) -> Dict:
    """
    Validate retrieval quality.

    Args:
        question: The user's question.
        retrieved_chunks: Chunk texts returned by the retriever, in rank order.
        ground_truth_chunks: Known-correct chunk texts. When provided, enables
            Semantic Recall@5, MRR, NDCG@5, Precision, Recall and F1. Chunks
            are matched semantically (embedding cosine similarity >=
            RETRIEVAL_MATCH_THRESHOLD), not by exact ID/text equality.
        ground_truth_answer: Known-correct answer (accepted for API
            compatibility; not used by retrieval metrics).

    Returns:
        Dict with num_chunks_retrieved, metrics, relevance_per_chunk, verdict.
        Metrics include: diversity, avg_relevance, max_relevance,
        min_relevance, hit_rate, hits_above_threshold, recall_at_5, mrr,
        ndcg_at_5, and (when ground truth is provided) precision, recall, f1.
    """
    results = {
        "num_chunks_retrieved": len(retrieved_chunks),
        "metrics": {},
    }

    if not retrieved_chunks:
        results["metrics"]["diversity"] = 0
        results["metrics"]["avg_relevance"] = 0
        results["metrics"]["max_relevance"] = 0
        results["metrics"]["min_relevance"] = 0
        results["metrics"]["hit_rate"] = 0
        results["metrics"]["hits_above_threshold"] = 0
        results["relevance_per_chunk"] = []
        results["metrics"]["recall_at_5"] = 0
        results["metrics"]["mrr"] = 0
        results["metrics"]["ndcg_at_5"] = 0
        results["verdict"] = "POOR"
        return results

    # --- Chunk diversity ---
    if len(retrieved_chunks) > 1:
        embeddings = get_embeddings()
        chunk_embs = embeddings.embed_documents(retrieved_chunks)
        norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = np.array(chunk_embs) / norms
        sim_matrix = normalized @ normalized.T
        n = len(retrieved_chunks)
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        avg_similarity = float(np.mean(upper_tri))
        results["metrics"]["diversity"] = round(1.0 - avg_similarity, 4)
    else:
        results["metrics"]["diversity"] = 1.0

    # --- Question-chunk relevance ---
    embeddings = get_embeddings()
    q_emb = embeddings.embed_query(question)
    chunk_embs = embeddings.embed_documents(retrieved_chunks)

    relevances = []
    for c_emb in chunk_embs:
        q_norm = q_emb / (np.linalg.norm(q_emb) or 1)
        c_norm = c_emb / (np.linalg.norm(c_emb) or 1)
        sim = float(np.dot(q_norm, c_norm))
        relevances.append(sim)

    results["metrics"]["avg_relevance"] = round(float(np.mean(relevances)), 4)
    results["metrics"]["max_relevance"] = round(float(np.max(relevances)), 4)
    results["metrics"]["min_relevance"] = round(float(np.min(relevances)), 4)
    results["relevance_per_chunk"] = [round(r, 4) for r in relevances]

    # --- Hit rate ---
    hits = sum(1 for r in relevances if r > HIT_THRESHOLD)
    results["metrics"]["hit_rate"] = round(hits / len(relevances), 4)
    results["metrics"]["hits_above_threshold"] = hits

    # --- Ground-truth based metrics (semantic matching) ---
    if ground_truth_chunks:
        truth_embs = embeddings.embed_documents(ground_truth_chunks)

        # Semantic match: cosine similarity >= RETRIEVAL_MATCH_THRESHOLD
        retrieved_hits = []
        for r_emb in chunk_embs:
            r_norm = r_emb / (np.linalg.norm(r_emb) or 1)
            max_sim = 0
            for t_emb in truth_embs:
                t_norm = t_emb / (np.linalg.norm(t_emb) or 1)
                sim = float(np.dot(r_norm, t_norm))
                max_sim = max(max_sim, sim)
            retrieved_hits.append(max_sim >= RETRIEVAL_MATCH_THRESHOLD)

        truth_hits = []
        for t_emb in truth_embs:
            t_norm = t_emb / (np.linalg.norm(t_emb) or 1)
            max_sim = 0
            for r_emb in chunk_embs:
                r_norm = r_emb / (np.linalg.norm(r_emb) or 1)
                sim = float(np.dot(t_norm, r_norm))
                max_sim = max(max_sim, sim)
            truth_hits.append(max_sim >= RETRIEVAL_MATCH_THRESHOLD)

        # Semantic Recall@5
        ground_truths_found = sum(truth_hits)
        results["metrics"]["recall_at_5"] = round(
            ground_truths_found / len(truth_embs) if truth_embs else 0, 4
        )

        # MRR
        first_relevant_rank = None
        for i, hit in enumerate(retrieved_hits):
            if hit:
                first_relevant_rank = i + 1
                break
        results["metrics"]["mrr"] = round(
            1.0 / first_relevant_rank if first_relevant_rank else 0, 4
        )

        # NDCG@5
        relevance_gains = [1.0 if hit else 0.0 for hit in retrieved_hits]
        results["metrics"]["ndcg_at_5"] = calculate_ndcg_at_k(
            relevance_gains, k=NDCG_K
        )

        # Precision / Recall / F1
        relevant_in_top_k = sum(retrieved_hits)
        true_positives = ground_truths_found
        false_positives = len(retrieved_chunks) - relevant_in_top_k
        false_negatives = len(truth_embs) - ground_truths_found

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results["metrics"]["precision"] = round(precision, 4)
        results["metrics"]["recall"] = round(recall, 4)
        results["metrics"]["f1"] = round(f1, 4)
    else:
        results["metrics"]["recall_at_5"] = None
        results["metrics"]["mrr"] = None
        results["metrics"]["ndcg_at_5"] = None

    # --- Verdict ---
    avg_rel = results["metrics"]["avg_relevance"]
    diversity = results["metrics"]["diversity"]

    if avg_rel > 0.6 and diversity > 0.3:
        results["verdict"] = "EXCELLENT"
    elif avg_rel > 0.4 and diversity > 0.2:
        results["verdict"] = "GOOD"
    elif avg_rel > 0.3:
        results["verdict"] = "FAIR"
    else:
        results["verdict"] = "POOR"

    return results
