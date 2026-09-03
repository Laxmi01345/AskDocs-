"""
Retrieval Validation - checks if the right chunks were retrieved.
"""
import numpy as np
from typing import List, Dict, Optional
from app.embeddings import get_embeddings


def validate_retrieval(
    question: str,
    retrieved_chunks: List[str],
    ground_truth_chunks: Optional[List[str]] = None,
    ground_truth_answer: Optional[str] = None,
) -> Dict:
    """
    Validate retrieval quality.
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
        results["verdict"] = "POOR"
        return results

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

    hit_threshold = 0.35
    hits = sum(1 for r in relevances if r > hit_threshold)
    results["metrics"]["hit_rate"] = round(hits / len(relevances), 4)
    results["metrics"]["hits_above_threshold"] = hits

    if ground_truth_chunks:
        truth_embs = embeddings.embed_documents(ground_truth_chunks)

        sim_threshold = 0.3
        retrieved_hits = []
        for r_emb in chunk_embs:
            r_norm = r_emb / (np.linalg.norm(r_emb) or 1)
            max_sim = 0
            for t_emb in truth_embs:
                t_norm = t_emb / (np.linalg.norm(t_emb) or 1)
                sim = float(np.dot(r_norm, t_norm))
                max_sim = max(max_sim, sim)
            retrieved_hits.append(max_sim >= sim_threshold)

        truth_hits = []
        for t_emb in truth_embs:
            t_norm = t_emb / (np.linalg.norm(t_emb) or 1)
            max_sim = 0
            for r_emb in chunk_embs:
                r_norm = r_emb / (np.linalg.norm(r_emb) or 1)
                sim = float(np.dot(t_norm, r_norm))
                max_sim = max(max_sim, sim)
            truth_hits.append(max_sim >= sim_threshold)

        ground_truths_found = sum(truth_hits)
        results["metrics"]["recall_at_5"] = round(
            ground_truths_found / len(truth_embs) if truth_embs else 0, 4
        )

        first_relevant_rank = None
        for i, hit in enumerate(retrieved_hits):
            if hit:
                first_relevant_rank = i + 1
                break
        results["metrics"]["mrr"] = round(
            1.0 / first_relevant_rank if first_relevant_rank else 0, 4
        )

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
