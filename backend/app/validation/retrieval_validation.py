"""
Retrieval Validation - checks if the right chunks were retrieved.
"""
import numpy as np
from typing import List, Dict
from app.embeddings import get_embeddings


def validate_retrieval(
    question: str,
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str] = None,
    ground_truth_answer: str = None,
) -> Dict:
    """
    Validate retrieval quality.

    Args:
        question: The user's question
        retrieved_chunks: List of retrieved chunk texts
        ground_truth_chunks: Known correct chunks (optional)
        ground_truth_answer: Known correct answer (optional)
    """
    results = {
        "num_chunks_retrieved": len(retrieved_chunks),
        "metrics": {},
    }

    # 1. Chunk Diversity - are we retrieving diverse content?
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

    # 2. Question-Chunk Relevance - how relevant are chunks to the question?
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

    # 3. Hit Rate - is at least one chunk above threshold?
    hit_threshold = 0.35
    hits = sum(1 for r in relevances if r > hit_threshold)
    results["metrics"]["hit_rate"] = round(hits / len(relevances), 4) if relevances else 0
    results["metrics"]["hits_above_threshold"] = hits

    # 4. Ground Truth Comparison (if provided)
    if ground_truth_chunks:
        retrieved_set = set(c.lower().strip() for c in retrieved_chunks)
        truth_set = set(c.lower().strip() for c in ground_truth_chunks)

        true_positives = len(retrieved_set & truth_set)
        false_positives = len(retrieved_set - truth_set)
        false_negatives = len(truth_set - retrieved_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results["metrics"]["precision"] = round(precision, 4)
        results["metrics"]["recall"] = round(recall, 4)
        results["metrics"]["f1"] = round(f1, 4)

    # 5. Verdict
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
