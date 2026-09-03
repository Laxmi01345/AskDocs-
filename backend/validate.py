"""
Full Validation Script - Retrieval + Generation validation.
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from app.retrieval import retrieve
from app.validation.retrieval_validation import validate_retrieval
from app.validation.generation_validation import validate_generation

METHODS = ["simple", "semantic", "hybrid", "reranked"]


def load_eval_dataset(path="employee_eval.json"):
    with open(path, "r") as f:
        return json.load(f)


def run_validation(doc_id, method="simple", dataset_path="employee_eval.json", top_k=3):
    dataset = load_eval_dataset(dataset_path)
    results = []

    retrieval_scores = []
    generation_scores = []
    faithfulness_scores = []
    relevancy_scores = []
    hallucination_scores = []
    recall_scores = []
    mrr_scores = []

    print(f"\n{'='*70}")
    print(f"  VALIDATION REPORT: {method.upper()}")
    print(f"{'='*70}\n")

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        ground_truth_chunks = item.get("ground_truth_chunks", None)
        print(f"[{i}/{len(dataset)}] Q: {question}")

        try:
            retrieved_docs = retrieve(doc_id, question, top_k, method)
            chunk_texts = [doc.page_content for doc in retrieved_docs]

            retrieval_result = validate_retrieval(
                question=question,
                retrieved_chunks=chunk_texts,
                ground_truth_chunks=ground_truth_chunks,
                ground_truth_answer=ground_truth,
            )

            context = "\n\n".join(chunk_texts)
            prompt = (
                "Answer the question using ONLY the context below.\n"
                "If the answer is not present in the context, say \"I don't know\".\n\n"
                f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            )
            from app.llm import CerebrasLLM
            llm = CerebrasLLM()
            answer = llm.invoke(prompt)

            generation_result = validate_generation(
                question=question,
                answer=answer,
                context=context,
                ground_truth=ground_truth,
            )

            r_score = retrieval_result["metrics"].get("avg_relevance", 0)
            g_score = generation_result["metrics"].get("overall_score", 0)
            faith = generation_result["metrics"].get("faithfulness", 0) or 0
            relev = generation_result["metrics"].get("answer_relevancy", 0) or 0
            halluc = generation_result["metrics"].get("hallucination_score", 0) or 0
            recall_at_5 = retrieval_result["metrics"].get("recall_at_5")
            mrr = retrieval_result["metrics"].get("mrr")

            retrieval_scores.append(r_score)
            if g_score is not None:
                generation_scores.append(g_score)
            faithfulness_scores.append(faith)
            relevancy_scores.append(relev)
            hallucination_scores.append(halluc)
            if recall_at_5 is not None:
                recall_scores.append(recall_at_5)
            if mrr is not None:
                mrr_scores.append(mrr)

            print(f"  Answer: {answer[:80]}...")
            print(f"  Retrieval: {retrieval_result['verdict']} (relevance={r_score:.2f})")
            g_str = f"{g_score:.2f}" if g_score else "N/A"
            print(f"  Generation: {generation_result['verdict']} (score={g_str})")
            print()

            results.append({
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "retrieval": retrieval_result,
                "generation": generation_result,
            })

        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({
                "question": question,
                "answer": None,
                "ground_truth": ground_truth,
                "retrieval": {"verdict": "ERROR", "metrics": {}},
                "generation": {"verdict": "ERROR", "metrics": {}},
                "error": str(e),
            })

    total = len(dataset)
    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
    avg_generation = sum(generation_scores) / len(generation_scores) if generation_scores else 0
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
    avg_hallucination = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0

    excellent_retrieval = sum(1 for r in results if r["retrieval"]["verdict"] == "EXCELLENT")
    excellent_generation = sum(1 for r in results if r["generation"]["verdict"] == "EXCELLENT")

    print(f"{'='*70}")
    print(f"  SUMMARY: {method.upper()}")
    print(f"{'='*70}")
    print(f"  Total Questions:          {total}")
    print(f"  Recall@5:                 {avg_recall*100:.1f}%")
    print(f"  MRR:                      {avg_mrr:.4f}")
    print(f"  Avg Relevance:            {avg_retrieval*100:.1f}%")
    print(f"  Avg Faithfulness:         {avg_faithfulness*100:.1f}%")
    print(f"  Avg Answer Relevancy:     {avg_relevancy*100:.1f}%")
    print(f"  Avg Hallucination:        {avg_hallucination*100:.1f}%")
    print(f"  Avg Generation Score:     {avg_generation*100:.1f}%")
    print(f"  Excellent Retrieval:      {excellent_retrieval}/{total} ({excellent_retrieval/total*100:.1f}%)")
    print(f"  Excellent Generation:     {excellent_generation}/{total} ({excellent_generation/total*100:.1f}%)")
    print(f"{'='*70}\n")

    report = {
        "method": method,
        "total": total,
        "recall_at_5": round(avg_recall, 4),
        "mrr": round(avg_mrr, 4),
        "avg_relevance": round(avg_retrieval, 4),
        "avg_faithfulness": round(avg_faithfulness, 4),
        "avg_answer_relevancy": round(avg_relevancy, 4),
        "avg_hallucination": round(avg_hallucination, 4),
        "avg_generation_score": round(avg_generation, 4),
        "excellent_retrieval": excellent_retrieval,
        "excellent_generation": excellent_generation,
        "results": results,
    }

    report_path = Path(f"report_{method}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to: {report_path}")

    return report


def compare_methods(doc_id, dataset_path="employee_eval.json", top_k=3):
    reports = {}
    for method in METHODS:
        reports[method] = run_validation(doc_id, method, dataset_path, top_k)

    baseline = reports["simple"]

    print(f"\n{'='*90}")
    print(f"  METHOD COMPARISON REPORT")
    print(f"  Baseline: Simple Chunking")
    print(f"  Dataset: {dataset_path} ({baseline['total']} questions)")
    print(f"{'='*90}\n")

    def pct_delta(val, base):
        if base == 0:
            return "---"
        delta = ((val - base) / base) * 100
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f}%"

    header = f"{'Configuration':<30} {'Recall@5':<12} {'MRR':<10} {'Correctness':<14} {'Faithfulness':<14}"
    sep = "-" * 90

    print(sep)
    print(header)
    print(sep)

    for method in METHODS:
        r = reports[method]
        recall = f"{r['recall_at_5']*100:.1f}%"
        mrr = f"{r['mrr']:.2f}"
        correctness = f"{r['avg_generation_score']*100:.1f}%"
        faithfulness = f"{r['avg_faithfulness']*100:.1f}%"

        if method == "simple":
            label = "Simple Chunking"
            delta_str = "Baseline"
        elif method == "semantic":
            label = "Semantic Chunking"
            delta_str = pct_delta(r['avg_generation_score'], baseline['avg_generation_score'])
        elif method == "hybrid":
            label = "Hybrid (BM25+Vec+RRF)"
            delta_str = pct_delta(r['avg_generation_score'], baseline['avg_generation_score'])
        elif method == "reranked":
            label = "Reranking"
            delta_str = pct_delta(r['avg_generation_score'], baseline['avg_generation_score'])
        else:
            label = method
            delta_str = ""

        print(f"{label:<30} {recall:<12} {mrr:<10} {correctness:<14} {faithfulness:<14}")

    print(sep)

    print(f"\n  IMPROVEMENT VS BASELINE (Simple Chunking)")
    print(sep)
    for method in METHODS[1:]:
        r = reports[method]
        recall_d = pct_delta(r['recall_at_5'], baseline['recall_at_5'])
        mrr_d = pct_delta(r['mrr'], baseline['mrr'])
        gen_d = pct_delta(r['avg_generation_score'], baseline['avg_generation_score'])
        faith_d = pct_delta(r['avg_faithfulness'], baseline['avg_faithfulness'])
        print(f"  {method.upper():<12} Recall: {recall_d:<10} MRR: {mrr_d:<10} Correctness: {gen_d:<10} Faithfulness: {faith_d}")
    print(sep)

    comparison = {
        "baseline": "simple",
        "dataset": dataset_path,
        "methods": {m: {
            "recall_at_5": reports[m]["recall_at_5"],
            "mrr": reports[m]["mrr"],
            "avg_relevance": reports[m]["avg_relevance"],
            "avg_faithfulness": reports[m]["avg_faithfulness"],
            "avg_generation_score": reports[m]["avg_generation_score"],
        } for m in METHODS}
    }
    with open("comparison_report.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to: comparison_report.json\n")

    return reports


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG validation")
    parser.add_argument("--doc-id", required=True, help="Document ID to test")
    parser.add_argument("--dataset", default="employee_eval.json", help="Eval dataset")
    parser.add_argument("--top-k", type=int, default=3, help="Chunks to retrieve")
    parser.add_argument("--method", choices=METHODS, help="Run single method")
    parser.add_argument("--compare", action="store_true", help="Run all methods and compare")
    args = parser.parse_args()

    if args.compare:
        compare_methods(args.doc_id, args.dataset, args.top_k)
    elif args.method:
        run_validation(args.doc_id, args.method, args.dataset, args.top_k)
    else:
        run_validation(args.doc_id, "simple", args.dataset, args.top_k)
