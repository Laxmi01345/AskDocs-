"""
Full Validation Script - Retrieval + Generation validation.
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from app.retrieval import _retrieve_and_rerank
from app.validation.retrieval_validation import validate_retrieval
from app.validation.generation_validation import validate_generation


def load_eval_dataset(path="employee_eval.json"):
    with open(path, "r") as f:
        return json.load(f)


def run_full_validation(doc_id, dataset_path="employee_eval.json", top_k=3):
    dataset = load_eval_dataset(dataset_path)
    results = []

    retrieval_scores = []
    generation_scores = []

    print(f"\n{'='*70}")
    print(f"  FULL VALIDATION REPORT")
    print(f"  Retrieval + Generation")
    print(f"{'='*70}\n")

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        print(f"[{i}/{len(dataset)}] Q: {question}")

        try:
            # Get retrieval results
            retrieved_docs = _retrieve_and_rerank(doc_id, question, top_k)

            chunk_texts = [doc.page_content for doc in retrieved_docs]

            # Retrieval Validation
            retrieval_result = validate_retrieval(
                question=question,
                retrieved_chunks=chunk_texts,
                ground_truth_answer=ground_truth,
            )

            # Get answer
            context = "\n\n".join(chunk_texts)
            from app.context_builder import build_rag_prompt
            from app.session import Session

            prompt = (
                "Answer the question using ONLY the context below.\n"
                "If the answer is not present in the context, say \"I don't know\".\n\n"
                f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            )
            from app.llm import CerebrasLLM
            llm = CerebrasLLM()
            answer = llm.invoke(prompt)

            # Generation Validation
            generation_result = validate_generation(
                question=question,
                answer=answer,
                context=context,
                ground_truth=ground_truth,
            )

            # Print Results
            r_score = retrieval_result["metrics"].get("avg_relevance", 0)
            g_score = generation_result["metrics"].get("overall_score", 0)

            retrieval_scores.append(r_score)
            if g_score is not None:
                generation_scores.append(g_score)

            print(f"  Answer: {answer[:80]}...")
            print(f"  Retrieval: {retrieval_result['verdict']} (relevance={r_score:.2f}, diversity={retrieval_result['metrics']['diversity']:.2f})")
            g_str = f"{g_score:.2f}" if g_score else "N/A"
            print(f"  Generation: {generation_result['verdict']} (score={g_str})")
            if generation_result["metrics"].get("unsupported_claims"):
                print(f"  ⚠ Unsupported claims: {generation_result['metrics']['unsupported_claims']}")
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

    # Summary
    total = len(dataset)
    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
    avg_generation = sum(generation_scores) / len(generation_scores) if generation_scores else 0

    excellent_retrieval = sum(1 for r in results if r["retrieval"]["verdict"] == "EXCELLENT")
    excellent_generation = sum(1 for r in results if r["generation"]["verdict"] == "EXCELLENT")

    print(f"{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Questions:          {total}")
    print(f"  Avg Retrieval Relevance:  {avg_retrieval:.4f}")
    print(f"  Avg Generation Score:     {avg_generation:.4f}")
    print(f"  Excellent Retrieval:      {excellent_retrieval}/{total} ({excellent_retrieval/total*100:.1f}%)")
    print(f"  Excellent Generation:     {excellent_generation}/{total} ({excellent_generation/total*100:.1f}%)")
    print(f"{'='*70}\n")

    # Save report
    report_path = Path("full_validation_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "total": total,
            "avg_retrieval_relevance": round(avg_retrieval, 4),
            "avg_generation_score": round(avg_generation, 4),
            "excellent_retrieval": excellent_retrieval,
            "excellent_generation": excellent_generation,
            "results": results,
        }, f, indent=2, default=str)
    print(f"Report saved to: {report_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full RAG validation")
    parser.add_argument("--doc-id", required=True, help="Document ID to test")
    parser.add_argument("--dataset", default="employee_eval.json", help="Eval dataset")
    parser.add_argument("--top-k", type=int, default=3, help="Chunks to retrieve")
    args = parser.parse_args()

    run_full_validation(args.doc_id, args.dataset, args.top_k)
