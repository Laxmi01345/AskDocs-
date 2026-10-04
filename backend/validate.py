"""
AskDocs Evaluation - single entry point.

Two stages:
  1. Retrieval evaluation (deterministic, local): Semantic Recall@5, MRR,
     NDCG@5, Precision, Recall, F1, average relevance, diversity.
     No API keys needed.
  2. Generation evaluation (Ragas LLM judge): faithfulness, answer_relevancy,
     answer_correctness by default; context_precision/context_recall optional
     via --ragas-metrics. Answers are generated once and reused.

Legacy custom LLM-as-judge remains available behind --custom-judge.

Usage:
  python validate.py --doc-id <id>                                  # full run
  python validate.py --doc-id <id> --compare                        # all methods
  python validate.py --doc-id <id> --skip-generation                # local only
  python validate.py --doc-id <id> --ragas-metrics faithfulness
  python validate.py --doc-id <id> --custom-judge
  python validate.py --doc-id <id> --limit 1                        # smoke test

Results are written to evaluation_results.json (incrementally: saved after
each method completes).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from app.retrieval import retrieve
from app.validation import (
    DEFAULT_RAGAS_METRICS,
    SUPPORTED_RAGAS_METRICS,
    validate_generation,
    validate_generation_ragas,
    validate_retrieval,
)

METHODS = ["simple", "semantic", "hybrid", "reranked"]
METHOD_LABELS = {
    "simple": "Simple Chunking",
    "semantic": "Semantic Chunking",
    "hybrid": "Hybrid (BM25+Vec+RRF)",
    "reranked": "Reranking",
}
OUTPUT_FILE = "evaluation_results.json"

# Retrieval metric keys in output schema order (spec).
RETRIEVAL_AGG_KEYS = [
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
    "precision",
    "recall",
    "f1",
    "average_relevance",
    "diversity",
]


def load_eval_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values):
    numeric = [v for v in values if isinstance(v, (int, float))]
    return round(sum(numeric) / len(numeric), 4) if numeric else None


def generate_answer(question: str, context: str) -> str:
    from app.llm import CerebrasLLM

    prompt = (
        "You are a helpful assistant answering questions about an employee handbook.\n"
        "Use the context below to answer the question directly and concisely.\n"
        "If the answer is clearly stated in the context, provide it.\n"
        "Only say \"I don't know\" if you have carefully checked the entire context "
        "and the answer is truly not there.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    return CerebrasLLM().invoke(prompt)


def evaluate_method(
    doc_id: str,
    method: str,
    dataset,
    top_k: int,
    skip_generation: bool = False,
    ragas_metrics=None,
    custom_judge: bool = False,
) -> dict:
    """
    Evaluate one retrieval method over the dataset.

    Returns a method result with retrieval, generation (unless skipped)
    and api_usage blocks.
    """
    per_question_retrieval = []
    generation_samples = []  # {question, answer, contexts, ground_truth}
    generation_errors = []
    answer_llm_calls = 0

    print(f"\n{'=' * 70}")
    print(f"  EVALUATION: {METHOD_LABELS.get(method, method)}")
    print(f"{'=' * 70}\n")

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        ground_truth_chunks = item.get("ground_truth_chunks", None)
        print(f"[{i}/{len(dataset)}] Q: {question}")

        chunk_texts = []
        try:
            retrieved_docs = retrieve(doc_id, question, top_k, method)
            chunk_texts = [doc.page_content for doc in retrieved_docs]

            retrieval_result = validate_retrieval(
                question=question,
                retrieved_chunks=chunk_texts,
                ground_truth_chunks=ground_truth_chunks,
                ground_truth_answer=ground_truth,
            )
            per_question_retrieval.append(
                {"question": question, **retrieval_result}
            )

            m = retrieval_result["metrics"]
            print(
                f"  Retrieval: {retrieval_result['verdict']} "
                f"(recall@5={m.get('recall_at_5')}, mrr={m.get('mrr')}, "
                f"ndcg@5={m.get('ndcg_at_5')})"
            )
        except Exception as e:
            print(f"  Retrieval ERROR: {e}")
            per_question_retrieval.append(
                {
                    "question": question,
                    "num_chunks_retrieved": 0,
                    "metrics": {},
                    "relevance_per_chunk": [],
                    "verdict": "ERROR",
                    "error": str(e),
                }
            )

        if not skip_generation:
            try:
                context = "\n\n".join(chunk_texts)
                answer = generate_answer(question, context)
                answer_llm_calls += 1
                print(f"  Answer: {answer[:80]}...")
                generation_samples.append(
                    {
                        "question": question,
                        "answer": answer,
                        "contexts": chunk_texts,
                        "ground_truth": ground_truth,
                    }
                )
            except Exception as e:
                print(f"  Generation ERROR: {e}")
                generation_errors.append({"question": question, "error": str(e)})
        print()

    retrieval_block = {
        "recall_at_5": _mean([r["metrics"].get("recall_at_5") for r in per_question_retrieval]),
        "mrr": _mean([r["metrics"].get("mrr") for r in per_question_retrieval]),
        "ndcg_at_5": _mean([r["metrics"].get("ndcg_at_5") for r in per_question_retrieval]),
        "precision": _mean([r["metrics"].get("precision") for r in per_question_retrieval]),
        "recall": _mean([r["metrics"].get("recall") for r in per_question_retrieval]),
        "f1": _mean([r["metrics"].get("f1") for r in per_question_retrieval]),
        "average_relevance": _mean([r["metrics"].get("avg_relevance") for r in per_question_retrieval]),
        "diversity": _mean([r["metrics"].get("diversity") for r in per_question_retrieval]),
        "per_question": per_question_retrieval,
    }

    method_result = {"retrieval": retrieval_block}
    api_usage = {
        "answer_llm_calls": answer_llm_calls,
        "ragas_judge_evaluations": 0,
        "custom_judge_calls": 0,
    }

    if not skip_generation:
        generation_block = {"per_question": []}
        errors = [f"answer generation failed: {e['error']} ({e['question']})"
                  for e in generation_errors]

        # --- Ragas (default generation evaluation) ---
        if generation_samples:
            try:
                ragas_result = validate_generation_ragas(
                    generation_samples,
                    metric_names=ragas_metrics,
                )
                generation_block.update(ragas_result["metrics"])
                errors.extend(ragas_result["errors"])
                api_usage["ragas_judge_evaluations"] = (
                    len(ragas_metrics or DEFAULT_RAGAS_METRICS)
                    * len(generation_samples)
                )
                print(f"  Ragas metrics: {ragas_result['metrics']}")
                if ragas_result["errors"]:
                    print(f"  Ragas warnings: {ragas_result['errors']}")
            except Exception as e:
                errors.append(f"ragas evaluation failed: {e}")
                print(f"  Ragas ERROR: {e}")
        else:
            errors.append("ragas evaluation skipped: no answers generated")
        generation_block["errors"] = errors

        # --- Legacy custom judge (opt-in) ---
        if custom_judge:
            custom_scores = []
            for sample in generation_samples:
                if not sample.get("answer"):
                    continue
                try:
                    judge = validate_generation(
                        question=sample["question"],
                        answer=sample["answer"],
                        context="\n\n".join(sample["contexts"]),
                        ground_truth=sample.get("ground_truth"),
                    )
                    generation_block["per_question"].append(
                        {
                            "question": sample["question"],
                            "answer": sample["answer"],
                            "metrics": judge["metrics"],
                            "verdict": judge["verdict"],
                        }
                    )
                    score = judge["metrics"].get("overall_score")
                    if score is not None:
                        custom_scores.append(score)
                    api_usage["custom_judge_calls"] += 4
                except Exception as e:
                    errors.append(f"custom judge failed: {e}")
            generation_block["custom_judge"] = {
                "custom_overall_generation_score": _mean(custom_scores)
            }
        else:
            for sample in generation_samples:
                generation_block["per_question"].append(
                    {
                        "question": sample["question"],
                        "answer": sample["answer"],
                        "ground_truth": sample.get("ground_truth"),
                    }
                )

        method_result["generation"] = generation_block
    else:
        method_result["generation"] = None

    method_result["api_usage"] = api_usage
    return method_result


def run_evaluation(
    doc_id: str,
    dataset_path: str,
    top_k: int,
    methods,
    skip_generation: bool,
    ragas_metrics,
    custom_judge: bool,
    limit: int = 0,
) -> dict:
    dataset = load_eval_dataset(dataset_path)
    if limit > 0:
        dataset = dataset[:limit]

    ragas_metrics = list(ragas_metrics or DEFAULT_RAGAS_METRICS)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "doc_id": doc_id,
            "dataset": dataset_path,
            "questions": len(dataset),
            "top_k": top_k,
            "mode": "retrieval_only" if skip_generation else "full",
            "ragas_metrics": None if skip_generation else ragas_metrics,
            "judge_model": None if skip_generation else os.environ.get(
                "RAGAS_JUDGE_MODEL", "qwen/qwen3.8-27b"
            ),
            "custom_judge": bool(custom_judge and not skip_generation),
            "methods": methods,
        },
        "results": {},
    }
    _save(results)

    for method in methods:
        results["results"][method] = evaluate_method(
            doc_id=doc_id,
            method=method,
            dataset=dataset,
            top_k=top_k,
            skip_generation=skip_generation,
            ragas_metrics=ragas_metrics,
            custom_judge=custom_judge and not skip_generation,
        )
        _save(results)  # incremental save after each method

    return results


def _save(results: dict):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


def _pct_delta(val, base):
    if base in (None, 0) or val is None:
        return "---"
    delta = ((val - base) / base) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def print_comparison(results: dict):
    methods = [m for m in results["config"]["methods"] if m in results["results"]]
    if not methods:
        return

    print(f"\n{'=' * 100}")
    print("  RETRIEVAL COMPARISON (deterministic, local)")
    print(f"{'=' * 100}")
    header = f"{'Method':<22} {'Recall@5':<10} {'MRR':<8} {'NDCG@5':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AvgRel':<8} {'Divers':<8}"
    print(header)
    print("-" * 100)
    for method in methods:
        r = results["results"][method]["retrieval"]

        def fmt(v, pct=False):
            if v is None:
                return "n/a"
            return f"{v * 100:.1f}%" if pct else f"{v:.4f}"

        print(
            f"{METHOD_LABELS.get(method, method):<22} "
            f"{fmt(r['recall_at_5'], True):<10} {fmt(r['mrr']):<8} "
            f"{fmt(r['ndcg_at_5']):<8} {fmt(r['precision'], True):<8} "
            f"{fmt(r['recall'], True):<8} {fmt(r['f1']):<8} "
            f"{fmt(r['average_relevance']):<8} {fmt(r['diversity']):<8}"
        )
    print("-" * 100)

    if results["config"]["mode"] == "retrieval_only":
        print("\n  (generation skipped - run without --skip-generation for Ragas metrics)")
        return

    ragas_metrics = results["config"]["ragas_metrics"]
    print(f"\n{'=' * 90}")
    print(f"  GENERATION COMPARISON (Ragas: {', '.join(ragas_metrics)})")
    print(f"{'=' * 90}")
    gen_header = f"{'Method':<22} " + " ".join(f"{m:<20}" for m in ragas_metrics)
    if results["config"]["custom_judge"]:
        gen_header += f" {'CustomJudge':<12}"
    print(gen_header)
    print("-" * 90)
    for method in methods:
        g = results["results"][method].get("generation")
        if not g:
            continue
        row = f"{METHOD_LABELS.get(method, method):<22} "
        for name in ragas_metrics:
            v = g.get(name)
            row += f"{(f'{v:.4f}' if v is not None else 'n/a'):<20} "
        if results["config"]["custom_judge"]:
            cj = (g.get("custom_judge") or {}).get("custom_overall_generation_score")
            row += f"{(f'{cj:.4f}' if cj is not None else 'n/a'):<12}"
        print(row)
    print("-" * 90)

    if "simple" in methods and len(methods) > 1:
        baseline = results["results"]["simple"]["retrieval"]
        print("\n  RETRIEVAL DELTA VS BASELINE (Simple Chunking)")
        for method in methods:
            if method == "simple":
                continue
            r = results["results"][method]["retrieval"]
            print(
                f"  {method:<12} "
                f"Recall@5: {_pct_delta(r['recall_at_5'], baseline['recall_at_5']):<10} "
                f"MRR: {_pct_delta(r['mrr'], baseline['mrr']):<10} "
                f"NDCG@5: {_pct_delta(r['ndcg_at_5'], baseline['ndcg_at_5']):<10} "
                f"F1: {_pct_delta(r['f1'], baseline['f1'])}"
            )


def main():
    # Windows consoles (cp1252) crash on characters like U+202F in LLM
    # answers; replace unencodable chars instead of raising.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="AskDocs RAG evaluation")
    parser.add_argument("--doc-id", required=True, help="Document ID to test")
    parser.add_argument("--dataset", default="employee_eval.json", help="Eval dataset (JSON)")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per question")
    parser.add_argument("--method", choices=METHODS, default=None, help="Run a single method (default: reranked)")
    parser.add_argument("--compare", action="store_true", help="Run all methods and compare")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Retrieval-only evaluation (no API keys needed)")
    parser.add_argument("--ragas-metrics", nargs="+", choices=SUPPORTED_RAGAS_METRICS,
                        default=None, metavar="METRIC",
                        help=f"Ragas metrics to run (default: {' '.join(DEFAULT_RAGAS_METRICS)})")
    parser.add_argument("--custom-judge", action="store_true",
                        help="Also run the legacy custom LLM-as-judge")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N questions (0 = all); useful for smoke tests")
    args = parser.parse_args()

    if args.compare:
        methods = METHODS
    else:
        methods = [args.method or "reranked"]

    results = run_evaluation(
        doc_id=args.doc_id,
        dataset_path=args.dataset,
        top_k=args.top_k,
        methods=methods,
        skip_generation=args.skip_generation,
        ragas_metrics=args.ragas_metrics,
        custom_judge=args.custom_judge,
        limit=args.limit,
    )

    print_comparison(results)

    total_usage = {
        "answer_llm_calls": sum(m["api_usage"]["answer_llm_calls"] for m in results["results"].values()),
        "ragas_judge_evaluations": sum(m["api_usage"]["ragas_judge_evaluations"] for m in results["results"].values()),
        "custom_judge_calls": sum(m["api_usage"]["custom_judge_calls"] for m in results["results"].values()),
    }
    print(f"\n  API usage: {total_usage}")
    print(f"  Results saved to: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
