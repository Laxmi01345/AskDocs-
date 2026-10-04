"""
Generation Validation - checks if the LLM answer is correct and faithful.

Two evaluation modes:
    - validate_generation(): custom LLM-as-judge (legacy, behind --custom-judge)
    - validate_generation_ragas(): Ragas-based metrics (default)

Ragas imports are lazy so retrieval-only runs never need them.
"""
from typing import Dict, List, Optional


def validate_generation(
    question: str,
    answer: str,
    context: str,
    ground_truth: str = None,
) -> Dict:
    """
    Validate generation quality with a custom LLM-as-judge.

    Legacy mode, enabled via --custom-judge. Produces overall_score and
    custom_overall_generation_score in validate.py.

    Args:
        question: The user's question
        answer: The LLM's answer
        context: The retrieved context used to generate the answer
        ground_truth: Known correct answer (optional)
    """
    from app.llm import CerebrasLLM

    results = {
        "answer": answer,
        "metrics": {},
    }

    llm = CerebrasLLM(max_tokens=200)

    # 1. Faithfulness - is the answer grounded in the context?
    faith_prompt = f"""Evaluate if this answer is faithful to the given context.

Context: {context[:1500]}
Answer: {answer}

Return ONLY a JSON object with:
- "score": 0.0 to 1.0 (1.0 = fully grounded in context)
- "supported_claims": list of claims supported by context
- "unsupported_claims": list of claims NOT found in context

JSON:"""

    try:
        faith_response = llm.invoke(faith_prompt)
        import json
        # Extract JSON from response
        start = faith_response.find("{")
        end = faith_response.rfind("}") + 1
        if start != -1 and end > start:
            faith_data = json.loads(faith_response[start:end])
            results["metrics"]["faithfulness"] = faith_data.get("score", 0)
            results["metrics"]["supported_claims"] = faith_data.get("supported_claims", [])
            results["metrics"]["unsupported_claims"] = faith_data.get("unsupported_claims", [])
        else:
            results["metrics"]["faithfulness"] = None
    except Exception as e:
        results["metrics"]["faithfulness"] = None
        results["metrics"]["faithfulness_error"] = str(e)

    # 2. Answer Relevancy - does the answer address the question?
    relevancy_prompt = f"""Evaluate if this answer addresses the question.

Question: {question}
Answer: {answer}

Return ONLY a JSON object with:
- "score": 0.0 to 1.0 (1.0 = perfectly relevant)
- "reason": brief explanation

JSON:"""

    try:
        relev_response = llm.invoke(relevancy_prompt)
        import json
        start = relev_response.find("{")
        end = relev_response.rfind("}") + 1
        if start != -1 and end > start:
            relev_data = json.loads(relev_response[start:end])
            results["metrics"]["answer_relevancy"] = relev_data.get("score", 0)
            results["metrics"]["relevancy_reason"] = relev_data.get("reason", "")
        else:
            results["metrics"]["answer_relevancy"] = None
    except Exception as e:
        results["metrics"]["answer_relevancy"] = None
        results["metrics"]["relevancy_error"] = str(e)

    # 3. Hallucination Check - did the LLM invent facts?
    hallucination_prompt = f"""Check if this answer contains hallucinations (facts not in the context).

Context: {context[:1500]}
Answer: {answer}

Return ONLY a JSON object with:
- "hallucination_score": 0.0 to 1.0 (0.0 = no hallucination, 1.0 = fully hallucinated)
- "hallucinated_facts": list of facts in answer NOT found in context

JSON:"""

    try:
        halluc_response = llm.invoke(hallucination_prompt)
        import json
        start = halluc_response.find("{")
        end = halluc_response.rfind("}") + 1
        if start != -1 and end > start:
            halluc_data = json.loads(halluc_response[start:end])
            results["metrics"]["hallucination_score"] = halluc_data.get("hallucination_score", 0)
            results["metrics"]["hallucinated_facts"] = halluc_data.get("hallucinated_facts", [])
        else:
            results["metrics"]["hallucination_score"] = None
    except Exception as e:
        results["metrics"]["hallucination_score"] = None
        results["metrics"]["hallucination_error"] = str(e)

    # 4. Ground Truth Comparison (if provided)
    if ground_truth:
        gt_prompt = f"""Compare this answer to the ground truth.

Answer: {answer}
Ground Truth: {ground_truth}

Return ONLY a JSON object with:
- "similarity": 0.0 to 1.0 (1.0 = identical meaning)
- "key_facts_matched": list of key facts from ground truth found in answer
- "key_facts_missed": list of key facts from ground truth NOT in answer

JSON:"""

        try:
            gt_response = llm.invoke(gt_prompt)
            import json
            start = gt_response.find("{")
            end = gt_response.rfind("}") + 1
            if start != -1 and end > start:
                gt_data = json.loads(gt_response[start:end])
                results["metrics"]["ground_truth_similarity"] = gt_data.get("similarity", 0)
                results["metrics"]["key_facts_matched"] = gt_data.get("key_facts_matched", [])
                results["metrics"]["key_facts_missed"] = gt_data.get("key_facts_missed", [])
        except Exception as e:
            results["metrics"]["ground_truth_error"] = str(e)

    # 5. Compute Overall Score
    faith = results["metrics"].get("faithfulness")
    relev = results["metrics"].get("answer_relevancy")
    halluc = results["metrics"].get("hallucination_score")

    scores = []
    if faith is not None:
        scores.append(faith)
    if relev is not None:
        scores.append(relev)
    if halluc is not None:
        scores.append(1.0 - halluc)  # Invert: low hallucination = high score

    results["metrics"]["overall_score"] = round(sum(scores) / len(scores), 4) if scores else None

    # 6. Verdict
    overall = results["metrics"]["overall_score"]
    if overall and overall >= 0.8:
        results["verdict"] = "EXCELLENT"
    elif overall and overall >= 0.6:
        results["verdict"] = "GOOD"
    elif overall and overall >= 0.4:
        results["verdict"] = "FAIR"
    else:
        results["verdict"] = "NEEDS_REVIEW"

    return results


def validate_generation_ragas(
    samples: List[Dict],
    metric_names: Optional[List[str]] = None,
    show_progress: bool = False,
) -> Dict:
    """
    Evaluate RAG answers with Ragas (default evaluation mode).

    Runs one ragas.evaluate() call over all samples (answers are generated
    once by validate.py before this call - no duplicate LLM generation).

    Args:
        samples: list of dicts with keys:
            question (str), answer (str), contexts (list[str]),
            ground_truth (str | None)
        metric_names: subset of SUPPORTED_RAGAS_METRICS; defaults to the
            3 core metrics (faithfulness, answer_relevancy,
            answer_correctness).
        show_progress: ragas progress bar (off by default).

    Returns:
        {
          "metrics": {name: mean_score_or_None},   # mean over samples
          "per_sample": [{"question": str, "metrics": {name: score_or_None}}],
          "errors": [str],
        }

    Raises:
        ValueError: on unknown metric names.
        RuntimeError: if GROQ_API_KEY is missing (judge LLM required).
    """
    import asyncio
    import math

    from app.validation.ragas_compat import (
        DEFAULT_RAGAS_METRICS,
        SUPPORTED_RAGAS_METRICS,
        build_ragas_embeddings,
        build_ragas_llm,
        build_ragas_metrics,
        ensure_ragas_ready,
    )

    errors = []
    # Skip samples without an answer - Ragas fails on None responses.
    total_in = len(samples)
    samples = [s for s in samples if s.get("answer")]
    if total_in > len(samples):
        errors.append(f"{total_in - len(samples)} sample(s) skipped (no answer)")
    if not samples:
        return {"metrics": {}, "per_sample": [], "errors": errors or ["no samples to evaluate"]}

    metric_names = list(metric_names or DEFAULT_RAGAS_METRICS)
    unknown = [n for n in metric_names if n not in SUPPORTED_RAGAS_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown Ragas metric(s): {unknown}. "
            f"Supported: {SUPPORTED_RAGAS_METRICS}"
        )

    ensure_ragas_ready()

    from datasets import Dataset
    from ragas import aevaluate

    rows = [
        {
            "user_input": s["question"],
            "response": s["answer"],
            "retrieved_contexts": list(s["contexts"]),
            "reference": s.get("ground_truth"),
        }
        for s in samples
    ]
    dataset = Dataset.from_list(rows)

    llm = build_ragas_llm()
    needs_embeddings = bool(
        set(metric_names) & {"answer_relevancy", "answer_correctness"}
    )
    embeddings = build_ragas_embeddings() if needs_embeddings else None
    metrics = build_ragas_metrics(metric_names, llm, embeddings)

    result = asyncio.run(
        aevaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=show_progress,
        )
    )

    per_sample = []
    for i, row in enumerate(result.scores):
        sample_metrics = {}
        for name in metric_names:
            value = row.get(name)
            if isinstance(value, float) and math.isnan(value):
                value = None
            elif isinstance(value, (int, float)):
                value = round(float(value), 4)
            sample_metrics[name] = value
        per_sample.append(
            {"question": samples[i]["question"], "metrics": sample_metrics}
        )

    for name in metric_names:
        scores = [ps["metrics"][name] for ps in per_sample]
        numeric = [v for v in scores if v is not None]
        if not numeric:
            errors.append(f"metric '{name}' produced no scores")

    aggregate = {}
    for name in metric_names:
        numeric = [
            ps["metrics"][name]
            for ps in per_sample
            if ps["metrics"][name] is not None
        ]
        aggregate[name] = (
            round(sum(numeric) / len(numeric), 4) if numeric else None
        )

    return {"metrics": aggregate, "per_sample": per_sample, "errors": errors}
