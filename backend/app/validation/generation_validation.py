"""
Generation Validation - checks if the LLM answer is correct and faithful.
"""
from typing import Dict, List
from app.llm import CerebrasLLM


def validate_generation(
    question: str,
    answer: str,
    context: str,
    ground_truth: str = None,
) -> Dict:
    """
    Validate generation quality.

    Args:
        question: The user's question
        answer: The LLM's answer
        context: The retrieved context used to generate the answer
        ground_truth: Known correct answer (optional)
    """
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
