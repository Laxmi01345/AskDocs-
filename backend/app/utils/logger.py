import json

LOG_FILE = "rag_logs.jsonl"


def log_rag_interaction(doc_id, question, answer, retrieved_chunks):
    log_entry = {
        "doc_id": doc_id,
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
