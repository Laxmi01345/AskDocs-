import os
import psycopg
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")


def init_db():
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT UNIQUE,
                    filename TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
                    content TEXT,
                    embedding TEXT
                );
            """)
        conn.commit()


def store_document(doc_id: str, filename: str, chunks: list[str], embeddings: list):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # Insert document
            cur.execute(
                "INSERT INTO documents (doc_id, filename) VALUES (%s, %s) ON CONFLICT (doc_id) DO NOTHING",
                (doc_id, filename),
            )
            
            # Insert chunks
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    "INSERT INTO chunks (doc_id, content, embedding) VALUES (%s, %s, %s)",
                    (doc_id, chunk, json.dumps(emb.tolist())),
                )
        conn.commit()


def retrieve_chunks(doc_id: str, question_emb: list, top_k: int = 3):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content, embedding FROM chunks WHERE doc_id = %s",
                (doc_id,),
            )
            rows = cur.fetchall()
    
    if not rows:
        return []
    
    # Compute similarity in Python
    scores = []
    question_emb_np = np.array(question_emb)
    for content, emb_json in rows:
        emb = np.array(json.loads(emb_json))
        sim = np.dot(question_emb_np, emb) / (np.linalg.norm(question_emb_np) * np.linalg.norm(emb))
        scores.append({"content": content, "score": float(sim)})
    
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]