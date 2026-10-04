# AskDocs – Multi-Method RAG Document Q&A

AskDocs v2 is a Retrieval-Augmented Generation (RAG) system with 4 retrieval methods and comparative evaluation.

Upload documents, ask questions, and compare retrieval strategies: Simple Chunking, Semantic Chunking, Hybrid (BM25+Vector+RRF), and Reranking.

**Live Demo:** [https://askdocs-1.onrender.com](https://askdocs-2.onrender.com/)

---

## Features

- **4 Retrieval Methods** – Simple, Semantic, Hybrid, Reranked
- **Document Upload** – Supports PDF, DOCX, and TXT files
- **Vector Retrieval** – Embedding similarity search with ChromaDB
- **BM25 Retrieval** – Keyword-based search for exact term matching
- **Hybrid Fusion** – Reciprocal Rank Fusion (RRF) combining BM25 + Vector
- **Cross-Encoder Reranking** – Re-ranks candidates for precision
- **Semantic Chunking** – Topic-aware splitting using embedding similarity
- **Conversational Memory** – Multi-turn sessions with summarization
- **Comparative Evaluation** – Semantic Recall@5, MRR, NDCG@5 (local) + Ragas generation metrics (LLM judge)

---

## Architecture

```
Upload → Parse → Chunk → Embed → Store
                                     ↓
Question → [Simple | Semantic | Hybrid | Reranked] → Context Assembly → LLM → Answer
```

### Retrieval Methods

| Method | Strategy | How It Works |
|--------|----------|--------------|
| **Simple** | Vector only | Embed query, cosine similarity, top-k chunks |
| **Semantic** | Topic-aware chunks | Split at topic boundaries, then vector search |
| **Hybrid** | BM25 + Vector + RRF | Combine keyword + semantic results with fusion |
| **Reranked** | Vector + Cross-Encoder | Vector search → cross-encoder re-scores → top-k |

---

## Tech Stack

### Backend

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| LLM | Groq (gpt-oss-20b) + Cerebras fallback |
| Embeddings | ChromaDB ONNX (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| BM25 | rank-bm25 (Okapi BM25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Text Splitting | RecursiveCharacterTextSplitter + Semantic |
| Document Parsing | PyPDF, python-docx, Docx2txt |
| Evaluation | Ragas 0.4.3 (judge: qwen/qwen3.8-27b via Groq) |

### Frontend

| Component | Technology |
|-----------|-----------|
| Framework | React 19 + Vite |
| Styling | Tailwind CSS |
| Markdown | react-markdown + @tailwindcss/typography |

### Deployment

| Component | Technology |
|-----------|-----------|
| Backend | Render (Docker) |
| Frontend | Render (Static Site) |
| Container | Docker (python:3.10-slim) |

---

## Project Structure

```
AskDocs-
├── backend/
│   ├── main.py                  # FastAPI routes + CORS
│   ├── app/
│   │   ├── embeddings.py        # ChromaDB ONNX embeddings
│   │   ├── database.py          # ChromaDB vector store
│   │   ├── retrieval.py         # 4 retrieval methods
│   │   ├── bm25_store.py        # BM25 index build/save/load
│   │   ├── chunking.py          # Text + semantic chunking
│   │   ├── parsing.py           # Document parsing
│   │   ├── llm.py               # Groq LLM + Cerebras fallback
│   │   ├── reranker.py          # Cross-encoder reranker
│   │   ├── session.py           # Session manager
│   │   ├── context_builder.py   # RAG prompt assembly
│   │   └── validation/
│   │       ├── retrieval_validation.py   # local metrics incl. NDCG@5
│   │       ├── generation_validation.py  # Ragas + custom judge
│   │       └── ragas_compat.py           # Ragas offline/Groq adapters
│   ├── validate.py              # Evaluation CLI (retrieval + generation)
│   ├── employee_eval.json       # Evaluation dataset (10 Q&A)
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   └── components/
│   │       ├── Upload.jsx
│   │       └── Chat.jsx
│   └── package.json
└── render.yaml
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/documents` | List all uploaded documents |
| `POST` | `/upload` | Upload a document (PDF/DOCX/TXT) |
| `POST` | `/ask` | Ask a question (with `method` param) |

### Example: Ask with Method Selection

```bash
curl -X POST https://askdocs-1.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"YOUR_DOC_ID","question":"What is the leave policy?","method":"hybrid"}'
```

Methods: `simple`, `semantic`, `hybrid`, `reranked`

---

## Evaluation

Evaluation has two independent stages, run from a single entry point (`backend/validate.py`):

1. **Retrieval evaluation** – deterministic and local (no API keys): Semantic
   Recall@5, MRR, NDCG@5, Precision, Recall, F1, average relevance, diversity.
2. **Generation evaluation** – [Ragas](https://docs.ragas.io) LLM judge:
   faithfulness, answer_relevancy, answer_correctness by default; optional
   context_precision and context_recall. Answers are generated once per
   question and reused across metrics.

Ground-truth chunks are matched **semantically** (embedding cosine similarity
≥ 0.30), so "Semantic Recall@5" is not exact chunk-ID matching.

### Running Validation

```bash
cd backend

# Full run (retrieval + generation + Ragas), single method
python validate.py --doc-id YOUR_DOC_ID

# Compare all 4 retrieval methods
python validate.py --doc-id YOUR_DOC_ID --compare

# Retrieval only (local, no API keys needed)
python validate.py --doc-id YOUR_DOC_ID --skip-generation

# Select Ragas metrics (default: faithfulness answer_relevancy answer_correctness)
python validate.py --doc-id YOUR_DOC_ID --ragas-metrics faithfulness context_precision

# Also run the legacy custom LLM-as-judge (opt-in)
python validate.py --doc-id YOUR_DOC_ID --custom-judge

# Smoke test: first 2 questions only
python validate.py --doc-id YOUR_DOC_ID --limit 2
```

Generation evaluation requires `GROQ_API_KEY` in `backend/.env`. The judge
model defaults to `qwen/qwen3.8-27b` (override via `RAGAS_JUDGE_MODEL` /
`RAGAS_JUDGE_BASE_URL`).

### Metrics

| Stage | Metric | Description |
|-------|--------|-------------|
| Retrieval | **Semantic Recall@5** | % of ground-truth chunks semantically matched in top-5 |
| Retrieval | **MRR** | Reciprocal rank of the first matched chunk |
| Retrieval | **NDCG@5** | Ranking quality of the retrieved list (binary gains) |
| Retrieval | **Precision / Recall / F1** | Set overlap with ground truth |
| Retrieval | **Avg Relevance** | Mean cosine similarity between question and chunks |
| Retrieval | **Diversity** | 1 − mean pairwise similarity among retrieved chunks |
| Generation | **Faithfulness** | Answer claims supported by retrieved context (Ragas) |
| Generation | **Answer Relevancy** | Answer addresses the question (Ragas) |
| Generation | **Answer Correctness** | Answer matches the reference answer (Ragas) |
| Generation | **Context Precision / Recall** | Retrieval quality per Ragas (optional) |

### Output

Results are written to `backend/evaluation_results.json` (saved incrementally
after each method). Per method it contains `retrieval`, `generation` and
`api_usage` blocks, each with aggregate scores plus `per_question` details.

The dataset (`backend/employee_eval.json`) has 10 questions with reference
answers and ground-truth chunks – scores are indicative, not benchmark-grade.
Compare methods on the same dataset rather than reading absolute numbers.

---

## Local Development

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Deployment

### Backend (Render – Docker)

1. Push to GitHub
2. Render → New → Web Service
3. Connect repo, select `backend/` as root directory
4. Runtime: Docker
5. Add env vars: `GROQ_API_KEY`, `CEREBRAS_API_KEY`
6. Deploy

### Frontend (Render – Static Site)

1. Render → New → Static Site
2. Connect repo, root directory: `frontend/`
3. Build command: `npm install && npm run build`
4. Publish directory: `dist`
5. Add env var: `VITE_API_URL=https://askdocs-1.onrender.com`
6. Deploy

---

## Demo
https://github.com/user-attachments/assets/3e2d3ca6-abd3-4b1e-a1d2-4f08ee27dcba

## Evaluation
https://drive.google.com/file/d/1KubFFPPsLc0eOX43jrfx-CeBp-cCyJXM/view?usp=drive_link
