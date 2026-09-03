# AskDocs v2 – Multi-Method RAG Document Q&A

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
- **Comparative Evaluation** – Recall@5, MRR, Correctness, Faithfulness

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
│   │       ├── retrieval_validation.py
│   │       └── generation_validation.py
│   ├── validate.py              # Comparative evaluation CLI
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

### Running Validation

```bash
cd backend

# Single method
python validate.py --doc-id YOUR_DOC_ID --method simple

# Compare all methods
python validate.py --doc-id YOUR_DOC_ID --compare
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Recall@5** | % of ground truth chunks found in retrieved results |
| **MRR** | Mean Reciprocal Rank (position of first relevant result) |
| **Correctness** | LLM-as-judge score for answer accuracy |
| **Faithfulness** | % of answer claims supported by context |

### Sample Results

| Method | Recall@5 | MRR | Correctness | Faithfulness |
|--------|----------|-----|-------------|--------------|
| Simple | 90.0% | 0.83 | 80.8% | 76.0% |
| Semantic | 90.0% | 0.83 | 74.8% | 66.0% |
| Hybrid | **100.0%** | **0.90** | 75.8% | 70.0% |
| Reranked | 90.0% | 0.83 | **82.7%** | 68.0% |

**Improvement over previous method:**
- Semantic vs Simple: +0% Recall, +0% MRR, -7.4% Correctness, -13.2% Faithfulness
- Hybrid vs Semantic: +11.1% Recall, +8.3% MRR, +1.3% Correctness, +6.1% Faithfulness
- Reranked vs Hybrid: -11.1% Recall, -7.8% MRR, +9.1% Correctness, -2.9% Faithfulness

**Best method by metric:**
- Highest Recall: Hybrid (100%)
- Highest MRR: Hybrid (0.90)
- Highest Correctness: Reranked (82.7%)
- Highest Faithfulness: Simple (76.0%)

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
