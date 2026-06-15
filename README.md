# AskDocs – RAG-based Document Q&A System

Built a RAG-based system for querying and extracting answers from documents.

- Implemented semantic search using embeddings and vector database
- Integrated LLMs for context-aware response generation
- Tech: Python, LLMs, Vector DB, Embeddings

AskDocs is a Retrieval-Augmented Generation (RAG) based Document Question Answering system.

It allows users to upload documents (PDF, DOCX, TXT), generate semantic embeddings, retrieve relevant context using hybrid search, and generate grounded answers using Cerebras AI.

**Live Demo:** [https://askdocs-1.onrender.com](https://askdocs-1.onrender.com)

---

## 🚀 Features

- 📄 **Document Upload** – Supports PDF, DOCX, and TXT files
- 🧠 **Hybrid Retrieval** – Combines BM25 keyword search + vector similarity search with Reciprocal Rank Fusion (RRF)
- 🔎 **Semantic Search** – ChromaDB with ONNX-based embeddings (all-MiniLM-L6-v2)
- 🤖 **Context-Aware Answers** – Cerebras AI LLM with RAG prompt assembly
- 💬 **Conversational Memory** – Multi-turn sessions with automatic summarization and sliding window
- ✅ **Validation Layer** – Retrieval validation + generation validation for answer quality
- 🎯 **Reranking** – Optional cross-encoder reranking for improved precision
- 💻 **Modern Frontend** – React + Vite + Tailwind CSS with markdown rendering

---

## 🏗️ Architecture

```
Upload → Parse → Chunk → Embed → Store
                                          ↓
Question → Hybrid Retrieve (BM25 + Vector) → RRF Fusion → [Optional Rerank] → Context Assembly → Cerebras LLM → Answer
```

### Pipeline Steps

1. **Parse** – Extract text from PDF/DOCX/TXT using LangChain document loaders
2. **Chunk** – Split into overlapping chunks using `RecursiveCharacterTextSplitter`
3. **Embed** – Generate embeddings with ChromaDB ONNX embeddings (all-MiniLM-L6-v2)
4. **Store** – Persist in ChromaDB vector store + raw chunks for BM25
5. **Retrieve** – Hybrid search: BM25 keyword matching + vector similarity
6. **Fuse** – Reciprocal Rank Fusion (RRF) merges ranked results from both retrievers
7. **Rerank** – Optional cross-encoder reranking for top precision
8. **Generate** – Assemble RAG prompt with conversation history and send to Cerebras AI

---

## 🛠️ Tech Stack

### Backend
| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| LLM | Cerebras AI |
| Embeddings | ChromaDB ONNX (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Keyword Search | rank_bm25 |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
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

## 📁 Project Structure

```
AskDocs-
├── backend/
│   ├── main.py                  # FastAPI routes + CORS + startup
│   ├── app/
│   │   ├── embeddings.py        # ChromaDB ONNX embeddings
│   │   ├── database.py          # ChromaDB vector store + chunk persistence
│   │   ├── retrieval.py         # Hybrid retrieval + RRF fusion + reranking
│   │   ├── bm25_store.py        # BM25 index build/cache/persist
│   │   ├── reranker.py          # Cross-encoder reranker (optional)
│   │   ├── chunking.py          # Text chunking
│   │   ├── parsing.py           # Document parsing (PDF/DOCX/TXT)
│   │   ├── llm.py               # Cerebras LLM wrapper
│   │   ├── session.py           # Session manager (TTL, LRU eviction)
│   │   ├── context_builder.py   # RAG prompt assembly with history
│   │   └── validation/          # Retrieval + generation validation
│   ├── Dockerfile               # Docker build config
│   ├── requirements.txt         # Python dependencies
│   ├── demo.txt                 # Sample document
│   ├── employee_eval.json       # Evaluation dataset
│   └── validate.py              # Validation script
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app (Upload → Chat flow)
│   │   ├── api.js               # API base URL config
│   │   └── components/
│   │       ├── Upload.jsx       # File upload
│   │       └── Chat.jsx         # Chat with sessions + markdown
│   ├── package.json
│   └── vite.config.js
└── render.yaml                  # Render Blueprint config
```

---

## 🧪 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/documents` | List all uploaded documents |
| `POST` | `/upload` | Upload a document (PDF/DOCX/TXT) |
| `POST` | `/ask` | Ask a question about a document |

### Example: Upload

```bash
curl -X POST https://askdocs-1.onrender.com/upload -F "file=@demo.txt"
```

Response:
```json
{
  "doc_id": "3da0ad63-71b9-47e9-b4a0-b0ef995839cc",
  "filename": "demo.txt",
  "chunks": 5
}
```

### Example: Ask

```bash
curl -X POST https://askdocs-1.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"3da0ad63-71b9-47e9-b4a0-b0ef995839cc","question":"What is the leave policy?"}'
```

Response:
```json
{
  "doc_id": "3da0ad63-71b9-47e9-b4a0-b0ef995839cc",
  "question": "What is the leave policy?",
  "answer": "The leave policy provides: PTO – 20 days per year...",
  "retrieved_chunks": [...],
  "session_id": "792ccec4-55a4-418f-ab1d-583221868b08"
}
```

---

## 📊 Evaluation

AskDocs includes a validation layer for retrieval and generation quality.

- **Dataset:** `backend/employee_eval.json` (15 Q&A pairs)
- **CLI:** `python validate.py` from the `backend/` folder

The validation scores retrieved chunks and generated answers against ground truth.

---

## 🚀 Local Development

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

Frontend runs on `http://localhost:5173` and connects to backend at `http://localhost:8000`.

---

## 🚢 Deployment

### Backend (Render – Docker)

1. Push to GitHub
2. Render → New → Web Service
3. Connect repo, select `backend/` as root directory
4. Runtime: Docker
5. Add env vars: `CEREBRAS_API_KEY`, `ENABLE_RERANKER=false`
6. Deploy

### Frontend (Render – Static Site)

1. Render → New → Static Site
2. Connect repo, root directory: `frontend/`
3. Build command: `npm install && npm run build`
4. Publish directory: `dist`
5. Add env var: `VITE_API_URL=https://askdocs-1.onrender.com`
6. Deploy

Or use the `render.yaml` Blueprint to deploy both services at once.

---

## 🎥 Demo
https://github.com/user-attachments/assets/3e2d3ca6-abd3-4b1e-a1d2-4f08ee27dcba
