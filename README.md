# AskDocs – RAG-based Document Q&A System

AskDocs is a Retrieval-Augmented Generation (RAG) based Document Question Answering system.

It allows users to upload a text document, generate semantic embeddings, retrieve relevant context using vector search, and generate grounded answers using Cerebras AI.


## 🚀 Features

- 📄 Upload text documents
- 🧠 Generate embeddings
- 🔎 Semantic search over document chunks
- 🤖 Context-aware answer generation (Cerebras AI)
- ⚡ FastAPI backend
- 💻 Modern frontend


## 🏗️ Architecture

Upload → Chunk → Embed → Store → Retrieve → Generate

1. Document is split into chunks
2. Embeddings are generated
3. Stored in vector store
4. Relevant chunks retrieved using semantic similarity
5. Sent to LLM (Cerebras AI) for answer generation


## 🛠️ Tech Stack

- FastAPI
- Sentence Transformers
- Vector Search
- Cerebras AI
- React / Vite (if used)
- Tailwind CSS (if used)


## 🎥 Demo
https://github.com/user-attachments/assets/3e2d3ca6-abd3-4b1e-a1d2-4f08ee27dcba
