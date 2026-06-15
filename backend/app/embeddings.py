import os
from dotenv import load_dotenv

load_dotenv()

_embeddings = None


class ChromaONNXEmbeddings:
    """Lightweight ONNX-based embeddings (no torch required, ~50MB vs 500MB+)."""

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._ef = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts):
        return self._ef(list(texts))

    def embed_query(self, text):
        return self._ef([text])[0]


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = ChromaONNXEmbeddings()
    return _embeddings
