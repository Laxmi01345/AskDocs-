import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a, b):
    import numpy as np
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def chunk_semantic(text: str, max_chunk_size: int = 1500, breakpoint_threshold: float = 0.5) -> List[str]:
    from app.embeddings import get_embeddings

    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return [text]

    embeddings = get_embeddings()
    sentence_embs = embeddings.embed_documents(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(sentence_embs[i - 1], sentence_embs[i])
        current_text = " ".join(current_chunk) + " " + sentences[i]

        if sim < breakpoint_threshold or len(current_text) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]
