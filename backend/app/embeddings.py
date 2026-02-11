from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    return _model.encode(text)


def embed_chunks(chunks: list[str]):
    return [_model.encode(c) for c in chunks]
