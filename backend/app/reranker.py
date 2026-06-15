import os
from dotenv import load_dotenv

load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() in ("true", "1", "yes")

_cross_encoder = None
_reranker_available = None


def get_cross_encoder():
    global _cross_encoder, _reranker_available
    if _reranker_available is False:
        return None
    if _cross_encoder is None:
        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            _cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
            _reranker_available = True
        except Exception:
            _reranker_available = False
            return None
    return _cross_encoder
