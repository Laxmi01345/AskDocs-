import os
from dotenv import load_dotenv
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    return _cross_encoder
