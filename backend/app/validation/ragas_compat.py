"""
Ragas compatibility layer.

Bridges local/offline AskDocs components with the installed ragas 0.4.3:

1. VertexAI shim: ragas 0.4.3 imports `langchain_community.chat_models.vertexai`
   which was removed from langchain-community 0.4.2. Ragas only uses
   ChatVertexAI/VertexAI in isinstance checks, so harmless stub classes are
   installed into sys.modules before ragas is imported.
2. LocalONNXEmbedding: adapter around AskDocs' local ONNX MiniLM embeddings
   for ragas metrics that need embeddings (AnswerRelevancy, AnswerCorrectness).
3. build_ragas_llm: an Instructor-compatible LLM via llm_factory() pointing at
   the Groq OpenAI-compatible endpoint (modern metrics reject
   LangchainLLMWrapper).
4. build_ragas_metrics: constructs requested metric instances.

All heavy imports are lazy: importing this module never requires ragas,
an API key, or embeddings.
"""
import os
import sys
import types
from typing import List

# Default 3 core generation metrics (spec).
DEFAULT_RAGAS_METRICS = ["faithfulness", "answer_relevancy", "answer_correctness"]

# Additional metrics selectable via --ragas-metrics.
SUPPORTED_RAGAS_METRICS = DEFAULT_RAGAS_METRICS + ["context_precision", "context_recall"]

_RAGAS_READY = False
_EMBEDDING_CLS = None


def ensure_ragas_ready() -> None:
    """
    Apply compatibility shims. Idempotent.
    Must run before any `import ragas...`.
    """
    global _RAGAS_READY
    if _RAGAS_READY:
        return

    # --- Shim 1: langchain_community.chat_models.vertexai (removed module) ---
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        stub = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:
            """Stub: ragas only uses this class in isinstance checks."""

        stub.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = stub

    # --- Shim 2: langchain_community.llms.VertexAI (may also be removed) ---
    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except Exception:
        import langchain_community.llms as _llms

        if not hasattr(_llms, "VertexAI"):

            class VertexAI:
                """Stub: ragas only uses this class in isinstance checks."""

            _llms.VertexAI = VertexAI

    _RAGAS_READY = True


def build_ragas_embeddings():
    """
    Build an embedding instance backed by AskDocs' local ONNX MiniLM-L6-V2.

    Subclasses ragas' BaseRagasEmbedding only after the shims are applied.
    No API keys needed. Returns an instance of BaseRagasEmbedding.
    """
    global _EMBEDDING_CLS
    ensure_ragas_ready()

    from ragas.embeddings.base import BaseRagasEmbedding

    if _EMBEDDING_CLS is None:

        class LocalONNXEmbedding(BaseRagasEmbedding):
            """Ragas embedding adapter over local ONNX MiniLM (offline)."""

            def __init__(self, cache=None):
                super().__init__(cache=cache)
                self._local = None

            @property
            def local(self):
                if self._local is None:
                    from app.embeddings import get_embeddings

                    self._local = get_embeddings()
                return self._local

            def embed_text(self, text: str, **kwargs) -> List[float]:
                return [float(x) for x in self.local.embed_query(text)]

            async def aembed_text(self, text: str, **kwargs) -> List[float]:
                return self.embed_text(text, **kwargs)

            def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
                return [
                    [float(x) for x in v]
                    for v in self.local.embed_documents(list(texts))
                ]

            async def aembed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
                return self.embed_texts(texts, **kwargs)

            # LangChain-style aliases: ragas' AnswerRelevancy calls
            # embeddings.embed_query() directly (duck-typed), so provide them.
            def embed_query(self, text: str) -> List[float]:
                return self.embed_text(text)

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.embed_texts(texts)

        _EMBEDDING_CLS = LocalONNXEmbedding

    return _EMBEDDING_CLS()


def build_ragas_llm():
    """
    Build an Instructor-compatible Ragas judge LLM on the Groq endpoint.

    Returns an InstructorBaseRagasLLM (NOT a LangchainLLMWrapper - modern
    ragas metrics raise ValueError for the wrapper).

    Env:
        GROQ_API_KEY      (required)
        RAGAS_JUDGE_MODEL (optional, default qwen/qwen3.8-27b)

    Note: the default judge deliberately differs from the answer model
    (gpt-oss-20b) to avoid self-preference bias, and stays within Groq's
    free-tier daily token budget for a full --compare run (~180k tokens).
    """
    ensure_ragas_ready()

    from dotenv import load_dotenv
    from openai import OpenAI
    from ragas.llms import llm_factory

    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required for generation evaluation (Ragas judge)."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("RAGAS_JUDGE_BASE_URL", "https://api.groq.com/openai/v1"),
    )
    model = os.environ.get("RAGAS_JUDGE_MODEL", "qwen/qwen3.8-27b")

    return llm_factory(model, provider="openai", client=client, temperature=0)


def build_ragas_metrics(names: List[str], llm, embeddings):
    """
    Construct requested Ragas metric instances.

    Args:
        names: metric names from SUPPORTED_RAGAS_METRICS
        llm: result of build_ragas_llm()
        embeddings: result of build_ragas_embeddings()

    Returns:
        list of metric instances ready for ragas.evaluate()
    """
    ensure_ragas_ready()

    from ragas.metrics._answer_correctness import AnswerCorrectness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    unknown = [n for n in names if n not in SUPPORTED_RAGAS_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown Ragas metric(s): {unknown}. "
            f"Supported: {SUPPORTED_RAGAS_METRICS}"
        )

    builders = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "answer_correctness": lambda: AnswerCorrectness(llm=llm, embeddings=embeddings),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
    }
    return [builders[name]() for name in names]
