"""
Local embedding model wrappers for RAG retrieval.

Provides multiple embedding models for comparison:
  - all-MiniLM-L6-v2: fast, lightweight (22M params, 384-dim) — default
  - all-mpnet-base-v2: more accurate (109M params, 768-dim) — higher quality
  - OpenAI text-embedding-3-small: API-based (1536-dim) — configured separately

Module 3: Attention, Transformers, Embeddings
"""
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

_models = {}


def _get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


class LocalEmbeddings(Embeddings):
    """LangChain-compatible wrapper around a local SentenceTransformer model.

    Default: all-MiniLM-L6-v2 (384-dim, fast, lightweight).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_model(self.model_name)
        return model.encode(texts, show_progress_bar=len(texts) > 100).tolist()

    def embed_query(self, text: str) -> list[float]:
        model = _get_model(self.model_name)
        return model.encode([text], show_progress_bar=False)[0].tolist()


class MPNetEmbeddings(Embeddings):
    """LangChain-compatible wrapper around all-mpnet-base-v2.

    Larger model (109M params, 768-dim) — more accurate than MiniLM
    but slower. Used for embedding comparison experiments.
    """

    def __init__(self):
        self.model_name = "sentence-transformers/all-mpnet-base-v2"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_model(self.model_name)
        return model.encode(texts, show_progress_bar=len(texts) > 100).tolist()

    def embed_query(self, text: str) -> list[float]:
        model = _get_model(self.model_name)
        return model.encode([text], show_progress_bar=False)[0].tolist()
