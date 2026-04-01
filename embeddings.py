"""
Local embedding model wrapper — replaces OpenAI text-embedding-3-small
with sentence-transformers/all-MiniLM-L6-v2 (runs locally, no API key needed).

Module 3: Attention, Transformers, Embeddings
"""
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


class LocalEmbeddings(Embeddings):
    """LangChain-compatible wrapper around a local SentenceTransformer model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_model()
        return model.encode(texts, show_progress_bar=len(texts) > 100).tolist()

    def embed_query(self, text: str) -> list[float]:
        model = _get_model()
        return model.encode([text], show_progress_bar=False)[0].tolist()
