from groq import AsyncGroq
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from app.core.config import settings


def load_embeddings_model() -> SentenceTransformer:
    """
    Carga el modelo de embeddings en memoria.
    """
    return SentenceTransformer(settings.EMBEDDINGS_MODEL_NAME)


def load_groq_client() -> AsyncGroq:
    """
    Inicializa el cliente asíncrono de Groq.
    """
    return AsyncGroq(api_key=settings.GROQ_API_KEY)


def load_reranker() -> CrossEncoder:
    """
    Carga el modelo de reranking en memoria.
    """
    return CrossEncoder(settings.RERANKER_MODEL_NAME)