import logging

from groq import AsyncGroq
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)


def load_embeddings_model() -> SentenceTransformer:
    """
    Carga el modelo de embeddings en memoria.
    """
    logger.info("Cargando modelo de embeddings...")
    embeddings_model = SentenceTransformer(settings.EMBEDDINGS_MODEL_NAME)
    logger.info("Modelo de embeddings cargado correctamente.")
    return embeddings_model


def load_groq_client() -> AsyncGroq:
    """
    Inicializa el cliente asíncrono de Groq.
    """
    logger.info("Inicializando cliente Groq.")
    async_groq = AsyncGroq(api_key=settings.GROQ_API_KEY)
    logger.info("Cliente Groq inicializado correctamente.")
    return async_groq


def load_reranker() -> CrossEncoder:
    """
    Carga el modelo de reranking en memoria.
    """
    logger.info("Cargando modelo de reranking...")
    reranking_model = CrossEncoder(settings.RERANKER_MODEL_NAME)
    logger.info("Modelo de reranking cargado correctamente.")
    return reranking_model
