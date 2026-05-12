import logging

from groq import AsyncGroq
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)


def load_embeddings_model() -> SentenceTransformer:
    """
    Carga el modelo de embeddings SentenceTransformer en memoria.

    Utiliza el nombre de modelo definido en la configuración
    (`settings.EMBEDDINGS_MODEL_NAME`). Si el modelo no está disponible
    localmente, se descarga automáticamente desde Hugging Face Hub.

    Returns:
        SentenceTransformer: Instancia del modelo de embeddings cargado.

    Raises:
        OSError: Si no es posible descargar o cargar el modelo desde
            Hugging Face Hub.
    """
    logger.info("Cargando modelo de embeddings...")
    embeddings_model = SentenceTransformer(settings.EMBEDDINGS_MODEL_NAME)
    logger.info("Modelo de embeddings cargado correctamente.")
    return embeddings_model


def load_groq_client() -> AsyncGroq:
    """
    Inicializa el cliente asíncrono de Groq para la comunicación con la API.

    Utiliza la clave de API definida en la configuración
    (`settings.GROQ_API_KEY`).

    Returns:
        AsyncGroq: Instancia del cliente asíncrono de Groq inicializada.
    """
    logger.info("Inicializando cliente Groq.")
    async_groq = AsyncGroq(api_key=settings.GROQ_API_KEY)
    logger.info("Cliente Groq inicializado correctamente.")
    return async_groq


def load_reranker() -> CrossEncoder:
    """
    Carga el modelo de reranking CrossEncoder en memoria.

    Utiliza el nombre de modelo definido en la configuración
    (`settings.RERANKER_MODEL_NAME`). Si el modelo no está disponible
    localmente, se descarga automáticamente desde Hugging Face Hub.

    Returns:
        CrossEncoder: Instancia del modelo de reranking cargado.

    Raises:
        OSError: Si no es posible descargar o cargar el modelo desde
            Hugging Face Hub.
    """
    logger.info("Cargando modelo de reranking...")
    reranking_model = CrossEncoder(settings.RERANKER_MODEL_NAME)
    logger.info("Modelo de reranking cargado correctamente.")
    return reranking_model
