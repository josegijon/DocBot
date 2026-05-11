import logging

from app.core.exceptions import EmbeddingGenerationException

logger = logging.getLogger(__name__)


def create_embeddings(texts, embeddings_model):
    """
    Transforma una lista de textos en sus representaciones vectoriales (embeddings).

    Args:
        texts (list[str]): Textos a procesar.
        embeddings_model: Modelo de SentenceTransformers cargado.

    Returns:
        list[list[float]]: Lista de vectores numéricos.
    """
    try:
        return embeddings_model.encode(texts).tolist()
    except Exception as e:
        error_msg = f"Error al generar representaciones vectoriales: {type(e).__name__} - {str(e)}"
        logger.error(error_msg)
        raise EmbeddingGenerationException(error_msg)
