import logging

from sentence_transformers import SentenceTransformer

from app.core.exceptions import EmbeddingGenerationException

logger = logging.getLogger(__name__)


def create_embeddings(
    texts: list[str], embeddings_model: SentenceTransformer
) -> list[list[float]]:
    """
    Transforma una lista de textos en sus representaciones vectoriales (embeddings).

    Args:
        texts (list[str]): Textos a procesar.
        embeddings_model (SentenceTransformer): Modelo de SentenceTransformers cargado.

    Returns:
        list[list[float]]: Lista de vectores numéricos, uno por cada texto de entrada.

    Raises:
        EmbeddingGenerationException: Si ocurre un error durante la generación de embeddings.
    """
    try:
        logger.info(f"Iniciando embeddings. A vectorizar: {len(texts)}")
        embeddings_encode = embeddings_model.encode(texts).tolist()
        logger.info(f"Embeddings finalizado. {len(embeddings_encode)} creados.")
        return embeddings_encode
    except Exception as e:
        error_msg = f"Error al generar representaciones vectoriales: {type(e).__name__} - {str(e)}"
        logger.error(error_msg)
        raise EmbeddingGenerationException(error_msg)
