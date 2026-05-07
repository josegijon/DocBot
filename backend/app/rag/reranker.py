import logging
from typing import Any

from sentence_transformers import CrossEncoder

from app.core.exceptions import NoChunksFoundException
from app.core.config import settings

logger = logging.getLogger(__name__)

def rerank(model: CrossEncoder, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reordena (rerank) una lista de fragmentos de texto (chunks) basándose en su relevancia
    para una consulta dada utilizando un modelo CrossEncoder.

    Args:
        model (CrossEncoder): El modelo CrossEncoder utilizado para predecir las puntuaciones de relevancia.
        query (str): La consulta o pregunta del usuario.
        chunks (list[dict[str, Any]]): Una lista de diccionarios, donde cada diccionario representa
            un fragmento de texto y contiene al menos la clave "text" con el contenido del fragmento.

    Returns:
        list[dict[str, Any]]: Los `RERANKER_TOP_K` mejores fragmentos ordenados por relevancia
        descendente. Cada diccionario de chunk devuelto incluye su puntuación actualizada
        bajo la clave "score".

    Raises:
        NoChunksFoundException: Si la lista de chunks proporcionada está vacía.
    """
    
    if not chunks:
        logger.error(
            "El reranker recibió una lista vacía de chunks. "
            "Verifique que el retriever haya encontrado coincidencias relevantes."
        )
        raise NoChunksFoundException(
            "No se encontraron chunks para reordenar. "
            "El retriever no devolvió resultados o el filtro de similitud eliminó todos los candidatos."
        )
    # Prepara los pares para el modelo
    pairs = [[query, chunk["text"]] for chunk in chunks]

    # Predice scores. Devuelve array de numpy con las puntuaciones
    scores = model.predict(pairs)

    # Empareja chunks con su nuevos scores y ordena
    ranked_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    # Actualiza el score y devuelve el RERANKER_TOP_K
    top_chunks = []
    for chunk, score in ranked_chunks[:settings.RERANKER_TOP_K]:
        chunk["score"] = round(float(score), 4)
        top_chunks.append(chunk)

    return top_chunks
