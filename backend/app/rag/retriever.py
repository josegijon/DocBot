"""
Módulo para la recuperación de contexto (RAG).

Se encarga de conectarse a la base de datos vectorial (ChromaDB) y realizar
una búsqueda semántica para obtener los fragmentos de texto más relevantes
a partir de la consulta del usuario.
"""

import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.rag.embeddings import create_embeddings
from app.core.exceptions import EmptyQueryError, VectorStoreException
from app.rag.chroma_client import get_chroma_client


logger = logging.getLogger(__name__)

_SCORE_ROUNDING_DECIMALS = 4


def retrieve(
    query: str, doc_id: str, embeddings_model: SentenceTransformer
) -> list[dict[str, Any]]:
    """
    Recupera los fragmentos de documento más relevantes para una consulta dada.

    Genera un embedding a partir de la consulta del usuario y realiza una búsqueda de similitud en la colección de ChromaDB asociada al `doc_id`.

    Args:
        query (str): La consulta o pregunta del usuario.
        doc_id (str): El identificador único del documento (que corresponde al nombre de la colección en ChromaDB).
        embeddings_model (SentenceTransformer): El modelo de embeddings utilizado para convertir la consulta en representación vectorial.

    Returns:
        list[dict[str, Any]]: Una lista de diccionarios con los metadatos de los fragmentos recuperados. Cada diccionario contiene:
            - 'text' (str): El contenido del fragmento de texto.
            - 'page' (int): El número de página de donde se extrajo.
            - 'score' (float): El nivel de similitud entre la consulta y el fragmento.

    Raises:
        EmptyQueryError: Si la consulta proporcionada está vacía o consiste únicamente de espacios en blanco.
        VectorStoreException: Si ocurre un error al acceder a la colección o al realizar la búsqueda semántica.
    """

    logger.info(f"Iniciando búsqueda semántica para {doc_id}")

    if not query.strip():
        logger.error("Query vacía o solo espacios en blanco.")
        raise EmptyQueryError(
            "La consulta proporcionada está vacía o contiene únicamente espacios en blanco. "
            "Por favor, proporcione una pregunta o mensaje válido."
        )

    client = get_chroma_client(doc_id)

    try:
        collection = client.get_collection(name=doc_id)
    except Exception as e:
        logger.error(f"Error en la colección del documento {doc_id}: {e}")
        raise VectorStoreException(f"Error en la colección del documento {doc_id}: {e}")

    query_vectors = create_embeddings([query], embeddings_model)

    try:
        results = collection.query(
            query_embeddings=[query_vectors],
            n_results=settings.N_RESULTS_RETRIEVE,
        )
    except Exception as e:
        logger.error(f"Error en la búsqueda semántica: {e}")
        raise VectorStoreException(f"Error en la búsqueda semántica: {e}")

    formatted_results = [
        {
            "text": text,
            "page": meta.get("page", 0),
            "score": round(1 - dist, _SCORE_ROUNDING_DECIMALS),
        }
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

    logger.info(
        f"Búsqueda completada: {len(formatted_results)} resultados recuperados para {doc_id}"
    )

    return formatted_results
