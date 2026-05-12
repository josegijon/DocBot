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


def retrieve(
    query: str, doc_id: str, embeddings_model: SentenceTransformer
) -> list[dict[str, Any]]:
    """
    Recupera los fragmentos de documento más relevantes para una consulta dada.

    Genera un embedding a partir de la consulta del usuario y realiza una búsqueda
    de similitud en la colección de ChromaDB asociada al `doc_id`.

    Args:
        query (str): La consulta o pregunta del usuario.
        doc_id (str): El identificador único del documento (que corresponde al nombre de la colección en ChromaDB).
        embeddings_model (SentenceTransformer): El modelo de lenguaje utilizado para convertir la consulta en embeddings.

    Returns:
        list[dict[str, Any]]: Una lista de diccionarios con los metadatos de los fragmentos recuperados.
                              Cada diccionario contiene las claves:
                              - 'text' (str): El contenido del fragmento de texto.
                              - 'page' (int): El número de página de donde se extrajo.
                              - 'score' (float): El nivel de similitud entre la consulta y el fragmento.

    Raises:
        EmptyQueryError: Si la consulta proporcionada está vacía o consiste únicamente de espacios en blanco.
    """

    if not query.strip():
        logger.error("Query vacía o solo espacios en blanco.")
        raise EmptyQueryError(
            "La consulta proporcionada está vacía o contiene únicamente espacios en blanco. "
            "Por favor, proporcione una pregunta o mensaje válido."
        )

    client = get_chroma_client(doc_id)

    try:
        # Accede a la collección
        collection = client.get_collection(name=doc_id)
    except Exception as e:
        logger.error(f"Error en la colección del documento {doc_id}: {e}")
        raise VectorStoreException(f"Error en la colección del documento {doc_id}: {e}")

    # Genera el embeddings de la consulta
    query_vectors = create_embeddings([query], embeddings_model)

    try:
        # Realiza la búsqueda semántica, return [ [res1, res2...] ]
        results = collection.query(
            query_embeddings=[query_vectors],
            n_results=settings.N_RESULTS_RETRIEVE,
        )
    except Exception as e:
        logger.error(f"Error en la búsqueda semántica: {e}")
        raise VectorStoreException(f"Error en la búsqueda semántica: {e}")

    formatted_results = [
        {"text": text, "page": meta.get("page", 0), "score": round(1 - dist, 4)}
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

    return formatted_results
