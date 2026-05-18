from fastapi import Request
from groq import AsyncGroq
from sentence_transformers import CrossEncoder, SentenceTransformer


def get_embeddings_model(request: Request) -> SentenceTransformer:
    """
    Obtiene el modelo de embeddings desde el estado de la aplicación.

    Args:
        request (Request): Objeto de solicitud de FastAPI.

    Returns:
        SentenceTransformer: Modelo de embeddings cargado en el estado de la aplicación.
    """
    return request.app.state.embeddings_model


def get_rerank_model(request: Request) -> CrossEncoder:
    """
    Obtiene el modelo de reranking desde el estado de la aplicación.

    Args:
        request (Request): Objeto de solicitud de FastAPI.

    Returns:
        CrossEncoder: Modelo de reranking cargado en el estado de la aplicación.
    """
    return request.app.state.rerank_model


def get_groq_client(request: Request) -> AsyncGroq:
    """
    Obtiene el cliente de Groq desde el estado de la aplicación.

    Args:
        request (Request): Objeto de solicitud de FastAPI.

    Returns:
        AsyncGroq: Cliente de Groq cargado en el estado de la aplicación.
    """
    return request.app.state.groq_client
