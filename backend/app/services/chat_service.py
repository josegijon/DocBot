"""Servicio de chat: streaming de respuestas usando pipeline RAG.

Este módulo expone utilidades para generar respuestas de asistente en modo
streaming: recupera contexto y fragmentos relevantes, reordena por relevancia,
construye el prompt y delega la generación token a token.
"""

import asyncio
import logging
from typing import AsyncGenerator

from groq import AsyncGroq
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import settings
from app.models.chat import MessageRole
from app.models.stream import StreamEvent
from app.rag.generator import generate
from app.rag.memory import add_message, get_history
from app.rag.prompt_builder import build_prompt
from app.rag.reranker import rerank
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


async def stream_chat_response(
    user_message: str,
    document_id: str,
    session_id: str,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    groq_client: AsyncGroq,
) -> AsyncGenerator[tuple[str, str | list[dict]], None]:
    """Transmite la respuesta del asistente en tiempo real usando un pipeline RAG.

    Recupera el historial de la sesión, obtiene los fragmentos relevantes del
    documento, los reordena por relevancia, construye el prompt y genera la
    respuesta token a token mediante streaming. Al finalizar, emite las fuentes
    utilizadas como referencia.

    Args:
        user_message (str): Mensaje del usuario a procesar.
        document_id (str): Identificador del documento consultado.
        session_id (str): Identificador único de la sesión de chat.
        embedding_model (SentenceTransformer): Modelo de embeddings para la
            recuperación semántica de fragmentos.
        reranker (CrossEncoder): Modelo para reordenar fragmentos por
            relevancia respecto a la consulta.
        groq_client (AsyncGroq): Cliente asíncrono de Groq para generación.

    Yields:
        tuple: Pares ``(evento, datos)`` donde:
            - ``("token", str)``: cada token generado durante la respuesta.
            - ``("sources", list[dict])``: lista de fuentes con claves
              ``"page"`` y ``"text"`` al finalizar el stream.
    """
    logger.info(
        f"Iniciando stream_chat_response | session_id: {session_id} | document_id: {document_id}"
    )

    history = get_history(str(session_id))
    retrieved_chunks = await asyncio.to_thread(
        retrieve, user_message, document_id, embedding_model
    )
    reranked_chunks = await asyncio.to_thread(
        rerank, reranker, user_message, retrieved_chunks
    )
    prompt = build_prompt(query=user_message, chunks=reranked_chunks, history=history)

    add_message(str(session_id), role=MessageRole.USER, content=user_message)

    accumulated_response = ""

    async for token in generate(prompt, groq_client):
        accumulated_response += token
        yield (StreamEvent.EVENT_TOKEN, token)

    add_message(
        str(session_id), role=MessageRole.ASSISTANT, content=accumulated_response
    )

    # Emitir las fuentes al final del stream para que el cliente las reciba
    # una vez completada la generación de la respuesta.
    citation_sources = [
        {
            "page": chunk["page"],
            "text": chunk["text"][: settings.SOURCE_SNIPPET_LENGTH] + "...",
        }
        for chunk in reranked_chunks
    ]

    logger.info(
        f"Stream finalizado | session_id: {session_id} | tokens generados: {len(accumulated_response)}"
    )
    yield (StreamEvent.EVENT_SOURCES, citation_sources)
