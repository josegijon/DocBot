import asyncio
import logging
from typing import AsyncGenerator

from groq import AsyncGroq
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import settings
from app.models.chat import MessageRole
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
        user_message: Mensaje del usuario que se desea procesar.
        document_id: Identificador del documento sobre el cual se realiza la consulta.
        session_id: Identificador único de la sesión de chat.
        embedding_model: Modelo de embeddings utilizado para la recuperación
            de fragmentos semánticamente relevantes.
        reranker: Modelo de reranking para reordenar los fragmentos por
            relevancia respecto a la consulta.
        groq_client: Cliente asíncrono de Groq para la generación de texto.

    Yields:
        Tuplas con el tipo de evento y los datos asociados:
        - ``("token", str)``: Cada token generado durante la respuesta.
        - ``("sources", list[dict])``: Lista de fuentes al finalizar el stream,
          con las claves ``"page"`` y ``"text"``.
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
        yield ("token", token)

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
    yield ("sources", citation_sources)
