from typing import AsyncGenerator
import asyncio

from groq import AsyncGroq
from sentence_transformers import CrossEncoder, SentenceTransformer
import logging

from app.core.config import settings
from app.rag.memory import add_message, get_history
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.rag.prompt_builder import build_prompt
from app.models.chat import MessageRole
from app.rag.generator import generate

logger = logging.getLogger(__name__)


async def chat_stream(
    message: str,
    doc_id: str,
    session_id: str,
    embeddings_model: SentenceTransformer,
    reranker: CrossEncoder,
    groq_client: AsyncGroq,
) -> AsyncGenerator[tuple[str, str | list[dict]], None]:

    logger.info(f"Iniciando chat_stream | session_id: {session_id} | doc_id: {doc_id}")

    history = get_history(str(session_id))
    initial_chunks = await asyncio.to_thread(
        retrieve, message, doc_id, embeddings_model
    )
    best_chunks = await asyncio.to_thread(rerank, reranker, message, initial_chunks)
    message_prompt = build_prompt(query=message, chunks=best_chunks, history=history)

    add_message(str(session_id), role=MessageRole.USER, content=message)

    full_response = ""

    async for token in generate(message_prompt, groq_client):
        full_response += token
        yield ("token", token)

    add_message(str(session_id), role=MessageRole.ASSISTANT, content=full_response)

    # Enviar fuentes al final del stream
    sources = [
        {
            "page": c["page"],
            "text": c["text"][: settings.SOURCE_SNIPPET_LENGTH] + "...",
        }
        for c in best_chunks
    ]

    logger.info(
        f"Stream finalizado | session_id: {session_id} | tokens generados: {len(full_response)}"
    )
    yield ("sources", sources)
