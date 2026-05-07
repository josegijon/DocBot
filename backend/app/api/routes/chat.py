import json
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.models.chat import ChatRequest, MessageRole
from app.rag.memory import get_history, add_message, delete_session
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.rag.prompt_builder import build_prompt
from app.rag.generator import generate


router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/")
async def chat(request: Request, body: ChatRequest):
    # Obtiene historial
    history = get_history(str(body.session_id))

    embeddings_model = request.app.state.embeddings_model
    rerank_model = request.app.state.reranker

    # Recuperar candidatos
    initial_chunks = await run_in_threadpool(
        retrieve, body.message, str(body.doc_id), embeddings_model
    )

    # Refinar resultados
    best_chunks = await run_in_threadpool(
        rerank, rerank_model, body.message, initial_chunks
    )

    # Construir prompt
    messages = build_prompt(query=body.message, chunks=best_chunks, history=history)

    # Añadir la pregunta del usuario al historial
    add_message(str(body.session_id), role=MessageRole.USER, content=body.message)

    # Extraemos cliente del estado de la app
    client = request.app.state.groq_client

    # Definir generador que capture la respuesta y la guarde
    async def event_generator():
        full_response = ""

        async for token in generate(messages, client):
            full_response += token
            # Fromato SSE para cada token
            data = json.dumps({"token": token})
            yield f"data: {data}\n\n"

        # Guarda la respuesta completa
        add_message(
            str(body.session_id), role=MessageRole.ASSISTANT, content=full_response
        )

        # Enviar fuentes al final del stream
        sources = [
            {"page": c["page"], "text": c["text"][:100] + "..."} for c in best_chunks
        ]
        yield f"data: {json.dumps({'sources': sources})}\n\n"

    # Devuelve respuesta en streaming
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{session_id}/history")
async def get_chat_session_history(session_id: UUID):
    history = get_history(str(session_id))

    return list(history)


@router.delete("/{session_id}")
async def clear_chat_session(session_id: UUID):
    delete_session(str(session_id))

    return {
        "status": "success",
        "message": f"Historial de la sesión {str(session_id)} eliminado.",
    }
