import json

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.models.chat import ChatRequest
from app.rag.memory import get_history, add_message, delete_session
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.rag.prompt_builder import build_prompt
from app.rag.generator import generate


router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/")
async def chat(request: ChatRequest):
    # Obtiene historial
    history = get_history(request.session_id)

    # Recuperar candidatos
    initial_chunks = await run_in_threadpool(retrieve, request.message, request.doc_id)

    # Refinar resultados
    best_chunks = await run_in_threadpool(rerank, request.message, initial_chunks)

    # Construir prompt
    messages = build_prompt(query=request.message, chunks=best_chunks, history=history)

    # Añadir la pregunta del usuario al historial
    add_message(request.session_id, role="user", content=request.message)

    # Definir generador que capture la respuesta y la guarde
    async def event_generator():
        full_response = ""
        for token in generate(messages):
            full_response += token
            # Fromato SSE para cada token
            data = json.dumps({"token": token})
            yield f"data: {data}\n\n"

        # Guarda la respuesta completa
        add_message(request.session_id, role="assistant", content=full_response)

        # Enviar fuentes al final del stream
        sources = [
            {"page": c["page"], "text": c["text"][:100] + "..."} for c in best_chunks
        ]
        yield f"data: {json.dumps({'sources': sources})}\n\n"

    # Devuelve respuesta en streaming
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{session_id}/history")
async def get_chat_session_history(session_id: str):
    history = get_history(session_id)

    return list(history)
