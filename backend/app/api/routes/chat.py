import json
import logging
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.models.chat import ChatRequest
from app.rag.memory import get_history, delete_session
from app.core.exceptions import AuthException, LLMException
from app.services.chat_sevice import chat_stream


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/")
async def chat(request: Request, body: ChatRequest):
    embeddings_model = request.app.state.embeddings_model
    rerank_model = request.app.state.reranker
    client = request.app.state.groq_client

    async def event_generator():
        try:
            async for kind, data in chat_stream(
                body.message,
                str(body.doc_id),
                str(body.session_id),
                embeddings_model,
                rerank_model,
                client,
            ):
                if kind == "token":
                    yield f"data: {json.dumps({'token': data})}\n\n"
                elif kind == "sources":
                    yield f"data: {json.dumps({'sources': data})}\n\n"

        except AuthException as e:
            logger.error(f"Error de autenticación en el stream: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'type': 'auth_error', 'message': str(e)})}\n\n"

        except LLMException as e:
            logger.error(f"Error del servicio LLM en el stream: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'type': 'llm_error', 'message': str(e)})}\n\n"

        except Exception as e:
            logger.critical(f"Error inesperado en el stream: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'type': 'unexpected_error', 'message': str(e)})}\n\n"

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
