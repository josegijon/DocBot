"""Endpoints de chat para la API del sistema de preguntas y respuestas.

Define las rutas para procesar mensajes de chat en tiempo real mediante
Server-Sent Events (SSE), consultar el historial de sesiones y eliminar
historiales de chat.
"""

import json
import logging
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.exceptions import AuthException, LLMException
from app.models.chat import ChatRequest
from app.rag.memory import delete_session, get_history
from app.services.chat_service import stream_chat_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/")
async def process_chat_message(
    request: Request, chat_request: ChatRequest
) -> StreamingResponse:
    """Procesa una pregunta del usuario y devuelve la respuesta en formato SSE.

    Recibe el mensaje del usuario, lo procesa a través del pipeline RAG
    (retrieval-augmented generation) y emite los tokens de la respuesta
    junto con las fuentes recuperadas como un stream de eventos.

    Args:
        request: Objeto de solicitud de FastAPI con los modelos cargados
            en el estado de la aplicación.
        chat_request: Cuerpo de la petición con el mensaje, documento y sesión.

    Returns:
        StreamingResponse con los eventos en formato Server-Sent Events.
    """
    embeddings_model = request.app.state.embeddings_model
    rerank_model = request.app.state.reranker
    groq_client = request.app.state.groq_client

    async def generate_sse_events():
        try:
            async for event_type, event_data in stream_chat_response(
                chat_request.message,
                str(chat_request.doc_id),
                str(chat_request.session_id),
                embeddings_model,
                rerank_model,
                groq_client,
            ):
                if event_type == "token":
                    yield f"data: {json.dumps({'token': event_data})}\n\n"
                elif event_type == "sources":
                    yield f"data: {json.dumps({'sources': event_data})}\n\n"

        except AuthException as e:
            logger.error(f"Error de autenticación en el stream: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'type': 'auth_error', 'message': str(e)})}\n\n"

        except LLMException as e:
            logger.error(f"Error del servicio LLM en el stream: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'type': 'llm_error', 'message': str(e)})}\n\n"

        except Exception as e:
            logger.critical(f"Error inesperado en el stream: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'type': 'unexpected_error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate_sse_events(), media_type="text/event-stream")


@router.get("/{session_id}/history")
async def get_chat_session_history(session_id: UUID) -> list[dict[str, str]]:
    """Obtiene el historial de mensajes de una sesión de chat.

    Args:
        session_id: Identificador único de la sesión.

    Returns:
        Lista de mensajes almacenados en el historial de la sesión.
    """
    session_history = get_history(str(session_id))

    return list(session_history)


@router.delete("/{session_id}")
async def clear_chat_session(session_id: UUID) -> dict[str, str]:
    """Elimina el historial de una sesión de chat.

    Args:
        session_id: Identificador único de la sesión a eliminar.

    Returns:
        Diccionario con el estado de la operación y un mensaje informativo.
    """
    delete_session(str(session_id))

    return {
        "status": "success",
        "message": f"Historial de la sesión {str(session_id)} eliminado.",
    }
