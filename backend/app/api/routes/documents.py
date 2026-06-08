import asyncio
import json
import logging
from typing import AsyncGenerator
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from sentence_transformers import SentenceTransformer

from app.api.deps import get_embeddings_model, get_groq_client
from app.core.config import settings
from app.core.exceptions import AuthException, DocumentNotFoundException, LLMException
from app.models.document import UploadResponse
from app.models.stream import StreamEvent
from app.rag.ingestor import process_pdf_ingestion
from app.rag.progress import IngestionStatus, get_progress
from app.services.document_service import (
    delete_document as delete_document_service,
)
from app.services.document_service import (
    document_exists,
    generate_summary,
    process_pdf_upload,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    uploaded_file: UploadFile,
    embeddings_model: SentenceTransformer = Depends(get_embeddings_model),
) -> UploadResponse:
    """
    Sube un documento al sistema y comienza su procesamiento en segundo plano.

    Args:
        background_tasks (BackgroundTasks): Tareas en segundo plano para procesar el archivo.
        uploaded_file (UploadFile): Archivo a subir.
        embeddings_model (SentenceTransformer): Modelo de embeddings para procesar el archivo.

    Returns:
        UploadResponse: Respuesta con el ID del documento, nombre del archivo y estado inicial.
    """
    document_id = str(uuid4())

    logger.info(
        f"Iniciando subida de archivo: {uploaded_file.filename} - {document_id}"
    )

    saved_file_path = await process_pdf_upload(uploaded_file, document_id)

    background_tasks.add_task(
        process_pdf_ingestion,
        str(saved_file_path),
        document_id,
        embeddings_model,
    )
    # Para producción se usaría Celery o ARQ con worker separado. Aquí añadiría complejidad y costo.

    logger.info(
        f"Subida de archivo {uploaded_file.filename} - {document_id} finalizada"
    )

    return UploadResponse(
        doc_id=document_id,
        filename=uploaded_file.filename,
        status=IngestionStatus.PROCESSING,
    )


@router.get("/{document_id}/status")
async def get_document_status(document_id: UUID) -> StreamingResponse:
    """
    Obtiene el estado de procesamiento de un documento en tiempo real.

    Args:
        document_id (UUID): ID del documento a consultar.

    Returns:
        StreamingResponse: Respuesta en tiempo real con el estado y progreso del documento.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """
        Generador de eventos para transmitir el estado y progreso de un documento.

        Yields:
            str: Cadena en formato JSON con el estado y progreso del documento.

        Raises:
            DocumentNotFoundException: Si el documento no se encuentra en el sistema.
        """
        if get_progress(str(document_id)) is None:
            logger.error(f"Documento no encontrado en el sistema: {document_id}")
            raise DocumentNotFoundException(
                f"Documento no encontrado en el sistema: {document_id}"
            )

        while True:
            entry = get_progress(str(document_id)) or {
                "status": IngestionStatus.PROCESSING,
                "progress": 0,
            }
            current_progress = entry["progress"]
            current_status = entry["status"]

            data = json.dumps({"status": current_status, "progress": current_progress})

            yield f"data: {data}\n\n"

            if current_progress >= 100:
                break

            if entry["status"] == IngestionStatus.FAILED:
                yield f"data: {json.dumps({'status': 'failed', 'error': 'Error inesperado en la process_pdf_ingestiona.'})}\n\n"
                break

            await asyncio.sleep(settings.SSE_POLL_INTERVAL_SECONDS)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{document_id}/summary")
async def stream_document_summary(
    document_id: UUID,
    embeddings_model: SentenceTransformer = Depends(get_embeddings_model),
    groq_client: AsyncGroq = Depends(get_groq_client),
) -> StreamingResponse:
    """
    Genera el resumen de un documento y lo transmite mediante SSE.

    Args:
        document_id (UUID): ID del documento a resumir.
        embeddings_model (SentenceTransformer): Modelo de embeddings.
        groq_client (AsyncGroq): Cliente Groq para consultas.

    Returns:
        StreamingResponse: Stream SSE que emite tokens parciales y un evento de finalización.

    Raises:
        AuthException: Si falla la autenticación con el servicio LLM.
        LLMException: Si ocurre un error en el servicio LLM.
    """

    async def summary_sse_generator():
        try:
            async for event_type, event_data in generate_summary(
                str(document_id),
                embeddings_model,
                groq_client,
            ):
                if event_type == StreamEvent.EVENT_TOKEN:
                    yield f"data: {json.dumps({'token': event_data})}\n\n"
                elif event_type == StreamEvent.EVENT_DONE:
                    yield f"data: {json.dumps({'done': True})}\n\n"

        except AuthException as auth_error:
            logger.error(f"Error de autenticación en el stream: {str(auth_error)}")
            yield f"event: stream_error\ndata: {json.dumps({'type': 'auth_error', 'message': str(auth_error)})}\n\n"

        except LLMException as llm_error:
            logger.error(f"Error del servicio LLM en el stream: {str(llm_error)}")
            yield f"event: stream_error\ndata: {json.dumps({'type': 'llm_error', 'message': str(llm_error)})}\n\n"

        except Exception as unexpected_error:
            logger.critical(f"Error inesperado en el stream: {str(unexpected_error)}")
            yield f"event: stream_error\ndata: {json.dumps({'type': 'unexpected_error', 'message': str(unexpected_error)})}\n\n"

    return StreamingResponse(summary_sse_generator(), media_type="text/event-stream")


@router.delete("/{document_id}")
async def delete_document(document_id: UUID) -> dict[str, str]:
    """
    Elimina un documento y sus datos asociados.

    Args:
        document_id (UUID): ID del documento a eliminar.

    Returns:
        dict[str, str]: Diccionario con el estado y mensaje de resultado.
    """
    await asyncio.to_thread(delete_document_service, str(document_id))
    return {
        "status": "success",
        "message": f"Documento {str(document_id)} eliminado correctamente.",
    }


@router.get("/{document_id}/exists")
async def check_document_exists(document_id: UUID) -> dict[str, bool]:
    """
    Comprueba si un documento existe en el sistema.

    Args:
        document_id (UUID): ID del documento a consultar.

    Returns:
        dict[str, bool]: Clave `exists` indicando si el documento existe.
    """
    exists_flag = await asyncio.to_thread(document_exists, str(document_id))
    return {"exists": exists_flag}
