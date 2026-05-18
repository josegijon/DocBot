import logging
from typing import AsyncGenerator

from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    BackgroundTasks,
)
from fastapi.responses import StreamingResponse
from app.models.document import UploadResponse
from uuid import UUID, uuid4
import asyncio
import json

from app.rag.ingestor import ingest
from app.rag.progress import (
    get_progress,
    IngestionStatus,
)
from app.services.document_service import process_upload
from app.api.deps import get_embeddings_model
from app.core.config import settings
from app.core.exceptions import DocumentNotFoundException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])


# --- Endpoint ---
@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    embeddings_model=Depends(get_embeddings_model),
) -> UploadResponse:
    """
    Sube un documento al sistema y comienza su procesamiento en segundo plano.

    Args:
        background_tasks (BackgroundTasks): Tareas en segundo plano para procesar el archivo.
        file (UploadFile): Archivo a subir.
        embeddings_model: Modelo de embeddings para procesar el archivo.

    Returns:
        UploadResponse: Respuesta con el ID del documento, nombre del archivo y estado inicial.
    """
    document_id = str(uuid4())

    logger.info(f"Iniciando subida de archivo: {file.filename} - {document_id}")

    file_path = await process_upload(file, document_id)

    background_tasks.add_task(
        ingest,
        str(file_path),
        document_id,
        embeddings_model,
    )
    # Para producción se usaría Celery o ARQ con worker separado. Aquí añadiría complejidad y costo.

    logger.info(f"Subida de archivo {file.filename} - {document_id} finalizada")

    return UploadResponse(
        doc_id=document_id, filename=file.filename, status=IngestionStatus.PROCESSING
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
                yield f"data: {json.dumps({'status': 'failed', 'error': 'Error inesperado en la ingesta.'})}\n\n"
                break

            await asyncio.sleep(settings.SSE_POLL_INTERVAL_SECONDS)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
