from fastapi import APIRouter, HTTPException, UploadFile, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.models.document import UploadResponse
from uuid import uuid4
from pathlib import Path
import asyncio
import json

from app.core.config import settings
from app.rag.ingestor import ingest
from app.rag.progress import progress_store

router = APIRouter(prefix="/api/documents", tags=["Documents"])


async def validate_pdf_file(file: UploadFile):
    """
    Centraliza todas las validaciones del archivo.
    Lanza HTTPException si algo falla, evitando lógica de códigos numéricos.
    """
    # Validar Tipo
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo debe ser un PDF válido.",
        )

    # Validar Tamaño
    content = await file.read()
    size_in_mb = len(content) / (1024**2)

    if size_in_mb > settings.MAX_PDF_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El archivo es demasiado grande ({size_in_mb:.2f}MB). Máximo: {settings.MAX_PDF_SIZE_MB}MB.",
        )

    await file.seek(0)
    return content


# --- Endpoint ---
@router.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile):
    content = await validate_pdf_file(file)

    doc_id = str(uuid4())

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{doc_id}.pdf"

    try:
        file_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No se pudo guardar el archivo en el servidor: {str(e)}",
        )

    background_tasks.add_task(ingest, str(file_path), doc_id)
    # Para producción se usaria Celery o ARQ con worker separado. Aquí añadiria complejidad y costo.

    return UploadResponse(doc_id=doc_id, filename=file.filename, chunks=0)


@router.get("/{doc_id}/status")
async def get_document_status(doc_id: str):
    async def event_generator():
        while True:
            entry = progress_store.get(doc_id, {"status": "processing", "progress": 0})
            current_progress = entry["progress"]
            current_status = entry["status"]

            data = json.dumps({"status": current_status, "progress": current_progress})

            yield f"data: {data}\n\n"

            if current_progress >= 100:
                break

            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
