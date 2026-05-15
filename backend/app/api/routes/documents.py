from fastapi import (
    APIRouter,
    UploadFile,
    BackgroundTasks,
    Request,
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

router = APIRouter(prefix="/api/documents", tags=["Documents"])


# --- Endpoint ---
@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile
):
    doc_id = str(uuid4())

    file_path = await process_upload(file, doc_id)

    client = request.app.state.embeddings_model

    background_tasks.add_task(
        ingest,
        str(file_path),
        doc_id,
        client,
    )
    # Para producción se usaria Celery o ARQ con worker separado. Aquí añadiria complejidad y costo.

    return UploadResponse(
        doc_id=doc_id, filename=file.filename, status=IngestionStatus.PROCESSING
    )


@router.get("/{doc_id}/status")
async def get_document_status(doc_id: UUID):
    async def event_generator():
        while True:
            entry = get_progress(str(doc_id)) or {
                "status": IngestionStatus.PROCESSING,
                "progress": 0,
            }
            current_progress = entry["progress"]
            current_status = entry["status"]

            data = json.dumps({"status": current_status, "progress": current_progress})

            yield f"data: {data}\n\n"

            if current_progress >= 100:
                break

            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
