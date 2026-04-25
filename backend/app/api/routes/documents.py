from fastapi import APIRouter, HTTPException, UploadFile, status
from app.models.document import UploadResponse
from uuid import uuid4
from pathlib import Path
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.rag.ingestor import ingest

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
async def upload_document(file: UploadFile):
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

    try:
        num_chunks = await run_in_threadpool(ingest, str(file_path), doc_id)
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar el contenido del PDF: {str(e)}",
        )

    return UploadResponse(doc_id=doc_id, filename=file.filename, chunks=num_chunks)
