import logging
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import (
    FileTooLargeException,
    FileWriteException,
    InvalidFileTypeException,
)

logger = logging.getLogger(__name__)


async def process_upload(file: UploadFile, doc_id: str) -> Path:
    if file.content_type != "application/pdf":
        logger.error("Error: el formato del archivo no es válido.")
        raise InvalidFileTypeException("Error: el formato del archivo no es válido.")

    content = b""

    logger.info(f"Iniciando lectura del archivo {file.filename}")

    while True:
        chunk = await file.read(settings.CHUNK_READ_SIZE)
        if not chunk:
            break
        content += chunk
        if len(content) / (1024**2) > settings.MAX_PDF_SIZE_MB:
            logger.error(
                f"El archivo supera el límite de {settings.MAX_PDF_SIZE_MB}MB."
            )
            raise FileTooLargeException(
                f"El archivo supera el límite de {settings.MAX_PDF_SIZE_MB}MB."
            )

    logger.info(f"Guardando archivo {doc_id} en disco.")

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{doc_id}.pdf"

    logger.info(f"Archivo {doc_id} guardado correctamente en {file_path}.")

    try:
        file_path.write_bytes(content)
    except OSError as e:
        raise FileWriteException(
            f"No se pudo guardar el archivo en el servidor: {str(e)}"
        )

    return file_path
