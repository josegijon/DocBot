"""
Servicio para procesar y guardar archivos PDF subidos por los usuarios.
"""

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


async def process_pdf_upload(uploaded_file: UploadFile, document_id: str) -> Path:
    """
    Procesa un archivo PDF subido y lo guarda en el sistema de archivos.

    Args:
        uploaded_file (UploadFile): Archivo subido por el usuario.
        document_id (str): Identificador único para el archivo.

    Returns:
        Path: Ruta del archivo guardado.

    Raises:
        InvalidFileTypeException: Si el archivo no es un PDF.
        FileTooLargeException: Si el archivo excede el tamaño máximo permitido.
        FileWriteException: Si ocurre un error al guardar el archivo.
    """
    if uploaded_file.content_type != "application/pdf":
        logger.error("Error: el formato del archivo no es válido.")
        raise InvalidFileTypeException("Error: el formato del archivo no es válido.")

    file_content = b""

    logger.info(f"Iniciando lectura del archivo {uploaded_file.filename}")

    while True:
        file_chunk = await uploaded_file.read(settings.CHUNK_READ_SIZE)
        if not file_chunk:
            break
        file_content += file_chunk
        if len(file_content) / (1024**2) > settings.MAX_PDF_SIZE_MB:
            logger.error(
                f"El archivo supera el límite de {settings.MAX_PDF_SIZE_MB}MB."
            )
            raise FileTooLargeException(
                f"El archivo supera el límite de {settings.MAX_PDF_SIZE_MB}MB."
            )

    logger.info(f"Guardando archivo {document_id} en disco.")

    upload_directory = Path(settings.UPLOAD_DIR)
    upload_directory.mkdir(parents=True, exist_ok=True)
    saved_file_path = upload_directory / f"{document_id}.pdf"

    try:
        saved_file_path.write_bytes(file_content)
    except OSError as error:
        raise FileWriteException(
            f"No se pudo guardar el archivo en el servidor: {str(error)}"
        )

    logger.info(f"Archivo {document_id} guardado correctamente en {saved_file_path}.")

    return saved_file_path
