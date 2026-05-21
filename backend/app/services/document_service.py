"""Servicio para procesar y guardar archivos PDF subidos por los usuarios.

Este módulo expone funciones para almacenar PDFs subidos y eliminar
documentos asociados (progreso, colección y archivo en disco).
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator

from fastapi import UploadFile
from groq import AsyncGroq
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.exceptions import (
    FileTooLargeException,
    FileWriteException,
    InvalidFileTypeException,
)
from app.rag.cache_summary import get_summary, save_summary
from app.rag.chroma_client import get_chroma_client
from app.rag.generator import generate
from app.rag.prompt_builder import build_prompt
from app.rag.progress import delete_progress
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


_SUMMARY_PROMPT = "Resumen del documento"


async def process_pdf_upload(uploaded_pdf_file: UploadFile, document_id: str) -> Path:
    """Guardar un archivo PDF subido en el sistema de archivos.

    Lee el contenido del archivo por chunks, valida el tipo y el tamaño,
    y escribe el fichero resultante en el directorio de cargas.

    Args:
        uploaded_pdf_file (UploadFile): Instancia del archivo subido.
        document_id (str): Identificador único que se emplea como nombre
            del archivo almacenado (sin extensión).

    Returns:
        pathlib.Path: Ruta absoluta al archivo guardado.

    Raises:
        InvalidFileTypeException: Si el archivo no es de tipo PDF.
        FileTooLargeException: Si el tamaño supera la configuración
            `settings.MAX_PDF_SIZE_MB`.
        FileWriteException: Si ocurre un error al escribir el fichero.
    """
    if uploaded_pdf_file.content_type != "application/pdf":
        logger.error("Error: el formato del archivo no es válido.")
        raise InvalidFileTypeException("Error: el formato del archivo no es válido.")

    pdf_bytes = b""

    logger.info(f"Iniciando lectura del archivo {uploaded_pdf_file.filename}")

    while True:
        chunk_bytes = await uploaded_pdf_file.read(settings.CHUNK_READ_SIZE)
        if not chunk_bytes:
            break
        pdf_bytes += chunk_bytes
        if len(pdf_bytes) / (1024**2) > settings.MAX_PDF_SIZE_MB:
            logger.error(
                f"El archivo supera el límite de {settings.MAX_PDF_SIZE_MB}MB."
            )
            raise FileTooLargeException(
                f"El archivo supera el límite de {settings.MAX_PDF_SIZE_MB}MB."
            )

    logger.info(f"Guardando archivo {document_id} en disco.")

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf_path = upload_dir / f"{document_id}.pdf"

    try:
        saved_pdf_path.write_bytes(pdf_bytes)
    except OSError as error:
        raise FileWriteException(
            f"No se pudo guardar el archivo en el servidor: {str(error)}"
        )

    logger.info(f"Archivo {document_id} guardado correctamente en {saved_pdf_path}.")

    return saved_pdf_path


def delete_document(document_id: str) -> None:
    """Eliminar todos los artefactos asociados a un documento.

    Esta función elimina el progreso asociado, la colección en Chroma
    y el fichero PDF del sistema de archivos.

    Args:
        document_id (str): Identificador único del documento.
    """

    delete_progress(document_id)
    client = get_chroma_client(document_id)
    client.delete_collection(name=document_id)

    (Path(settings.UPLOAD_DIR) / f"{document_id}.pdf").unlink(missing_ok=True)


async def generate_summary(
    document_id: str, embedding_model: SentenceTransformer, groq_client: AsyncGroq
) -> AsyncGenerator:
    """Generar un resumen en streaming para un documento.

    Comprueba si existe un resumen cacheado; si no, recupera fragmentos
    relevantes mediante el modelo de embeddings, construye el prompt y
    genera el resumen en streaming. Emite tuplas ("token", str) por cada
    token generado y, al final, ("done", document_id).

    Args:
        document_id (str): Identificador único del documento.
        embedding_model (SentenceTransformer): Modelo de embeddings usado para la recuperación.
        groq_client (AsyncGroq): Cliente asíncrono para la generación en streaming.

    Yields:
        AsyncGenerator: Tuplas con los tokens generados y señal de finalización.
    """
    logger.info(f"Iniciando resumen del documento {document_id}.")

    accumulated_summary = get_summary(document_id)

    if accumulated_summary is None:
        retrieved_chunks = await asyncio.to_thread(
            retrieve, _SUMMARY_PROMPT, document_id, embedding_model
        )

        prompt = build_prompt(_SUMMARY_PROMPT, retrieved_chunks, [])

        accumulated_summary = ""

        async for token in generate(prompt, groq_client):
            accumulated_summary += token
            yield ("token", token)

        save_summary(document_id, accumulated_summary)
    else:
        yield ("token", accumulated_summary)

    logger.info(f"Resumen del documento {document_id} finalizado.")

    yield ("done", document_id)
