"""Modelos de datos relacionados con la gestión de documentos."""

from uuid import UUID

from pydantic import BaseModel

from app.models.ingestion_status import IngestionStatus


class UploadResponse(BaseModel):
    """Respuesta devuelta tras la subida de un documento.

    Attributes:
        doc_id: Identificador único del documento subido.
        filename: Nombre original del archivo subido.
        status: Estado actual de la ingesta del documento.
    """

    doc_id: UUID
    filename: str
    status: IngestionStatus = IngestionStatus.PROCESSING
