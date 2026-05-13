from uuid import UUID

from pydantic import BaseModel

from app.models.ingestion_status import IngestionStatus


class UploadResponse(BaseModel):
    doc_id: UUID
    filename: str
    status: IngestionStatus = IngestionStatus.PROCESSING
