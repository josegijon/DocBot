from uuid import UUID

from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: UUID
    filename: str
    chunks: int
