from pydantic import BaseModel


class ChatRequest(BaseModel):
    doc_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    sources: list
