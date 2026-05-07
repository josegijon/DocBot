from pydantic import BaseModel
from enum import Enum


class ChatRequest(BaseModel):
    doc_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    sources: list


class MessageRole(str, Enum):
    """
    Roles de mensaje para el sistema de chat.
    
    Define los tipos de roles que pueden participar en una conversación,
    utilizados para identificar el autor de cada mensaje en el contexto
    del modelo de lenguaje.
    
    Attributes:
        SYSTEM: Mensaje del sistema que proporciona instrucciones o contexto.
        ASSISTANT: Mensaje generado por el asistente de IA.
        USER: Mensaje enviado por el usuario.
    """
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"