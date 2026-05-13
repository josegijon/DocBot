from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, StringConstraints, field_validator


class Source(BaseModel):
    """Representa una fuente de contexto recuperada en una consulta RAG.

    Attributes:
        text: Contenido textual del fragmento recuperado.
        page: Número de página de origen del fragmento.
        score: Puntuación de relevancia del fragmento respecto a la consulta.
    """

    text: str
    page: int
    score: float


class ChatRequest(BaseModel):
    """Representa la solicitud de un mensaje de chat entrante.

    Attributes:
        doc_id: Identificador único del documento sobre el que se consulta.
        session_id: Identificador único de la sesión de conversación.
        message: Texto del mensaje del usuario, entre 5 y 500 caracteres.
    """

    doc_id: UUID
    session_id: UUID
    message: Annotated[str, StringConstraints(min_length=5, max_length=500)]

    @field_validator("message", mode="before")
    @classmethod
    def clean_spaces(cls, value: str) -> str:
        """Elimina espacios en blanco al inicio y final del mensaje."""
        return value.strip()


class ChatResponse(BaseModel):
    """Representa la respuesta generada por el sistema de chat.

    Attributes:
        session_id: Identificador de la sesión de conversación asociada.
        sources: Lista de fuentes de contexto utilizadas para generar la respuesta.
    """

    session_id: UUID
    sources: list[Source]


class MessageRole(str, Enum):
    """Roles de mensaje para el sistema de chat.

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
