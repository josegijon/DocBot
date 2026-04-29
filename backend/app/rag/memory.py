from collections import deque

from app.core.exceptions import SessionException
from app.core.config import settings

# Diccionario global para almacenar historiales de chat por ID de sesión
# Se usa deque con maxlen para rotar automáticamente los mensajes más antiguos.
sessions: dict[str, deque[dict[str, str]]] = {}


def validate_session_id(session_id: str) -> None:
    """
    Valida que el identificador de sesión sea una cadena de texto válida.

    Args:
        session_id (str): ID de la sesión a validar.

    Raises:
        SessionException: Si el session_id es nulo o no es de tipo string.
    """
    if not session_id or not isinstance(session_id, str):
        raise SessionException("session_id inválido")


def get_history(session_id: str) -> list[dict[str, str]]:
    """
    Recupera el historial de conversación de una sesión específica. Valida el ID antes de operar.
    Si la sesión no existe, se inicializa una nueva con un límite de mensajes.

    Args:
        session_id (str): Identificador único de la sesión.

    Returns:
        list: Lista con los últimos mensajes de la conversación. La lista es una copia para proteger el deque interno.
    """
    validate_session_id(session_id)

    if session_id not in sessions:
        sessions[session_id] = deque(maxlen=settings.CONVERSATION_MAX_TURNS)

    return list(sessions[session_id])


def add_message(session_id: str, role: str, content: str) -> None:
    """
    Añade un nuevo mensaje al historial de una sesión.

    Args:
        session_id (str): ID de la sesión.
        role (str): Rol del emisor ('user', 'assistant' o 'system').
        content (str): Contenido textual del mensaje.
    """
    history = get_history(session_id)
    history.append({"role": role, "content": content})


def delete_session(session_id: str) -> None:
    """
    Elimina por completo una sesión y su historial de la memoria.

    Args:
        session_id (str): ID de la sesión a eliminar.
    """
    sessions.pop(session_id, None)
