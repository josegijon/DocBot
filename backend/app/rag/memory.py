from collections import deque
import logging

from app.core.config import settings
from app.models.chat import MessageRole

logger = logging.getLogger(__name__)


# Diccionario global para almacenar historiales de chat por ID de sesión
# Se usa deque con maxlen para rotar automáticamente los mensajes más antiguos.
sessions: dict[str, deque[dict[str, str]]] = {}


def get_history(session_id: str) -> list[dict[str, str]]:
    """
    Recupera el historial de conversación de una sesión específica.

    Args:
        session_id (str): Identificador único de la sesión.

    Returns:
        list[dict[str, str]]: Lista con los últimos mensajes de la conversación.
            Devuelve una lista vacía si la sesión no existe. La lista es una copia
            para proteger el deque interno.
    """

    return list(sessions.get(session_id, []))


def add_message(session_id: str, role: MessageRole, content: str) -> None:
    """
    Añade un nuevo mensaje al historial de una sesión.

    Args:
        session_id (str): ID de la sesión.
        role (MessageRole): Rol del emisor (USER, ASSISTANT o SYSTEM).
        content (str): Contenido textual del mensaje.
    """
    if session_id not in sessions:
        logger.info(f"Nueva sesión generada en memoria: {session_id}")
        sessions[session_id] = deque(maxlen=settings.CONVERSATION_MAX_TURNS)

    sessions[session_id].append({"role": role.value, "content": content})


def delete_session(session_id: str) -> None:
    """
    Elimina por completo una sesión y su historial de la memoria.

    Args:
        session_id (str): ID de la sesión a eliminar.
    """
    sessions.pop(session_id, None)
    logger.debug(f"Sesión eliminada en memoria: {session_id}")
