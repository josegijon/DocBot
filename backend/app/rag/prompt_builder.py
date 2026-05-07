"""
Módulo para construir prompts para el sistema RAG.

Este módulo contiene las plantillas y funciones necesarias para construir
prompts estructurados que se enviarán al modelo de lenguaje, incluyendo
el contexto del documento y el historial de conversación.
"""
import logging
from typing import Any

from app.models.chat import MessageRole
from app.core.exceptions import EmptyQueryError


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Eres un asistente experto en análisis de documentos. Tu objetivo es responder preguntas de forma precisa, profesional y honesta.

REGLAS CRÍTICAS DE COMPORTAMIENTO:
1. BASES DE CONOCIMIENTO: Responde ÚNICAMENTE basándote en el contexto proporcionado. No utilices conocimiento externo ni inventes datos.
2. AUSENCIA DE INFORMACIÓN: Si la respuesta no se encuentra en el contexto, responde exactamente: "Lo siento, pero la información solicitada no está disponible en el documento proporcionado."
3. CITAS OBLIGATORIAS: Cada vez que menciones un dato o hagas una afirmación, debes indicar la página de origen al final del párrafo o frase (ej: [Pág. 5]).
4. ESTILO: Sé directo y conciso. No divagues.
5. IDIOMA: Responde siempre en el mismo idioma en el que el usuario te hable.

CONTEXTO DEL DOCUMENTO:
-----------------------
{context_str}
-----------------------
"""


def build_prompt(query: str, chunks: list[dict[str, Any]], history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Construye un prompt estructurado para el modelo de lenguaje.

    Esta función toma una consulta del usuario, fragmentos de documentos recuperados
    y el historial de conversación para construir una lista de mensajes listos
    para ser enviados al modelo de lenguaje.

    Args:
        query: La pregunta o consulta del usuario. Debe ser una cadena no vacía.
        chunks: Lista de fragmentos de documentos recuperados, donde cada fragmento
            es un diccionario con las claves 'page' (número de página) y 'text'
            (contenido del fragmento).
        history: Lista de mensajes del historial de conversación, donde cada
            mensaje es un diccionario con las claves 'role' y 'content'.

    Returns:
        Una lista de diccionarios representando los mensajes del prompt,
        incluyendo el mensaje del sistema, el historial y la consulta actual.

    Raises:
        EmptyQueryError: Si la query proporcionada está vacía o contiene solo espacios.

    Example:
        >>> chunks = [{"page": 1, "text": "Contenido del documento"}]
        >>> history = [{"role": "user", "content": "Hola"}, {"role": "assistant", "content": "Hola, ¿en qué puedo ayudarte?"}]
        >>> messages = build_prompt("¿Qué dice el documento?", chunks, history)
    """
    if not query.strip():
        logger.error("Query vacía o solo espacios en blanco.")
        raise EmptyQueryError(
            "La consulta proporcionada está vacía o contiene únicamente espacios en blanco. "
            "Por favor, proporcione una pregunta o mensaje válido."
        )

    context_blocks = [f"[Chunk - Página {c['page']}]: {c['text']}" for c in chunks]
    context_str = "\n\n".join(context_blocks)

    messages = []
    messages.append(
        {"role": MessageRole.SYSTEM.value, "content": SYSTEM_PROMPT.format(context_str=context_str)}
    )

    messages.extend(history)
    messages.append({"role": MessageRole.USER.value, "content": query})

    return messages