"""Módulo para generación de respuestas mediante streaming desde la API de Groq.

Proporciona la función `generate` que consume el modelo LLM configurado
y emite fragmentos de texto de forma asíncrona, incluyendo métricas de
rendimiento (latencia, velocidad de chunks) y manejo de errores específicos
de la API.
"""

import logging
import time
from typing import AsyncGenerator

from groq import AsyncGroq, AuthenticationError, BadRequestError

from app.core.config import settings
from app.core.exceptions import AuthException, LLMException, ModelException

logger = logging.getLogger(__name__)


async def generate(
    messages: list[dict[str, str]], client: AsyncGroq
) -> AsyncGenerator[str, None]:
    """Generador asíncrono para streaming de respuestas desde Groq.

    Gestiona errores específicos de la API mapeándolos a excepciones internas
    y registra métricas de rendimiento (TTFT, latencia total, velocidad).

    Args:
        messages: Lista de mensajes del historial de conversación en formato
            ``[{"role": "...", "content": "..."}, ...]``.
        client: Instancia autenticada de :class:`groq.AsyncGroq` lista para
            realizar peticiones al modelo LLM.

    Yields:
        Fragmentos de texto generados por el modelo en modo streaming.

    Raises:
        AuthException: Si la API Key de Groq es inválida o ha expirado.
        ModelException: Si la petición al modelo es inválida (BadRequest).
        LLMException: Para cualquier otro error inesperado del servicio LLM.
    """
    try:
        logger.info(f"Iniciando generación LLM (Modelo: {settings.GROQ_MODEL})")

        start_time = time.perf_counter()
        chunks_count = 0
        is_first_chunk = True

        completion = await client.chat.completions.create(
            model=settings.GROQ_MODEL, messages=messages, stream=True
        )

        async for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                if is_first_chunk:
                    ttft = time.perf_counter() - start_time
                    logger.debug(f"Primer chunk recibido (TTFT): {ttft:.3f}s")
                    is_first_chunk = False

                chunks_count += 1
                yield content

        total_latency = time.perf_counter() - start_time
        velocidad = chunks_count / total_latency if total_latency > 0 else 0

        logger.info(
            f"Generación exitosa | Chunks: {chunks_count} | "
            f"Latencia total: {total_latency:.2f}s | Velocidad: {velocidad:.1f} chunks/s"
        )

    except AuthenticationError:
        logger.error("Error de autenticación con Groq: API Key inválida.")
        raise AuthException("La API Key de Groq es inválida o ha expirado.")

    except BadRequestError as e:
        logger.error(f"Error en la petición a Groq (BadRequest): {str(e)}")
        raise ModelException(f"Error en la configuración del modelo: {str(e)}")

    except Exception as e:
        # Errores no previstos: red, límites de tasa, timeouts, etc.
        error_msg = (
            f"Error inesperado en el servicio LLM: {type(e).__name__} - {str(e)}"
        )
        logger.critical(error_msg)
        raise LLMException(error_msg)
