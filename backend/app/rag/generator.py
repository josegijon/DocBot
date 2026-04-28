from groq import AsyncGroq, AuthenticationError, BadRequestError
import logging

from app.core.config import settings
from app.core.exceptions import AuthException, ModelException, LLMException

logger = logging.getLogger(__name__)


async def generate(messages: list, client: AsyncGroq):
    """
    Generador asíncrono para streaming de respuestas desde Groq.
    Gestiona errores específicos de la API mapeándolos a excepciones internas.
    """
    try:
        completion = await client.chat.completions.create(
            model=settings.GROQ_MODEL, messages=messages, stream=True
        )

        async for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except AuthenticationError:
        logger.error("Error de autenticación con Groq: API Key inválida.")
        raise AuthException("La API Key de Groq es inválida o ha expirado.")

    except BadRequestError as e:
        logger.error(f"Error en la petición a Groq (BadRequest): {str(e)}")
        raise ModelException(f"Error en la configuración del modelo: {str(e)}")

    except Exception as e:
        # Capturamos cualquier otro error (red, límites de tasa, etc.)
        error_msg = (
            f"Error inesperado en el servicio LLM: {type(e).__name__} - {str(e)}"
        )
        logger.critical(error_msg)
        raise LLMException(error_msg)
