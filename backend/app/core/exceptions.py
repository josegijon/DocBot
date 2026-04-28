class DocBotException(Exception):
    """Excepción base para todos los errores específicos de la aplicación."""

    pass


class LLMException(DocBotException):
    """Errores relacionados con Groq."""

    pass


class AuthException(LLMException):
    """
    Error de autenticación con el proveedor de LLM.
    API Key inválida o expirada.
    """

    pass


class ModelException(LLMException):
    """El modelo no existe o no está disponible."""

    pass
