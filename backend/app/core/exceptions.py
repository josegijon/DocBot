class DocBotException(Exception):
    """Excepción base para todos los errores específicos de la aplicación."""

    pass


# --- Rama de Inteligencia Artificial (LLM) ---


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


# --- Rama de Recuperación y Almacenamiento (RAG) ---


class RAGException(DocBotException):
    """Errores base para el sistema de Recuperación Aumentada por Generación."""

    pass


class EmbeddingGenerationException(RAGException):
    """Error al generar representaciones vectoriales."""

    pass


class IngestionException(RAGException):
    """Errores ocurridos durante el proceso de ingesta de documentos."""

    pass


class SessionException(RAGException):
    """Errores relacionados con la sesión de chat."""

    pass


class PDFNotFoundException(IngestionException):
    """El archivo PDF especificado no existe en la ruta proporcionada."""

    pass


class PDFEmptyException(IngestionException):
    """El archivo PDF no contiene texto extraíble o está dañado."""

    pass


class VectorStoreException(IngestionException):
    """Errores relacionados con la base de datos vectorial (ChromaDB)."""

    pass


class VectorStoreInternalException(VectorStoreException):
    """Errores de infraestructura (corrupción, permisos, disco)."""

    pass


class DocumentNotFoundException(VectorStoreException):
    """El documento/colección solicitado no existe en la base de datos."""

    pass


class PromptException(RAGException):
    """Errores relacionados con la construcción de prompts."""

    pass


class EmptyQueryError(PromptException):
    """El query proporcionado está vacío o contiene solo espacios."""

    pass


class RetrievalException(RAGException):
    """Errores ocurridos durante el proceso de recuperación de información."""

    pass


class NoChunksFoundException(RetrievalException):
    """No se encontraron fragmentos (chunks) de texto relevantes para la consulta dada."""

    pass
