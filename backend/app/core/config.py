"""Configuración central de la aplicación.

Define las variables de entorno y valores por defecto utilizados
en toda la aplicación, gestionados a través de pydantic-settings
con carga desde un archivo ``.env``.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración global de la aplicación DocBot.

    Carga sus valores desde variables de entorno con soporte
    para un archivo ``.env``. Las variables sin valor por defecto
    (como ``GROQ_API_KEY``) son obligatorias y provocarán un
    error si no se definen.

    Attributes:
        GROQ_API_KEY: Clave de API de Groq para acceso al modelo LLM.
        CHROMA_PERSIST_DIR: Ruta del directorio donde ChromaDB
            persiste los vectores en disco.
        UPLOAD_DIR: Ruta del directorio donde se almacenan los
            archivos PDF subidos temporalmente.
        MAX_PDF_SIZE_MB: Tamaño máximo permitido (en MB) para
            la subida de archivos PDF.
        CORS_ORIGINS: Lista de orígenes permitidos para peticiones
            CORS desde el frontend.
        GROQ_MODEL: Nombre del modelo LLM de Groq utilizado para
            generar respuestas.
        EMBEDDINGS_MODEL_NAME: Nombre del modelo de embeddings
            utilizado para la vectorización de documentos.
        CONVERSATION_MAX_TURNS: Número máximo de turnos que se
            mantienen en el historial de conversación.
        RERANKER_TOP_K: Cantidad de documentos candidatos que el
            reranker reordena y filtra.
        RERANKER_MODEL_NAME: Nombre del modelo cross-encoder
            utilizado para el reranking de resultados.
        N_RESULTS_RETRIEVE: Número de resultados que el retriever
            devuelve antes de aplicar el reranker.
        CHUNK_SIZE: Tamaño en caracteres de cada fragmento (chunk)
            al dividir los documentos.
        CHUNK_OVERLAP: Número de caracteres de solapamiento entre
            fragmentos consecutivos para preservar contexto.
        SOURCE_SNIPPET_LENGTH: Longitud en caracteres del fragmento
            de fuente incluido en las respuestas generadas.
    """

    GROQ_API_KEY: str
    CHROMA_PERSIST_DIR: Path = "./storage/chroma"
    UPLOAD_DIR: Path = "./storage/uploads"
    MAX_PDF_SIZE_MB: int = 50
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDINGS_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CONVERSATION_MAX_TURNS: int = 6
    RERANKER_TOP_K: int = 3
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    N_RESULTS_RETRIEVE: int = 10
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    SOURCE_SNIPPET_LENGTH: int = 100
    CHUNK_READ_SIZE: int = 1 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
