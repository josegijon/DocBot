from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents
from app.core.config import settings
from app.core.exception_handlers import (
    auth_exception_handler,
    document_not_found_handler,
    empty_query_handler,
    file_too_large_handler,
    file_write_handler,
    generic_docbot_handler,
    invalid_file_type_handler,
    llm_exception_handler,
    pdf_not_found_handler,
    vector_store_internal_handler,
)
from app.core.exceptions import (
    AuthException,
    DocBotException,
    DocumentNotFoundException,
    EmptyQueryError,
    FileTooLargeException,
    FileWriteException,
    InvalidFileTypeException,
    LLMException,
    PDFNotFoundException,
    VectorStoreInternalException,
)
from app.core.ml_models import load_embeddings_model, load_groq_client, load_reranker


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gestiona el ciclo de vida de la aplicación.

    Args:
        app (FastAPI): Instancia de la aplicación.

    Yields:
        AsyncGenerator[None, None]: Control del ciclo de vida durante el lifespan.

    Notas:
        Durante el arranque:
        - Carga el modelo de embeddings en memoria.
        - Inicializa el cliente asíncrono de Groq.
        - Carga el modelo de reranker en memoria.
        - Almacena las instancias en `app.state` para acceso global.
    """
    app.state.embeddings_model = load_embeddings_model()
    app.state.groq_client = load_groq_client()
    app.state.rerank_model = load_reranker()

    yield

    await app.state.groq_client.close()


app = FastAPI(
    title="DocBot API",
    version="0.1.0",
    description="Backend para el sistema de gestión de documentos y chatbot",
    lifespan=lifespan,
)

# --- Registro de Exception Handlers Globales ---
app.add_exception_handler(EmptyQueryError, empty_query_handler)
app.add_exception_handler(PDFNotFoundException, pdf_not_found_handler)
app.add_exception_handler(AuthException, auth_exception_handler)
app.add_exception_handler(LLMException, llm_exception_handler)
app.add_exception_handler(DocumentNotFoundException, document_not_found_handler)
app.add_exception_handler(VectorStoreInternalException, vector_store_internal_handler)
app.add_exception_handler(InvalidFileTypeException, invalid_file_type_handler)
app.add_exception_handler(FileTooLargeException, file_too_large_handler)
app.add_exception_handler(FileWriteException, file_write_handler)

app.add_exception_handler(DocBotException, generic_docbot_handler)  # Red de seguridad

# Configuración de Middleware para permitir peticiones Cross-Origin (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro de routers por dominio
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
def get_health_status() -> dict[str, str]:
    """Endpoint de verificación de estado.

    Returns:
        dict[str, str]: Diccionario que indica el estado del servicio ('ok').
    """
    return {"status": "ok"}
