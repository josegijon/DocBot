from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.ml_models import load_embeddings_model, load_groq_client, load_reranker
from api.routes import documents, chat

from app.core.exceptions import (
    AuthException,
    DocBotException,
    DocumentNotFoundException,
    FileTooLargeException,
    InvalidFileTypeException,
    LLMException,
    PDFNotFoundException,
    EmptyQueryError,
    VectorStoreInternalException,
)
from app.core.exception_handlers import (
    auth_exception_handler,
    document_not_found_handler,
    empty_query_handler,
    file_too_large_handler,
    invalid_file_type_handler,
    pdf_not_found_handler,
    llm_exception_handler,
    generic_docbot_handler,
    vector_store_internal_handler,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación.

    Durante el arranque:
        - Carga el modelo de embeddings en memoria.
        - Inicializa el cliente asíncrono de Groq.
        - Carga el modelo de reranker en memoria.
        - Almacena las instancias en 'app.state' para acceso global.
    """
    app.state.embeddings = load_embeddings_model()
    app.state.groq_client = load_groq_client()
    app.state.reranker = load_reranker()

    yield


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
app.add_exception_handler(DocBotException, generic_docbot_handler)  # Red de seguridad
app.add_exception_handler(InvalidFileTypeException, invalid_file_type_handler)
app.add_exception_handler(FileTooLargeException, file_too_large_handler)

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
def health():
    """
    Endpoint de verificación de estado.

    Se utiliza para monitorear que el servicio está levantado y responde correctamente.
    Returns:
        dict: Un diccionario indicando que el status es 'ok'.
    """
    return {"status": "ok"}
