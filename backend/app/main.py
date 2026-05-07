from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.ml_models import load_embeddings_model, load_groq_client, load_reranker
from api.routes import documents, chat


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
