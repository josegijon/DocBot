from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from .api.routes import documents, chat

app = FastAPI(
    title="DocBot API",
    version="0.1.0",
    description="Backend para el sistema de gestión de documentos y chatbot"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(documents.router)
app.include_router(chat.router)

@app.get('/health')
def health(): 
    return {"status": "ok"}