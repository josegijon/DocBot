import logging

import chromadb

from chromadb.api import ClientAPI

from app.core.config import settings
from app.core.exceptions import VectorStoreException

logger = logging.getLogger(__name__)


def get_chroma_client(doc_id: str) -> ClientAPI:
    try:
        # Conecta al cliente
        client = chromadb.PersistentClient(
            path=f"{settings.CHROMA_PERSIST_DIR}/{doc_id}"
        )
    except Exception as e:
        logger.error(f"Error al inicializar ChromaDB para {doc_id}: {str(e)}")
        raise VectorStoreException(
            f"Error de conexión con la base de datos vectorial: {str(e)}"
        )

    return client
