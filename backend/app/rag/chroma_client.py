"""Cliente de conexión a ChromaDB para almacenamiento vectorial persistente.

Este módulo proporciona una función para inicializar clientes de ChromaDB
con persistencia en disco, gestionando errores comunes de sistema, base de
datos y configuración.
"""

import logging
import sqlite3

import chromadb
from chromadb.api import ClientAPI
from chromadb.errors import InvalidConfigurationError

from app.core.config import settings
from app.core.exceptions import VectorStoreInternalException

logger = logging.getLogger(__name__)


def get_chroma_client(doc_id: str) -> ClientAPI:
    """Inicializa y devuelve un cliente persistente de ChromaDB para un documento específico.

    Crea o abre una base de datos vectorial persistente en disco, asociada al
    identificador de documento proporcionado. Gestiona errores de sistema,
    corrupción de base de datos y problemas de configuración, convirtiéndolos
    en excepciones del dominio de la aplicación.

    Args:
        doc_id: Identificador único del documento. Se utiliza para crear o
            localizar el directorio de persistencia de la base de datos vectorial.

    Returns:
        Instancia de :class:`chromadb.api.ClientAPI` conectada al almacén
        vectorial del documento especificado.

    Raises:
        VectorStoreInternalException: Si ocurre un error de sistema o permisos al
            acceder al directorio de persistencia, si el archivo de base de
            datos está corrupto, si la configuración de ChromaDB es inválida
            o si se produce un error inesperado durante la inicialización.
    """
    path = f"{settings.CHROMA_PERSIST_DIR}/{doc_id}"

    try:
        logger.info(f"Iniciado conexión ChromaDB para {doc_id}")
        client = chromadb.PersistentClient(path=path)
        logger.info(f"Conectado con éxito para {doc_id}.")

    except OSError as e:
        logger.error(
            f"Error de sistema/permisos al acceder a ChromaDB en {path}: {str(e)}"
        )
        raise VectorStoreInternalException(
            "No se pudo acceder al directorio de la base de datos. Verifique que la ruta existe y tiene permisos de lectura/escritura."
        )

    except sqlite3.DatabaseError as e:
        logger.error(
            f"Corrupción o error interno en SQLite para el documento {doc_id}: {str(e)}"
        )
        raise VectorStoreInternalException(
            "El archivo de la base de datos vectorial parece estar dañado o corrupto."
        )

    except InvalidConfigurationError as e:
        logger.error(f"Configuración inválida al inicializar ChromaDB: {str(e)}")
        raise VectorStoreInternalException(
            "Error de configuración en el motor de base de datos vectorial."
        )

    except Exception as e:
        logger.error(
            f"Error inesperado al inicializar ChromaDB para {doc_id}: {str(e)}"
        )
        raise VectorStoreInternalException(
            f"Ocurrió un error inesperado de conexión con la base de datos vectorial: {str(e)}"
        )

    return client
