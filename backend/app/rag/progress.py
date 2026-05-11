from enum import Enum
import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class IngestionStatus(str, Enum):
    """Estados posibles del proceso de ingesta."""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ProgressEntry(TypedDict):
    """Estructura de cada entrada en el almacén de progreso."""

    status: IngestionStatus
    progress: int


# Diccionario privado
# La clave es el doc_id y el valor un ProgressEntry
_progress_store: dict[str, ProgressEntry] = {}


def get_progress(doc_id: str) -> ProgressEntry | None:
    """
    Obtiene el estado de progreso de un documento.

    Args:
        doc_id: Identificador único del documento.

    Returns:
        ProgressEntry | None: Los datos de progreso o None si el doc_id no existe.
    """
    return _progress_store.get(doc_id)


def set_progress(doc_id: str, status: IngestionStatus, progress: int) -> None:
    """
    Actualiza o crea una entrada de progreso para un documento.

    Args:
        doc_id: Identificador único del documento.
        status: Estado actual del proceso de ingesta.
        progress: Porcentaje de progreso (0-100).
    """
    if not (0 <= progress <= 100):
        logger.warning(
            f"Valor de progreso inválido ({progress}) para doc_id '{doc_id}'. Se ajustará al rango [0, 100]."
        )

    progress = max(0, min(progress, 100))
    _progress_store[doc_id] = {"status": status, "progress": progress}


def delete_progress(doc_id: str) -> None:
    """
    Elimina la información de progreso de un documento del almacén.

    Args:
        doc_id: Identificador único del documento a eliminar.
    """
    _progress_store.pop(doc_id)
    logger.info(f"Progreso eliminado para doc_id '{doc_id}'")
