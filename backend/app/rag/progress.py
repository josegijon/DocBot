from enum import Enum
from typing import TypedDict


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

    Returns:
        ProgressEntry | None: Los datos de progreso o None si el doc_id no existe.
    """
    return _progress_store.get(doc_id)


def set_progress(doc_id: str, status: IngestionStatus, progress: int) -> None:
    """
    Actualiza o crea una entrada de progreso para un documento.
    """
    _progress_store[doc_id] = {"status": status, "progress": progress}


def delete_progress(doc_id: str) -> None:
    """
    Elimina la información de progreso de un documento del almacén.
    """
    _progress_store.pop(doc_id, None)
