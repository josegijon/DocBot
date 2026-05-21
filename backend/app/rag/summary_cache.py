"""Caché en memoria para resúmenes de documentos.

Este módulo proporciona funciones simples para obtener, guardar y eliminar
resúmenes asociados a un `document_id`. La caché es volátil y sólo existe
dentro del proceso en ejecución.
"""

import logging

logger = logging.getLogger(__name__)


_summary_cache: dict[str, str] = {}


def get_summary(document_id) -> str | None:
    """Recupera el resumen almacenado para un documento.

    Args:
        document_id: Identificador único del documento cuyo resumen se desea obtener.

    Returns:
        El resumen asociado al `document_id` si existe, o `None` en caso contrario.
    """
    return _summary_cache.get(document_id)


def save_summary(document_id, summary) -> None:
    """Guarda o actualiza el resumen de un documento en la caché.

    Args:
        document_id: Identificador único del documento.
        summary: Texto del resumen a almacenar.

    Returns:
        None
    """
    _summary_cache[document_id] = summary


def delete_summary(document_id) -> None:
    """Elimina el resumen asociado a un `document_id` de la caché.

    Args:
        document_id: Identificador único del documento.

    Returns:
        None

    Notas:
        La operación es silenciosa si no existe una entrada para `document_id`.
    """
    _summary_cache.pop(document_id, None)
    logger.info(f"Resumen eliminado para document_id '{document_id}'")
