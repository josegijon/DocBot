from enum import Enum


class IngestionStatus(str, Enum):
    """Estados posibles del proceso de ingesta de documentos.

    Attributes:
        PROCESSING: El documento se está procesando actualmente.
        READY: El documento ha sido procesado exitosamente y está listo.
        FAILED: El procesamiento del documento ha fallado.
    """

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
