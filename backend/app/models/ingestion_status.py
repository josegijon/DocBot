from enum import Enum


class IngestionStatus(str, Enum):
    """Estados posibles del proceso de ingesta."""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
