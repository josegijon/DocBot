"""Módulo que define el enum `StreamEvent` para eventos de streaming.

Provee constantes que representan eventos emitidos durante procesos de
streaming de tokens.
"""

from enum import Enum


class StreamEvent(str, Enum):
    """Enum que representa eventos emitidos durante un streaming de tokens.

    Attributes:
        EVENT_DONE (str): Indica que el streaming ha finalizado.
        EVENT_TOKEN (str): Indica que se ha producido un token intermedio.
    """

    EVENT_DONE = "done"
    EVENT_TOKEN = "token"
