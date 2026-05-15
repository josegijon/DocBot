"""Manejadores de excepciones centralizados para la aplicación DocBot.

Este módulo define las funciones handler que FastAPI utiliza para capturar
excepciones específicas del dominio y devolver respuestas de error
estandarizadas en formato JSON.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AuthException,
    DocBotException,
    DocumentNotFoundException,
    EmptyQueryError,
    FileTooLargeException,
    InvalidFileTypeException,
    LLMException,
    PDFNotFoundException,
    VectorStoreInternalException,
)


def _build_error_response(
    request: Request, exc: Exception, status_code: int
) -> JSONResponse:
    """Construye una respuesta JSON estandarizada para errores.

    Args:
        request: Objeto de solicitud de FastAPI que contiene la información
            de la petición que provocó el error.
        exc: Instancia de la excepción capturada.
        status_code: Código de estado HTTP que se devolverá en la respuesta.

    Returns:
        Respuesta JSON con el tipo de error, mensaje, ruta y código de estado.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "path": request.url.path,
            "status_code": status_code,
        },
    )


async def empty_query_handler(request: Request, exc: EmptyQueryError) -> JSONResponse:
    """Maneja consultas vacías enviadas por el usuario.

    Args:
        request: Objeto de solicitud de FastAPI.
        exc: Instancia de EmptyQueryError con el detalle del error.

    Returns:
        Respuesta JSON con código de estado 400 Bad Request.
    """
    return _build_error_response(request, exc, status.HTTP_400_BAD_REQUEST)


async def pdf_not_found_handler(
    request: Request, exc: PDFNotFoundException
) -> JSONResponse:
    """Maneja errores cuando no se encuentra un archivo PDF.

    Args:
        request: Objeto de solicitud de FastAPI.
        exc: Instancia de PDFNotFoundException con el detalle del error.

    Returns:
        Respuesta JSON con código de estado 404 Not Found.
    """
    return _build_error_response(request, exc, status.HTTP_404_NOT_FOUND)


async def auth_exception_handler(request: Request, exc: AuthException) -> JSONResponse:
    """Maneja errores de autenticación relacionados con la configuración (API Key).

    Args:
        request: Objeto de solicitud de FastAPI.
        exc: Instancia de AuthException con el detalle del error.

    Returns:
        Respuesta JSON con código de estado 500 Internal Server Error.
    """
    return _build_error_response(request, exc, status.HTTP_500_INTERNAL_SERVER_ERROR)


async def llm_exception_handler(request: Request, exc: LLMException) -> JSONResponse:
    """Maneja errores del proveedor de LLM (Groq u otro servicio externo).

    Args:
        request: Objeto de solicitud de FastAPI.
        exc: Instancia de LLMException con el detalle del error.

    Returns:
        Respuesta JSON con código de estado 503 Service Unavailable.
    """
    return _build_error_response(request, exc, status.HTTP_503_SERVICE_UNAVAILABLE)


async def generic_docbot_handler(
    request: Request, exc: DocBotException
) -> JSONResponse:
    """Maneja excepciones de dominio no mapeadas por otros handlers.

    Actúa como red de seguridad para cualquier excepción de DocBot que no
    tenga un handler específico asignado.

    Args:
        request: Objeto de solicitud de FastAPI.
        exc: Instancia de DocBotException con el detalle del error.

    Returns:
        Respuesta JSON con código de estado 500 Internal Server Error.
    """
    return _build_error_response(request, exc, status.HTTP_500_INTERNAL_SERVER_ERROR)


async def document_not_found_handler(
    request: Request, exc: DocumentNotFoundException
) -> JSONResponse:
    """Maneja errores cuando un documento no existe en la base de datos vectorial.

    Args:
        request: Objeto de solicitud de FastAPI.
        exc: Instancia de DocumentNotFoundException con el detalle del error.

    Returns:
        Respuesta JSON con código de estado 404 Not Found.
    """
    return _build_error_response(request, exc, status.HTTP_404_NOT_FOUND)


async def vector_store_internal_handler(
    request: Request, exc: VectorStoreInternalException
) -> JSONResponse:
    """Maneja errores internos de la base de datos vectorial.

    Args:
        request: Objeto de solicitud de FastAPI.
        exc: Instancia de VectorStoreInternalException con el detalle del error.

    Returns:
        Respuesta JSON con código de estado 500 Internal Server Error.
    """
    return _build_error_response(request, exc, status.HTTP_500_INTERNAL_SERVER_ERROR)


async def invalid_file_type_handler(
    request: Request, exc: InvalidFileTypeException
) -> JSONResponse:
    """Maneja errores cuando el documento no es de tipo PDF.

    Args:
        request: Objeto de solicitud de FastAPI.
        exc: Instancia de InvalidFileTypeException con el detalle del error.

    Returns:
        Respuesta JSON con código de estado 400 Bad Request.
    """

    return _build_error_response(request, exc, status.HTTP_400_BAD_REQUEST)


async def file_too_large_handler(
    request: Request, exc: FileTooLargeException
) -> JSONResponse:
    """Maneja errores cuando un documento supera el límite de tamaño.

    Args:
        request: Objeto de solicitud de FastAPI.
        exc: Instancia de FileTooLargeException con el detalle del error.

    Returns:
        Respuesta JSON con código de estado 413 Content Too Large.
    """

    return _build_error_response(request, exc, status.HTTP_413_CONTENT_TOO_LARGE)
