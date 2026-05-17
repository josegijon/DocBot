from pathlib import Path
from uuid import uuid4
import logging

import chromadb
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.exceptions import (
    PDFEmptyException,
    PDFNotFoundException,
    VectorStoreInternalException,
)
from app.models.ingestion_status import IngestionStatus
from app.rag.chroma_client import get_chroma_client
from app.rag.embeddings import create_embeddings
from app.rag.progress import set_progress

logger = logging.getLogger(__name__)

# Configuración del divisor de texto para fragmentación del PDF.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)


def extract_text_from_pdf(pdf_path: Path, document_id: str) -> list[Document]:
    """
    Extrae el texto de un archivo PDF y lo convierte en una lista de objetos Document.

    Args:
        pdf_path (Path): Ruta local al archivo PDF.
        document_id (str): Identificador único del documento.

    Returns:
        list[Document]: Lista de documentos de LangChain (uno por página).

    Raises:
        PDFNotFoundException: Si el archivo no existe o no se puede abrir.
    """
    try:
        pdf_document = fitz.open(pdf_path)
    except FileNotFoundError:
        logger.error(f"Archivo no encontrado: {pdf_path}")
        raise PDFNotFoundException(f"El archivo PDF en {pdf_path} no existe.")
    except Exception as error:
        logger.error(f"Error al abrir el PDF con PyMuPDF: {str(error)}")
        raise PDFNotFoundException(f"No se pudo abrir el archivo PDF: {str(error)}")

    langchain_documents = [
        Document(
            page_content=page.get_text(),
            metadata={"page": page.number + 1, "document_id": document_id},
        )
        for page in pdf_document
    ]

    pdf_document.close()
    return langchain_documents


def split_into_chunks(langchain_documents: list[Document]) -> list[Document]:
    """
    Divide los documentos extensos en fragmentos más pequeños (chunks).

    Args:
        langchain_documents (list[Document]): Lista de documentos por página.

    Returns:
        list[Document]: Lista de fragmentos de texto procesados.
    """
    return text_splitter.split_documents(langchain_documents)


def extract_text_content(chunks: list[Document]) -> list[str]:
    """
    Extrae únicamente el contenido textual de una lista de fragmentos.

    Args:
        chunks (list[Document]): Lista de objetos Document de LangChain.

    Returns:
        list[str]: Lista de cadenas de texto.
    """
    return [chunk.page_content for chunk in chunks]


def initialize_chroma_collection(document_id: str) -> chromadb.Collection:
    """
    Inicializa el cliente de base de datos vectorial ChromaDB y crea una colección.

    Args:
        document_id (str): Identificador único usado como nombre de la colección y ruta.

    Returns:
        chromadb.Collection: La colección de ChromaDB lista para insertar datos.

    Raises:
        VectorStoreInternalException: Si ocurre un error al inicializar la colección.
    """
    chroma_client = get_chroma_client(document_id)

    try:
        chroma_collection = chroma_client.get_or_create_collection(
            name=document_id,
        )
        return chroma_collection
    except Exception as error:
        logger.error(f"Error en la colección del documento {document_id}: {str(error)}")
        raise VectorStoreInternalException(
            f"Error interno al inicializar la colección {document_id}: {str(error)}"
        )


def insert_chunks_into_collection(
    chroma_collection: chromadb.Collection,
    chunk_texts: list[str],
    chunk_vectors: list[list[float]],
    processed_chunks: list[Document],
) -> None:
    """
    Inserta los fragmentos de texto, sus vectores y metadatos en la colección de ChromaDB.

    Args:
        chroma_collection (chromadb.Collection): Colección de ChromaDB activa.
        chunk_texts (list[str]): Contenido textual de los fragmentos.
        chunk_vectors (list[list[float]]): Embeddings correspondientes.
        processed_chunks (list[Document]): Fragmentos originales para extraer metadatos.

    Raises:
        VectorStoreInternalException: Si ocurre un error al insertar los fragmentos.
    """
    try:
        chroma_collection.add(
            ids=[str(uuid4()) for _ in processed_chunks],
            documents=chunk_texts,
            metadatas=[chunk.metadata for chunk in processed_chunks],
            embeddings=chunk_vectors,
        )
    except Exception as error:
        logger.error(f"Error al insertar chunks en ChromaDB: {str(error)}")
        raise VectorStoreInternalException(
            f"No se pudieron guardar los fragmentos en la base de datos: {str(error)}"
        )


def process_pdf_ingestion(
    pdf_path: Path, document_id: str, embedding_model: SentenceTransformer
) -> int:
    """
    Orquestador del pipeline RAG: extrae, fragmenta, vectoriza y almacena un PDF.

    Este proceso actualiza periódicamente el 'progress_store' para que el frontend
    pueda mostrar el estado de la carga en tiempo real.

    Args:
        pdf_path (Path): Ruta del archivo PDF a procesar.
        document_id (str): ID único para el documento.
        embedding_model (SentenceTransformer): Instancia del modelo de embeddings a utilizar.

    Returns:
        int: Número total de fragmentos insertados en la base de datos.

    Raises:
        PDFEmptyException: Si el archivo PDF no contiene texto extraíble.
    """
    try:
        logger.info(f"Iniciando ingesta del documento: {document_id}")

        set_progress(document_id, IngestionStatus.PROCESSING, 0)

        langchain_documents = extract_text_from_pdf(pdf_path, document_id)

        # Validación de PDF vacío: sin contenido extraíble en ninguna página.
        if not langchain_documents or all(
            not document.page_content.strip() for document in langchain_documents
        ):
            set_progress(document_id, IngestionStatus.FAILED, 0)
            logger.warning(
                f"El documento {document_id} está vacío o no tiene texto legible."
            )
            raise PDFEmptyException("El archivo PDF no contiene texto extraíble.")

        set_progress(document_id, IngestionStatus.PROCESSING, 25)

        processed_chunks = split_into_chunks(langchain_documents)
        set_progress(document_id, IngestionStatus.PROCESSING, 50)

        chunk_texts = extract_text_content(processed_chunks)

        chunk_vectors = create_embeddings(chunk_texts, embedding_model)
        set_progress(document_id, IngestionStatus.PROCESSING, 75)

        chroma_collection = initialize_chroma_collection(document_id)

        insert_chunks_into_collection(
            chroma_collection, chunk_texts, chunk_vectors, processed_chunks
        )

        set_progress(document_id, IngestionStatus.READY, 100)

        logger.info(
            f"Indexación completada: {chroma_collection.count} fragmentos de {document_id}"
        )

        return chroma_collection.count()

    except PDFEmptyException:
        raise
    except Exception as error:
        set_progress(document_id, IngestionStatus.FAILED, 0)
        logger.error(f"Error inesperado en la ingesta: {error}")
