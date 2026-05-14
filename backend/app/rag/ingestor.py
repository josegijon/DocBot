from pathlib import Path
from uuid import uuid4

import chromadb
import fitz
import logging
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


def get_text_pdf(pdf_path: Path, doc_id: str) -> list[Document]:
    """
    Extrae el texto de un archivo PDF y lo convierte en una lista de objetos Document.

    Args:
        pdf_path (Path): Ruta local al archivo PDF.
        doc_id (str): Identificador único del documento.

    Returns:
        list[Document]: Lista de documentos de LangChain (uno por página).
    """
    try:
        doc = fitz.open(pdf_path)
    except FileNotFoundError:
        logger.error(f"Archivo no encontrado: {pdf_path}")
        raise PDFNotFoundException(f"El archivo PDF en {pdf_path} no existe.")
    except Exception as e:
        logger.error(f"Error al abrir el PDF con PyMuPDF: {str(e)}")
        raise PDFNotFoundException(f"No se pudo abrir el archivo PDF: {str(e)}")

    documents_langchain = [
        Document(
            page_content=page.get_text(),
            metadata={"page": page.number + 1, "doc_id": doc_id},
        )
        for page in doc
    ]

    doc.close()
    return documents_langchain


def create_chunks(documents_langchain: list[Document]) -> list[Document]:
    """
    Divide los documentos extensos en fragmentos más pequeños (chunks).

    Args:
        documents_langchain (list[Document]): Lista de documentos por página.

    Returns:
        list[Document]: Lista de fragmentos de texto procesados.
    """
    return text_splitter.split_documents(documents_langchain)


def extract_texts(chunks: list[Document]) -> list[str]:
    """
    Extrae únicamente el contenido textual de una lista de fragmentos.

    Args:
        chunks (list[Document]): Lista de objetos Document de LangChain.

    Returns:
        list[str]: Lista de cadenas de texto.
    """
    return [chunk.page_content for chunk in chunks]


def initialize_client(doc_id: str) -> chromadb.Collection:
    """
    Inicializa el cliente de base de datos vectorial ChromaDB y crea una colección.

    Args:
        doc_id (str): Identificador único usado como nombre de la colección y ruta.

    Returns:
        chromadb.Collection: La colección de ChromaDB lista para insertar datos.
    """
    client = get_chroma_client(doc_id)

    try:
        collection = client.get_or_create_collection(
            name=doc_id,
        )
        return collection
    except Exception as e:
        logger.error(f"Error en la colección del documento {doc_id}: {str(e)}")
        raise VectorStoreInternalException(
            f"Error interno al inicializar la colección {doc_id}: {str(e)}"
        )


def insert_chunks(
    collection: chromadb.Collection,
    texts: list[str],
    vectors: list[list[float]],
    final_chunks: list[Document],
) -> None:
    """
    Inserta los fragmentos de texto, sus vectores y metadatos en la colección de ChromaDB.

    Args:
        collection: Colección de ChromaDB activa.
        texts (list[str]): Contenido textual de los fragmentos.
        vectors (list[list[float]]): Embeddings correspondientes.
        final_chunks (list[Document]): Fragmentos originales para extraer metadatos.
    """
    try:
        collection.add(
            ids=[str(uuid4()) for _ in final_chunks],
            documents=texts,
            metadatas=[chunk.metadata for chunk in final_chunks],
            embeddings=vectors,
        )
    except Exception as e:
        logger.error(f"Error al insertar chunks en ChromaDB: {str(e)}")
        raise VectorStoreInternalException(
            f"No se pudieron guardar los fragmentos en la base de datos: {str(e)}"
        )


def ingest(pdf_path: Path, doc_id: str, embeddings_model: SentenceTransformer) -> int:
    """
    Orquestador del pipeline RAG: extrae, fragmenta, vectoriza y almacena un PDF.

    Este proceso actualiza periódicamente el 'progress_store' para que el frontend
    pueda mostrar el estado de la carga en tiempo real.

    Args:
        pdf_path (Path): Ruta del archivo PDF a procesar.
        doc_id (str): ID único para el documento.
        embeddings_model: Instancia del modelo de embeddings a utilizar.

    Returns:
        int: Número total de fragmentos insertados en la base de datos.
    """
    logger.info(f"Iniciando ingesta del documento: {doc_id}")

    set_progress(doc_id, IngestionStatus.PROCESSING, 0)

    documents_langchain = get_text_pdf(pdf_path, doc_id)

    # Validación de PDF vacío: sin contenido extraíble en ninguna página.
    if not documents_langchain or all(
        not d.page_content.strip() for d in documents_langchain
    ):
        set_progress(doc_id, IngestionStatus.FAILED, 0)
        logger.warning(f"El documento {doc_id} está vacío o no tiene texto legible.")
        raise PDFEmptyException("El archivo PDF no contiene texto extraíble.")

    set_progress(doc_id, IngestionStatus.PROCESSING, 25)

    final_chunks = create_chunks(documents_langchain)
    set_progress(doc_id, IngestionStatus.PROCESSING, 50)

    texts = extract_texts(final_chunks)

    vectors = create_embeddings(texts, embeddings_model)
    set_progress(doc_id, IngestionStatus.PROCESSING, 75)

    collection = initialize_client(doc_id)

    insert_chunks(collection, texts, vectors, final_chunks)

    set_progress(doc_id, IngestionStatus.READY, 100)

    logger.info(f"Indexación completada: {collection.count} fragmentos de {doc_id}")

    return collection.count()
