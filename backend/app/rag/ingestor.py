import fitz
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb
from uuid import uuid4

from app.core.config import settings
from app.rag.progress import (
    set_progress,
    IngestionStatus,
)
from app.core.exceptions import (
    PDFNotFoundException,
    VectorStoreException,
    PDFEmptyException,
)

logger = logging.getLogger(__name__)


# Configuración del divisor de texto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Tamaño máximo de cada trozo (caracteres)
    chunk_overlap=150,  # Cuántos caracteres se repiten del trozo anterior
)


def get_text_pdf(pdf_path, doc_id):
    """
    Extrae el texto de un archivo PDF y lo convierte en una lista de objetos Document.

    Args:
        pdf_path (str): Ruta local al archivo PDF.
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


def create_chunks(documents_langchain):
    """
    Divide los documentos extensos en fragmentos más pequeños (chunks).

    Args:
        documents_langchain (list[Document]): Lista de documentos por página.

    Returns:
        list[Document]: Lista de fragmentos de texto procesados.
    """
    return text_splitter.split_documents(documents_langchain)


def extract_texts(chunks):
    """
    Extrae únicamente el contenido textual de una lista de fragmentos.

    Args:
        chunks (list[Document]): Lista de objetos Document de LangChain.

    Returns:
        list[str]: Lista de cadenas de texto.
    """
    return [chunk.page_content for chunk in chunks]


def create_embeddings(texts, embeddings_model):
    """
    Transforma una lista de textos en sus representaciones vectoriales (embeddings).

    Args:
        texts (list[str]): Textos a procesar.
        embeddings_model: Modelo de SentenceTransformers cargado.

    Returns:
        list[list[float]]: Lista de vectores numéricos.
    """
    return embeddings_model.encode(texts).tolist()


def initialize_client(doc_id):
    """
    Inicializa el cliente de base de datos vectorial ChromaDB y crea una colección.

    Args:
        doc_id (str): Identificador único usado como nombre de la colección y ruta.

    Returns:
        chromadb.Collection: La colección de ChromaDB lista para insertar datos.
    """
    try:
        client = chromadb.PersistentClient(
            path=f"{settings.CHROMA_PERSIST_DIR}/{doc_id}"
        )
        collection = client.get_or_create_collection(
            name=doc_id,
        )
        return collection
    except Exception as e:
        logger.error(f"Error al inicializar ChromaDB para {doc_id}: {str(e)}")
        raise VectorStoreException(
            f"Error de conexión con la base de datos vectorial: {str(e)}"
        )


def insert_chunks(collection, texts, vectors, final_chunks):
    """
    Inserta los fragmentos de texto, sus vectores y metadatos en la colección de ChromaDB.

    Args:
        collection: Colección de ChromaDB activa.
        texts (list[str]): Contenido textual de los fragmentos.
        vectors (list): Embeddings correspondientes.
        final_chunks (list[Document]): Fragmentos originales para extraer metadatos.
    """
    try:
        collection.add(
            ids=[str(uuid4()) for _ in final_chunks],
            documents=texts,  # Contenido textual para recuperarlo despues
            metadatas=[chunk.metadata for chunk in final_chunks],  # Info extra
            embeddings=vectors,  # Representación numérica para la búsqueda semántica
        )
    except Exception as e:
        logger.error(f"Error al insertar chunks en ChromaDB: {str(e)}")
        raise VectorStoreException(
            f"No se pudieron guardar los fragmentos en la base de datos: {str(e)}"
        )


def ingest(pdf_path, doc_id, embeddings_model):
    """
    Orquestador del pipeline RAG: extrae, fragmenta, vectoriza y almacena un PDF.

    Este proceso actualiza periódicamente el 'progress_store' para que el frontend
    pueda mostrar el estado de la carga en tiempo real.

    Args:
        pdf_path (str): Ruta del archivo PDF a procesar.
        doc_id (str): ID único para el documento.
        embeddings_model: Instancia del modelo de embeddings a utilizar.

    Returns:
        int: Número total de fragmentos insertados en la base de datos.
    """
    set_progress(doc_id, IngestionStatus.PROCESSING, 0)

    # Coge el texto del pdf
    documents_langchain = get_text_pdf(pdf_path, doc_id)

    # Validación de PDF vacío
    if not documents_langchain or all(
        not d.page_content.strip() for d in documents_langchain
    ):
        set_progress(doc_id, IngestionStatus.FAILED, 0)
        logger.warning(f"El documento {doc_id} está vacío o no tiene texto legible.")
        raise PDFEmptyException("El archivo PDF no contiene texto extraíble.")

    set_progress(doc_id, IngestionStatus.PROCESSING, 25)

    # Fragmentación
    final_chunks = create_chunks(documents_langchain)
    set_progress(doc_id, IngestionStatus.PROCESSING, 50)

    # Extrae solo el texto de cada chunk para generar los vectores
    texts = extract_texts(final_chunks)

    # Vectorización
    vectors = create_embeddings(texts, embeddings_model)
    set_progress(doc_id, IngestionStatus.PROCESSING, 75)

    # Inicializa el cliente de base de datos
    collection = initialize_client(doc_id)

    # Inserta los datos
    insert_chunks(collection, texts, vectors, final_chunks)

    set_progress(doc_id, IngestionStatus.READY, 100)

    return collection.count()
