import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import chromadb
from uuid import uuid4

from app.core.config import settings
from app.rag.progress import progress_store

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Tamaño máximo de cada trozo (caracteres)
    chunk_overlap=150,  # Cuántos caracteres se repiten del trozo anterior
)

embeddings = SentenceTransformer("all-MiniLM-L6-v2")


def update_progress_store(doc_id, status, percent):
    progress_store[doc_id] = {"status": status, "progress": percent}


def ingest(pdf_path, doc_id):
    update_progress_store(doc_id, "processing", 0)

    doc = fitz.open(pdf_path)

    documents_langchain = [
        Document(
            page_content=page.get_text(),
            metadata={"page": page.number, "doc_id": doc_id},
        )
        for page in doc
    ]

    doc.close()

    update_progress_store(doc_id, "processing", 25)

    # Crea una lista de chunks
    final_chunks = text_splitter.split_documents(documents_langchain)
    update_progress_store(doc_id, "processing", 50)

    # Extrae solo el texto de cada chunk para generar los vectores
    texts = [chunk.page_content for chunk in final_chunks]

    # Genera los vectores (embeddings)
    vectors = embeddings.encode(texts).tolist()
    update_progress_store(doc_id, "processing", 75)

    # Inicializa el cliente de base de datos persistente
    client = chromadb.PersistentClient(path=f"{settings.CHROMA_PERSIST_DIR}/{doc_id}")

    # Crea o recupera una colección específica
    collection = client.get_or_create_collection(
        name=doc_id,
    )

    # Inserta los fragmentos, sus vectores y metadatos en la base de datos
    collection.add(
        ids=[str(uuid4()) for chunk in final_chunks],
        documents=texts,  # Contenido textual para recuperarlo despues
        metadatas=[chunk.metadata for chunk in final_chunks],  # Info extra
        embeddings=vectors,  # Representación numérica para la búsqueda semántica
    )

    update_progress_store(doc_id, "ready", 100)

    return (
        collection.count()
    )  # Devuelve el total de fragmentos almacenados en la colección
