import chromadb

from app.core.config import settings
from app.core.ml_models import embeddings


def retrieve(query: str, doc_id: str):
    # Conecta al cliente
    client = chromadb.PersistentClient(path=f"{settings.CHROMA_PERSIST_DIR}/{doc_id}")

    # Accede a la collección
    collection = client.get_or_create_collection(name=doc_id)

    # Genera el embeddings de la consulta
    query_vectors = embeddings.encode(query).tolist()

    # Realiza la búsqueda semántica, return [ [res1, res2...] ]
    results = collection.query(
        query_embeddings=[query_vectors],
        n_results=10,
    )

    formatted_results = [
        {"text": text, "page": meta.get("page", 0), "score": round(1 - dist, 4)}
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

    return formatted_results
