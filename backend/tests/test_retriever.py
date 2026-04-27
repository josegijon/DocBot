from pathlib import Path
from uuid import uuid4

from app.rag.retriever import retrieve
from app.rag.ingestor import ingest


def test_retriever():
    doc_id = str(uuid4())
    file_path = Path(__file__).parent / "fixtures" / "test.pdf"
    ingest(str(file_path), doc_id)

    results = retrieve("Donde se realiza el analisis", doc_id)
    assert len(results) > 0
    assert "text" in results[0]
    assert "page" in results[0]
    assert "score" in results[0]
