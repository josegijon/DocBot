from pathlib import Path
from uuid import uuid4

from app.rag.ingestor import ingest


def test_ingestor():
    doc_id = str(uuid4())
    fixtures_dir = Path(__file__).parent / "fixtures"
    file_path = fixtures_dir / "test.pdf"
    assert ingest(str(file_path), doc_id) > 0
