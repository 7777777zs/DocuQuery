from fastapi.testclient import TestClient

from app.main import app
from app.routers import upload as upload_router


def test_upload_accepts_pdf_and_returns_document_id(monkeypatch):
    fake_index = object()

    def fake_ingest(file_bytes: bytes, filename: str):
        assert file_bytes == b"%PDF-1.4 test"
        assert filename == "sample.pdf"
        return fake_index, [{"source": filename, "page": 1, "chunk_index": 0}]

    monkeypatch.setattr(upload_router, "ingest_document", fake_ingest)

    response = TestClient(app).post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert response.status_code == 200
    document_id = response.json()["document_id"]
    assert document_id
    assert upload_router.document_store.exists(document_id)


def test_upload_rejects_non_pdf_files():
    response = TestClient(app).post(
        "/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_rejects_pdf_extension_with_non_pdf_mime():
    response = TestClient(app).post(
        "/upload",
        files={"file": ("notes.pdf", b"plain text", "text/plain")},
    )

    assert response.status_code == 415
