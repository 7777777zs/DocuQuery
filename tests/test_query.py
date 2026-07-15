from fastapi.testclient import TestClient

from app.main import app
from app.routers import query as query_router


def test_health_endpoint_reports_ok():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_not_found_for_unknown_document():
    response = TestClient(app).post(
        "/query",
        json={"question": "What is this about?", "document_id": "missing"},
    )

    assert response.status_code == 404


def test_query_returns_answer_with_citations(monkeypatch):
    query_router.document_store.store("doc-1", object(), [{"source": "sample.pdf", "page": 1}])

    retrieved = [{"content": "The answer is 42.", "source": "sample.pdf", "page": 1, "score": 0.1}]
    monkeypatch.setattr(query_router, "retrieve", lambda *args, **kwargs: retrieved)
    monkeypatch.setattr(
        query_router,
        "generate_answer",
        lambda question, chunks: {
            "answer": "The answer is 42. [Source: sample.pdf, Page: 1]",
            "citations": [
                {"content": "The answer is 42.", "source": "sample.pdf", "page": 1}
            ],
        },
    )

    response = TestClient(app).post(
        "/query",
        json={"question": "What is the answer?", "document_id": "doc-1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The answer is 42. [Source: sample.pdf, Page: 1]",
        "citations": [
            {"content": "The answer is 42.", "source": "sample.pdf", "page": 1}
        ],
    }
