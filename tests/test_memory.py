"""Async coverage for document-scoped conversation memory."""

from __future__ import annotations

import os
import textwrap
from collections.abc import AsyncIterator, Awaitable, Callable
from io import BytesIO
from typing import Any

import httpx
import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.config import get_settings
from app.main import app
from app.routers import query as query_router
from app.routers import upload as upload_router
from app.store.session_store import SessionStore

PDF_TEXT = (
    "The company reported Q3 revenue of $4.2 million, up 15% year-over-year. "
    "Operating expenses were $3.1 million. Net profit was $1.1 million. The CEO "
    "John Smith stated the growth was driven by expansion into European markets. "
    "The CTO Jane Doe highlighted three key technical achievements: migration to "
    "cloud infrastructure, launch of the mobile app, and implementation of real-time "
    "analytics. The board approved a $500,000 R&D budget increase for Q4."
)


def _escape_pdf_text(value: str) -> str:
    """Escape text for use inside a PDF literal string."""

    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for anyio-powered async tests."""

    return "asyncio"


@pytest.fixture
def test_pdf() -> bytes:
    """Create a one-page, extractable-text PDF in memory using pypdf."""

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_reference}
            )
        }
    )

    commands = ["BT", "/F1 11 Tf", "72 720 Td"]
    for line in textwrap.wrap(PDF_TEXT, width=78):
        commands.append(f"({_escape_pdf_text(line)}) Tj")
        commands.append("0 -15 Td")
    commands.append("ET")
    content_stream = DecodedStreamObject()
    content_stream.set_data("\n".join(commands).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(content_stream)

    output = BytesIO()
    writer.write(output)
    pdf_bytes = output.getvalue()
    assert "Q3 revenue" in PdfReader(BytesIO(pdf_bytes)).pages[0].extract_text()
    return pdf_bytes


@pytest.fixture
def fresh_app() -> Any:
    """Reset the process-local session store for each test."""

    app.state.session_store = SessionStore()
    return app


@pytest.fixture
async def async_client(fresh_app: Any) -> AsyncIterator[httpx.AsyncClient]:
    """Provide an AsyncClient that sends requests directly to the FastAPI app."""

    transport = httpx.ASGITransport(app=fresh_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client


@pytest.fixture
def upload_pdf() -> Callable[[httpx.AsyncClient, bytes], Awaitable[dict[str, str]]]:
    """Upload fixture bytes and return the document and session identifiers."""

    async def _upload(client: httpx.AsyncClient, pdf_bytes: bytes) -> dict[str, str]:
        response = await client.post(
            "/upload",
            files={"file": ("financial_report.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        return {"document_id": payload["document_id"], "session_id": payload["session_id"]}

    return _upload


@pytest.fixture
def offline_services(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, str]]]:
    """Patch external RAG boundaries and retain histories supplied to generation."""

    captured_histories: list[list[dict[str, str]]] = []

    def fake_ingest(file_bytes: bytes, filename: str) -> tuple[object, list[dict[str, Any]]]:
        assert file_bytes
        return object(), [
            {
                "content": PDF_TEXT,
                "source": filename,
                "page": 1,
                "chunk_index": 0,
            }
        ]

    def fake_retrieve(*_: Any, **__: Any) -> list[dict[str, Any]]:
        return [
            {
                "content": PDF_TEXT,
                "source": "financial_report.pdf",
                "page": 1,
                "score": 0.0,
            }
        ]

    def fake_generate(
        question: str,
        _: list[dict[str, Any]],
        chat_history: list[dict[str, str]],
    ) -> dict[str, Any]:
        captured_histories.append(chat_history)
        return {"answer": f"Answer for: {question}", "citations": []}

    monkeypatch.setattr(upload_router, "ingest_document", fake_ingest)
    monkeypatch.setattr(query_router, "retrieve", fake_retrieve)
    monkeypatch.setattr(query_router, "generate_answer", fake_generate)
    return captured_histories


def _integration_enabled() -> bool:
    """Return whether the opt-in, billed OpenAI integration suite may run."""

    get_settings.cache_clear()
    return bool(get_settings().OPENAI_API_KEY) and os.getenv("RUN_INTEGRATION_TESTS") == "1"


requires_openai = pytest.mark.skipif(
    not _integration_enabled(),
    reason="Set OPENAI_API_KEY and RUN_INTEGRATION_TESTS=1 to run OpenAI integration tests.",
)


@pytest.mark.anyio
async def test_upload_returns_session_id(
    async_client: httpx.AsyncClient,
    offline_services: list[list[dict[str, str]]],
    test_pdf: bytes,
    upload_pdf: Callable[[httpx.AsyncClient, bytes], Awaitable[dict[str, str]]],
) -> None:
    """Uploading a PDF creates non-empty document and session IDs."""

    payload = await upload_pdf(async_client, test_pdf)

    assert isinstance(payload["document_id"], str) and payload["document_id"]
    assert isinstance(payload["session_id"], str) and payload["session_id"]


@pytest.mark.anyio
async def test_first_query_creates_session_if_none_provided(
    async_client: httpx.AsyncClient,
    offline_services: list[list[dict[str, str]]],
    test_pdf: bytes,
    upload_pdf: Callable[[httpx.AsyncClient, bytes], Awaitable[dict[str, str]]],
) -> None:
    """A query without session_id receives a new session and non-empty answer."""

    uploaded = await upload_pdf(async_client, test_pdf)
    response = await async_client.post(
        "/query",
        json={"question": "What was the Q3 revenue?", "document_id": uploaded["document_id"]},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["answer"]
    assert payload["session_id"]
    assert payload["session_id"] != uploaded["session_id"]


@pytest.mark.integration
@requires_openai
@pytest.mark.anyio
async def test_follow_up_uses_context(
    async_client: httpx.AsyncClient,
    test_pdf: bytes,
    upload_pdf: Callable[[httpx.AsyncClient, bytes], Awaitable[dict[str, str]]],
) -> None:
    """Requires OPENAI_API_KEY: verifies a revenue follow-up resolves its referent."""

    uploaded = await upload_pdf(async_client, test_pdf)
    first = await async_client.post(
        "/query",
        json={
            "question": "What was the Q3 revenue?",
            "document_id": uploaded["document_id"],
            "session_id": uploaded["session_id"],
        },
    )
    assert first.status_code == 200
    assert "4.2" in first.json()["answer"].lower() or "million" in first.json()["answer"].lower()

    follow_up = await async_client.post(
        "/query",
        json={
            "question": "How much did it grow?",
            "document_id": uploaded["document_id"],
            "session_id": first.json()["session_id"],
        },
    )
    assert follow_up.status_code == 200
    assert "15" in follow_up.json()["answer"].lower() or "percent" in follow_up.json()["answer"].lower()


@pytest.mark.integration
@requires_openai
@pytest.mark.anyio
async def test_pronoun_resolution_across_turns(
    async_client: httpx.AsyncClient,
    test_pdf: bytes,
    upload_pdf: Callable[[httpx.AsyncClient, bytes], Awaitable[dict[str, str]]],
) -> None:
    """Requires OPENAI_API_KEY: verifies 'he' resolves to John Smith."""

    uploaded = await upload_pdf(async_client, test_pdf)
    first = await async_client.post(
        "/query",
        json={
            "question": "Who is the CEO?",
            "document_id": uploaded["document_id"],
            "session_id": uploaded["session_id"],
        },
    )
    assert first.status_code == 200
    assert "john smith" in first.json()["answer"].lower()

    follow_up = await async_client.post(
        "/query",
        json={
            "question": "What did he say?",
            "document_id": uploaded["document_id"],
            "session_id": first.json()["session_id"],
        },
    )
    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "growth" in answer or "european" in answer


@pytest.mark.integration
@requires_openai
@pytest.mark.anyio
async def test_multi_turn_chain(
    async_client: httpx.AsyncClient,
    test_pdf: bytes,
    upload_pdf: Callable[[httpx.AsyncClient, bytes], Awaitable[dict[str, str]]],
) -> None:
    """Requires OPENAI_API_KEY: verifies multi-turn achievement ordering context."""

    uploaded = await upload_pdf(async_client, test_pdf)
    first = await async_client.post(
        "/query",
        json={
            "question": "What were the three technical achievements?",
            "document_id": uploaded["document_id"],
            "session_id": uploaded["session_id"],
        },
    )
    assert first.status_code == 200
    first_answer = first.json()["answer"].lower()
    assert "cloud" in first_answer and "mobile" in first_answer and "analytics" in first_answer

    second = await async_client.post(
        "/query",
        json={
            "question": "Which one of those was mentioned first?",
            "document_id": uploaded["document_id"],
            "session_id": first.json()["session_id"],
        },
    )
    assert second.status_code == 200
    assert "cloud" in second.json()["answer"].lower()

    third = await async_client.post(
        "/query",
        json={
            "question": "Summarize everything we've discussed so far.",
            "document_id": uploaded["document_id"],
            "session_id": second.json()["session_id"],
        },
    )
    assert third.status_code == 200
    third_answer = third.json()["answer"].lower()
    assert "technical" in third_answer and ("first" in third_answer or "cloud" in third_answer)


@pytest.mark.integration
@requires_openai
@pytest.mark.anyio
async def test_separate_sessions_are_isolated(
    async_client: httpx.AsyncClient,
    test_pdf: bytes,
    upload_pdf: Callable[[httpx.AsyncClient, bytes], Awaitable[dict[str, str]]],
) -> None:
    """Requires OPENAI_API_KEY: verifies one document's sessions do not share history."""

    uploaded = await upload_pdf(async_client, test_pdf)
    session_a = await async_client.post(
        "/query",
        json={"question": "What was the revenue?", "document_id": uploaded["document_id"]},
    )
    session_b = await async_client.post(
        "/query",
        json={"question": "Who is the CTO?", "document_id": uploaded["document_id"]},
    )
    assert session_a.status_code == 200 and session_b.status_code == 200
    session_id_a = session_a.json()["session_id"]
    session_id_b = session_b.json()["session_id"]
    assert session_id_a != session_id_b

    follow_a = await async_client.post(
        "/query",
        json={
            "question": "What did I just ask about?",
            "document_id": uploaded["document_id"],
            "session_id": session_id_a,
        },
    )
    follow_b = await async_client.post(
        "/query",
        json={
            "question": "What did I just ask about?",
            "document_id": uploaded["document_id"],
            "session_id": session_id_b,
        },
    )
    assert "revenue" in follow_a.json()["answer"].lower()
    assert "cto" in follow_b.json()["answer"].lower() or "jane doe" in follow_b.json()["answer"].lower()


@pytest.mark.anyio
async def test_invalid_session_id_returns_error(
    async_client: httpx.AsyncClient,
    offline_services: list[list[dict[str, str]]],
    test_pdf: bytes,
    upload_pdf: Callable[[httpx.AsyncClient, bytes], Awaitable[dict[str, str]]],
) -> None:
    """Unknown sessions receive a clear not-found response without model calls."""

    uploaded = await upload_pdf(async_client, test_pdf)
    response = await async_client.post(
        "/query",
        json={
            "question": "What was the revenue?",
            "document_id": uploaded["document_id"],
            "session_id": "nonexistent-session-id-12345",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


@pytest.mark.anyio
async def test_history_trimming(
    async_client: httpx.AsyncClient,
    fresh_app: Any,
    offline_services: list[list[dict[str, str]]],
    test_pdf: bytes,
    upload_pdf: Callable[[httpx.AsyncClient, bytes], Awaitable[dict[str, str]]],
) -> None:
    """Twelve exchanges retain only the newest ten and forget the first question."""

    uploaded = await upload_pdf(async_client, test_pdf)
    session_id = uploaded["session_id"]
    for number in range(1, 13):
        response = await async_client.post(
            "/query",
            json={
                "question": f"question-{number}",
                "document_id": uploaded["document_id"],
                "session_id": session_id,
            },
        )
        assert response.status_code == 200

    history = fresh_app.state.session_store.get_history(session_id)
    assert len(history) == 20
    assert history[0] == {"role": "user", "content": "question-3"}

    follow_up = await async_client.post(
        "/query",
        json={
            "question": "What was question-1?",
            "document_id": uploaded["document_id"],
            "session_id": session_id,
        },
    )
    assert follow_up.status_code == 200
    assert all(message["content"] != "question-1" for message in offline_services[-1])


@pytest.mark.anyio
async def test_session_id_persists_across_queries(
    async_client: httpx.AsyncClient,
    offline_services: list[list[dict[str, str]]],
    test_pdf: bytes,
    upload_pdf: Callable[[httpx.AsyncClient, bytes], Awaitable[dict[str, str]]],
) -> None:
    """Every query that supplies a session receives the same session ID back."""

    uploaded = await upload_pdf(async_client, test_pdf)
    session_id = uploaded["session_id"]
    returned_ids = []
    for question in ("What was the revenue?", "Who is the CEO?", "What was the profit?"):
        response = await async_client.post(
            "/query",
            json={
                "question": question,
                "document_id": uploaded["document_id"],
                "session_id": session_id,
            },
        )
        assert response.status_code == 200
        returned_ids.append(response.json()["session_id"])

    assert returned_ids == [session_id, session_id, session_id]
