from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
import requests

import frontend


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self  # type: ignore[assignment]
            raise error


def test_initialize_session_state_sets_defaults_without_overwriting_history():
    state: dict[str, Any] = {"chat_history": [{"role": "user", "content": "hello"}]}

    frontend.initialize_session_state(state)

    assert state["document_id"] is None
    assert state["session_id"] is None
    assert state["uploaded_file_hash"] is None
    assert state["uploaded_filename"] is None
    assert state["chat_history"] == [{"role": "user", "content": "hello"}]


def test_upload_document_posts_pdf_and_returns_document_and_session_ids():
    post = Mock(return_value=FakeResponse({"document_id": "doc-123", "session_id": "session-123"}))

    result = frontend.upload_document("paper.pdf", b"pdf bytes", post)

    assert result == {"document_id": "doc-123", "session_id": "session-123"}
    post.assert_called_once_with(
        "http://localhost:8000/upload",
        files={"file": ("paper.pdf", b"pdf bytes", "application/pdf")},
        timeout=frontend.REQUEST_TIMEOUT_SECONDS,
    )


def test_query_document_posts_question_and_returns_answer_with_citations():
    post = Mock(
        return_value=FakeResponse(
            {
                "answer": "The answer is 42.",
                "session_id": "session-123",
                "citations": [
                    {"content": "42", "source": "paper.pdf", "page": 1}
                ],
            }
        )
    )

    result = frontend.query_document("What is the answer?", "doc-123", "session-123", post)

    assert result["answer"] == "The answer is 42."
    assert result["citations"] == [
        {"content": "42", "source": "paper.pdf", "page": 1}
    ]
    assert result["session_id"] == "session-123"
    post.assert_called_once_with(
        "http://localhost:8000/query",
        json={
            "question": "What is the answer?",
            "document_id": "doc-123",
            "session_id": "session-123",
        },
        timeout=frontend.REQUEST_TIMEOUT_SECONDS,
    )


def test_upload_document_raises_readable_error_for_api_failure():
    post = Mock(return_value=FakeResponse({"detail": "Only PDF files are supported"}, 415))

    with pytest.raises(frontend.ApiRequestError, match="Only PDF files are supported"):
        frontend.upload_document("notes.txt", b"text", post)


def test_render_citations_shows_source_page_and_content(monkeypatch):
    expander = Mock()
    expander.__enter__ = Mock(return_value=expander)
    expander.__exit__ = Mock(return_value=False)
    markdown = Mock()
    caption = Mock()
    monkeypatch.setattr(frontend.st, "expander", Mock(return_value=expander))
    monkeypatch.setattr(frontend.st, "markdown", markdown)
    monkeypatch.setattr(frontend.st, "caption", caption)

    frontend.render_citations(
        [{"source": "paper.pdf", "page": 3, "content": "Relevant passage."}]
    )

    markdown.assert_called_once_with("**paper.pdf — page 3**")
    caption.assert_called_once_with("Relevant passage.")
