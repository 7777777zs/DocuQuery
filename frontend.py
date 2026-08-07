"""Streamlit interface for uploading PDFs and querying DocuQuery."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, MutableMapping
from typing import Any

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 120


class ApiRequestError(RuntimeError):
    """Raised when the DocuQuery API cannot satisfy a frontend request."""


def initialize_session_state(state: MutableMapping[str, Any]) -> None:
    """Initialize the document and chat values used for one Streamlit session."""

    state.setdefault("document_id", None)
    state.setdefault("session_id", None)
    state.setdefault("chat_history", [])
    state.setdefault("uploaded_file_hash", None)
    state.setdefault("uploaded_filename", None)


def _error_message(error: requests.RequestException) -> str:
    """Convert an HTTP or connection failure into a concise user-facing message."""

    response = getattr(error, "response", None)
    if response is not None:
        try:
            payload = response.json()
            detail = payload.get("detail") if isinstance(payload, dict) else None
        except ValueError:
            detail = None
        if detail:
            return str(detail)
        return f"The API returned HTTP {response.status_code}."
    if isinstance(error, requests.Timeout):
        return "The API request timed out. Please try again."
    if isinstance(error, requests.ConnectionError):
        return "Cannot reach the DocuQuery API at http://localhost:8000."
    return "The API request failed. Please try again."


def upload_document(
    filename: str,
    file_bytes: bytes,
    post: Callable[..., requests.Response] = requests.post,
) -> dict[str, str]:
    """Upload a PDF to the API and return its document and session identifiers."""

    try:
        response = post(
            f"{API_BASE_URL}/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise ApiRequestError(_error_message(error)) from error

    payload = response.json()
    document_id = payload.get("document_id")
    session_id = payload.get("session_id")
    if not isinstance(document_id, str) or not document_id:
        raise ApiRequestError("The upload response did not include a document ID.")
    if not isinstance(session_id, str) or not session_id:
        raise ApiRequestError("The upload response did not include a session ID.")
    return {"document_id": document_id, "session_id": session_id}


def query_document(
    question: str,
    document_id: str,
    session_id: str | None,
    post: Callable[..., requests.Response] = requests.post,
) -> dict[str, Any]:
    """Submit a question to the API and return its answer and citations."""

    try:
        response = post(
            f"{API_BASE_URL}/query",
            json={
                "question": question,
                "document_id": document_id,
                "session_id": session_id,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise ApiRequestError(_error_message(error)) from error

    payload = response.json()
    answer = payload.get("answer")
    citations = payload.get("citations", [])
    returned_session_id = payload.get("session_id")
    if (
        not isinstance(answer, str)
        or not isinstance(citations, list)
        or not isinstance(returned_session_id, str)
        or not returned_session_id
    ):
        raise ApiRequestError("The query response was not in the expected format.")
    return {"answer": answer, "citations": citations, "session_id": returned_session_id}


def render_citations(citations: list[dict[str, Any]]) -> None:
    """Render answer citations below an assistant message."""

    if not citations:
        return

    with st.expander(f"Sources ({len(citations)})"):
        for citation in citations:
            source = citation.get("source", "Unknown source")
            page = citation.get("page", "?")
            content = citation.get("content", "")
            st.markdown(f"**{source} — page {page}**")
            st.caption(content)


def render_chat_history(chat_history: list[dict[str, Any]]) -> None:
    """Render the messages retained in the current browser session."""

    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citations(message.get("citations", []))


def main() -> None:
    """Render the DocuQuery upload and question-answering interface."""

    st.set_page_config(page_title="DocuQuery", page_icon="📄", layout="centered")
    initialize_session_state(st.session_state)

    st.title("DocuQuery")
    st.caption("Upload one PDF, then ask grounded questions about its contents.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if file_hash != st.session_state.uploaded_file_hash:
            with st.spinner("Indexing PDF..."):
                try:
                    upload_result = upload_document(uploaded_file.name, file_bytes)
                except ApiRequestError as error:
                    st.error(f"Upload failed: {error}")
                else:
                    st.session_state.document_id = upload_result["document_id"]
                    st.session_state.session_id = upload_result["session_id"]
                    st.session_state.uploaded_file_hash = file_hash
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"Indexed {uploaded_file.name}")

    document_id = st.session_state.document_id
    if not document_id:
        st.info("Upload a PDF to begin asking questions.")
        return

    st.caption(f"Ready to answer questions about {st.session_state.uploaded_filename}.")
    render_chat_history(st.session_state.chat_history)

    question = st.chat_input("Ask a question about this document")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            try:
                result = query_document(question, document_id, st.session_state.session_id)
            except ApiRequestError as error:
                st.error(f"Question failed: {error}")
                return

        st.markdown(result["answer"])
        render_citations(result["citations"])
        st.session_state.session_id = result["session_id"]
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"],
            }
        )


if __name__ == "__main__":
    main()
