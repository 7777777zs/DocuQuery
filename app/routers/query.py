"""Document question-answering endpoint."""

from fastapi import APIRouter, HTTPException, Request, status

from app.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse
from app.services.generation import generate_answer
from app.services.retrieval import retrieve
from app.store.document_store import document_store

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_document(query_request: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve context for a question and return a cited generated answer."""

    stored_document = document_store.get(query_request.document_id)
    if stored_document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    session_store = request.app.state.session_store
    session_id = query_request.session_id
    if session_id is None:
        session_id = session_store.create_session()
        session_store.bind_to_document(session_id, query_request.document_id)
    elif not session_store.exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    elif not session_store.belongs_to_document(session_id, query_request.document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session does not belong to this document",
        )

    chat_history = session_store.get_history(session_id)
    settings = get_settings()
    try:
        retrieved_chunks = retrieve(
            query_request.question,
            stored_document["index"],
            stored_document["metadata"],
            settings.TOP_K,
        )
        generated = generate_answer(query_request.question, retrieved_chunks, chat_history)
        session_store.add_message(session_id, "user", query_request.question)
        session_store.add_message(session_id, "assistant", generated["answer"])
        return QueryResponse(**generated, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to answer query") from exc
