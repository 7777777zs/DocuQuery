"""Document question-answering endpoint."""

from fastapi import APIRouter, HTTPException, status

from app.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse
from app.services.generation import generate_answer
from app.services.retrieval import retrieve
from app.store.document_store import document_store

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    """Retrieve context for a question and return a cited generated answer."""

    stored_document = document_store.get(request.document_id)
    if stored_document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    settings = get_settings()
    try:
        retrieved_chunks = retrieve(
            request.question,
            stored_document["index"],
            stored_document["metadata"],
            settings.TOP_K,
        )
        generated = generate_answer(request.question, retrieved_chunks)
        return QueryResponse(**generated)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to answer query") from exc
