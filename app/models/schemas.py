"""Request and response schemas for the DocuQuery API."""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned after a PDF is indexed."""

    document_id: str


class QueryRequest(BaseModel):
    """Question and document identifier supplied to the query endpoint."""

    question: str = Field(min_length=1)
    document_id: str = Field(min_length=1)


class Citation(BaseModel):
    """A source chunk cited by a generated answer."""

    content: str
    source: str
    page: int


class QueryResponse(BaseModel):
    """Answer and source citations returned by the query endpoint."""

    answer: str
    citations: list[Citation]
