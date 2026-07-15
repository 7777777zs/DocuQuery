"""PDF upload endpoint."""

from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, status
from pypdf.errors import PdfReadError

from app.models.schemas import UploadResponse
from app.services.ingestion import ingest_document
from app.store.document_store import document_store

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    """Validate, ingest, and store an uploaded PDF."""

    filename = file.filename or ""
    is_pdf_mime = file.content_type == "application/pdf"
    is_pdf_filename = filename.lower().endswith(".pdf")
    if not is_pdf_mime or not is_pdf_filename:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported",
        )

    file_bytes = await file.read()
    try:
        index, metadata = ingest_document(file_bytes, filename)
    except PdfReadError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to ingest PDF") from exc

    document_id = str(uuid4())
    document_store.store(document_id, index, metadata)
    return UploadResponse(document_id=document_id)
