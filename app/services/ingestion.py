"""PDF parsing, chunking, embedding, and FAISS index creation."""

from io import BytesIO
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import get_settings


def ingest_document(file_bytes: bytes, filename: str) -> tuple[FAISS, list[dict[str, Any]]]:
    """Parse a PDF, split its pages into chunks, embed them, and build a FAISS index."""

    settings = get_settings()
    reader = PdfReader(BytesIO(file_bytes))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    page_documents = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": filename, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]
    chunks = [
        chunk
        for page_document in page_documents
        if page_document.page_content.strip()
        for chunk in splitter.split_documents([page_document])
    ]

    if not chunks:
        raise ValueError("The PDF does not contain extractable text")

    metadata: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["source"] = filename
        chunk.metadata["chunk_index"] = chunk_index
        metadata.append(
            {
                "content": chunk.page_content,
                "source": filename,
                "page": int(chunk.metadata["page"]),
                "chunk_index": chunk_index,
            }
        )

    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    index = FAISS.from_documents(chunks, embeddings)
    return index, metadata
