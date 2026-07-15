"""Process-local storage for indexed documents."""

from typing import Any


class DocumentStore:
    """Map document identifiers to FAISS indexes and their chunk metadata."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, Any]] = {}

    def store(self, document_id: str, index: Any, metadata: list[dict[str, Any]]) -> None:
        """Store an index and its metadata under a document identifier."""

        self._documents[document_id] = {"index": index, "metadata": metadata}

    def get(self, document_id: str) -> dict[str, Any] | None:
        """Return stored document data, or ``None`` when it is unknown."""

        return self._documents.get(document_id)

    def exists(self, document_id: str) -> bool:
        """Return whether a document identifier has been indexed."""

        return document_id in self._documents


document_store = DocumentStore()
