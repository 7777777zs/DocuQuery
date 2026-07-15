"""Similarity retrieval from FAISS indexes."""

from typing import Any

from langchain_community.vectorstores import FAISS


def retrieve(
    query: str,
    faiss_index: FAISS,
    metadata: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Return the top matching chunks with source metadata and similarity scores."""

    matches = faiss_index.similarity_search_with_score(query, k=top_k)
    retrieved: list[dict[str, Any]] = []
    for document, score in matches:
        chunk_index = document.metadata.get("chunk_index")
        stored_metadata = (
            metadata[chunk_index]
            if isinstance(chunk_index, int) and 0 <= chunk_index < len(metadata)
            else document.metadata
        )
        retrieved.append(
            {
                "content": document.page_content,
                "source": stored_metadata.get("source", "unknown"),
                "page": int(stored_metadata.get("page", 0)),
                "score": float(score),
            }
        )
    return retrieved
