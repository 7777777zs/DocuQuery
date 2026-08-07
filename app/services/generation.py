"""Prompt construction and cited answer generation."""

import re
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.models.schemas import Citation

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Use the provided context to answer questions. "
    "If the user references something from a previous answer, use the conversation history "
    "to understand what they mean. Answer using ONLY the provided context. Always cite sources "
    "using [Source: filename, Page: N] format. If the context doesn't contain the answer, say so."
)
_CITATION_PATTERN = re.compile(r"\[Source:\s*(.*?),\s*Page:\s*(\d+)\]")


def generate_answer(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Generate an answer from retrieved chunks and resolve its cited sources."""

    settings = get_settings()
    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[{index}]\n"
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content: {chunk['content']}"
        )
    context = "\n\n".join(context_blocks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    model = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    history_messages = [
        (message["role"], message["content"])
        for message in (chat_history or [])
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]
    response = model.invoke([("system", SYSTEM_PROMPT), *history_messages, ("human", prompt)])
    answer = str(response.content)

    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()
    for source, page_text in _CITATION_PATTERN.findall(answer):
        page = int(page_text)
        key = (source.strip(), page)
        if key in seen:
            continue
        seen.add(key)
        matching_chunk = next(
            (
                chunk
                for chunk in retrieved_chunks
                if chunk["source"] == key[0] and int(chunk["page"]) == page
            ),
            None,
        )
        citations.append(
            Citation(
                content=matching_chunk["content"] if matching_chunk else "",
                source=key[0],
                page=page,
            )
        )

    return {"answer": answer, "citations": citations}
