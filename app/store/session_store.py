"""Process-local storage for document-scoped chat sessions."""

from typing import Literal
from uuid import uuid4

MessageRole = Literal["user", "assistant"]
ChatMessage = dict[str, str]


class SessionStore:
    """Store bounded conversation histories and their owning document IDs."""

    _MAX_MESSAGES = 20

    def __init__(self) -> None:
        self._sessions: dict[str, list[ChatMessage]] = {}
        self._document_ids: dict[str, str] = {}

    def create_session(self) -> str:
        """Create an empty session and return its UUID identifier."""

        session_id = str(uuid4())
        self._sessions[session_id] = []
        return session_id

    def bind_to_document(self, session_id: str, document_id: str) -> None:
        """Bind an existing session to the document that owns its conversation."""

        if not self.exists(session_id):
            raise KeyError(f"Unknown session ID: {session_id}")
        self._document_ids[session_id] = document_id

    def belongs_to_document(self, session_id: str, document_id: str) -> bool:
        """Return whether a session belongs to a given document identifier."""

        return self._document_ids.get(session_id) == document_id

    def add_message(self, session_id: str, role: MessageRole, content: str) -> None:
        """Append a user or assistant message and retain at most ten exchanges."""

        if not self.exists(session_id):
            raise KeyError(f"Unknown session ID: {session_id}")
        if role not in {"user", "assistant"}:
            raise ValueError("Session messages must have a user or assistant role")

        history = self._sessions[session_id]
        history.append({"role": role, "content": content})
        while len(history) > self._MAX_MESSAGES:
            del history[:2]

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Return a copy of the session history in chronological order."""

        if not self.exists(session_id):
            raise KeyError(f"Unknown session ID: {session_id}")
        return [message.copy() for message in self._sessions[session_id]]

    def clear_session(self, session_id: str) -> None:
        """Clear a session's conversation while retaining its document binding."""

        if not self.exists(session_id):
            raise KeyError(f"Unknown session ID: {session_id}")
        self._sessions[session_id].clear()

    def exists(self, session_id: str) -> bool:
        """Return whether a session identifier is known to this store."""

        return session_id in self._sessions
