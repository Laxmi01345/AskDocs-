import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatTurn:
    user_message: str
    assistant_message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    doc_id: str
    turns: list = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    @property
    def turn_count(self):
        return len(self.turns)


class SessionManager:
    def __init__(self, max_sessions=1000, ttl_seconds=3600):
        self._sessions = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds

    def create_session(self, doc_id):
        session_id = str(uuid.uuid4())
        with self._lock:
            self._evict_if_needed()
            self._sessions[session_id] = Session(session_id=session_id, doc_id=doc_id)
        return session_id

    def get_session(self, session_id):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session.last_accessed > self._ttl_seconds:
                del self._sessions[session_id]
                return None
            session.last_accessed = time.time()
            return session

    def add_turn(self, session_id, user_message, assistant_message):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.turns.append(ChatTurn(user_message=user_message, assistant_message=assistant_message))
            session.last_accessed = time.time()
            return session

    def set_summary(self, session_id, summary):
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.summary = summary

    def delete_session(self, session_id):
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def _evict_if_needed(self):
        if len(self._sessions) >= self._max_sessions:
            sorted_sessions = sorted(self._sessions.items(), key=lambda x: x[1].last_accessed)
            evict_count = max(1, self._max_sessions // 10)
            for sid, _ in sorted_sessions[:evict_count]:
                del self._sessions[sid]


session_manager = SessionManager()
