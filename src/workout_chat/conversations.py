"""Stored conversations: what the sidebar lists and what a reloaded page restores.

Streamlit-free on purpose. Two stores share one small interface:
  * MemoryStore    - process memory. Used for guests, for tests, and for Kevin when Firestore is unavailable.
                     Survives page reloads, not app restarts.
  * FirestoreStore - Kevin's history in Firestore (same GCP project + service account as the Sheet).
                     Keeps only the newest ``KEVIN_HISTORY_LIMIT`` conversations.

A conversation's id is also its LangGraph thread_id. ``messages`` holds the same dicts app.py renders
(the greeting is not stored; it is rebuilt from the current data every time).
"""
from __future__ import annotations

import copy
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

logger = logging.getLogger(__name__)

TITLE_CHARS = 60
FIRESTORE_MAX_BYTES = 900_000  # Firestore's hard limit is 1 MiB per document; leave headroom for field names


@dataclass
class Conversation:
    id: str
    owner: str
    messages: list[dict] = field(default_factory=list)
    title: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add(self, message: dict) -> None:
        self.messages.append(message)
        if not self.title and message.get("role") == "user":
            self.title = make_title(message.get("content", ""))
        self.updated_at = time.time()


@dataclass(frozen=True)
class Summary:
    """One sidebar row."""
    id: str
    title: str
    updated_at: float


class ConversationStore(Protocol):
    persistent: bool  # True when conversations survive an app restart

    def get(self, conv_id: str) -> Conversation | None: ...
    def save(self, conv: Conversation) -> None: ...
    def list_recent(self, owner: str, limit: int) -> list[Summary]: ...
    def prune(self, owner: str, keep: int) -> None: ...


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def make_title(question: str) -> str:
    text = " ".join(str(question).split())
    return text if len(text) <= TITLE_CHARS else text[: TITLE_CHARS - 1].rstrip() + "…"


def assistant_message(result, question: str) -> dict:
    """The stored assistant reply for a ``graph.TurnResult`` (``debug`` is only shown to Kevin)."""
    debug = ""
    if result.route:
        debug = f"route: {result.route}"
        if result.models_used:
            debug += " · models: " + ", ".join(f"{k}={v}" for k, v in result.models_used.items())
        if result.standalone_question and result.standalone_question != question:
            debug += f" · understood as: {result.standalone_question}"
        if result.error:
            debug += f" · detail: {result.error}"
    return {
        "role": "assistant", "content": result.answer, "figure_json": result.figure_json,
        "code": result.code, "language": result.language, "debug": debug,
    }


def history_pairs(messages: Iterable[dict], max_messages: int) -> list[tuple[str, str]]:
    """The last ``max_messages`` (role, text) pairs, used to rebuild a LangGraph thread after a restart."""
    pairs = [(m["role"], str(m.get("content") or "")) for m in messages if m.get("role") in ("user", "assistant")]
    return pairs[-max_messages:] if max_messages > 0 else []


def relative_time(then: float, now: float | None = None) -> str:
    seconds = max(0, int((now if now is not None else time.time()) - then))
    if seconds < 60:
        return "just now"
    for size, unit in ((86_400, "d"), (3_600, "h"), (60, "m")):
        if seconds >= size:
            return f"{seconds // size}{unit} ago"
    return "just now"


def fit_messages(messages: list[dict], max_bytes: int) -> list[dict]:
    """A copy of ``messages`` that JSON-encodes to at most ``max_bytes``.

    Charts go first, oldest first (each marked ``figure_dropped`` so the UI can say so).
    Then, only if still too big, the oldest messages. The newest message is always kept.
    """
    out = copy.deepcopy(messages)

    def size() -> int:
        return len(json.dumps(out, ensure_ascii=False).encode("utf-8"))

    if size() <= max_bytes:
        return out
    for msg in out:
        if msg.get("figure_json"):
            msg["figure_json"] = None
            msg["figure_dropped"] = True
            if size() <= max_bytes:
                return out
    while len(out) > 1 and size() > max_bytes:
        out.pop(0)
    return out


def record_reply(store: ConversationStore, snapshot: Conversation, message: dict, keep: int | None) -> None:
    """Append the assistant ``message`` to the stored conversation and save it (runs in the background job).

    ``snapshot`` is the conversation as the page last saved it; it is used if the stored copy is missing
    (for example the earlier save failed). ``keep`` prunes the owner's history to that many conversations.
    """
    conv = store.get(snapshot.id) or copy.deepcopy(snapshot)
    conv.add(message)
    store.save(conv)
    if keep:
        try:
            store.prune(conv.owner, keep)
        except Exception:  # noqa: BLE001 - pruning is housekeeping; never lose the answer over it
            logger.exception("Pruning old conversations failed")


# --------------------------------------------------------------------------
# Process-memory store
# --------------------------------------------------------------------------
class MemoryStore:
    persistent = False

    def __init__(self, max_items: int = 500):
        self._items: dict[str, Conversation] = {}
        self._lock = threading.Lock()
        self._max_items = max_items

    def get(self, conv_id: str) -> Conversation | None:
        with self._lock:
            conv = self._items.get(conv_id)
            return copy.deepcopy(conv) if conv else None

    def save(self, conv: Conversation) -> None:
        with self._lock:
            self._items[conv.id] = copy.deepcopy(conv)
            if len(self._items) > self._max_items:
                oldest = sorted(self._items.values(), key=lambda c: c.updated_at)
                for c in oldest[: len(self._items) - self._max_items]:
                    del self._items[c.id]

    def _newest(self, owner: str) -> list[Conversation]:
        return sorted((c for c in self._items.values() if c.owner == owner), key=lambda c: c.updated_at, reverse=True)

    def list_recent(self, owner: str, limit: int) -> list[Summary]:
        with self._lock:
            return [Summary(c.id, c.title, c.updated_at) for c in self._newest(owner)[:limit]]

    def prune(self, owner: str, keep: int) -> None:
        with self._lock:
            for c in self._newest(owner)[keep:]:
                del self._items[c.id]


# --------------------------------------------------------------------------
# Firestore store (one collection, one owner)
# --------------------------------------------------------------------------
class FirestoreStore:
    """Conversations of one ``owner`` in one Firestore collection (doc id = conversation id).

    One collection per owner keeps every query on the automatic single-field index for ``updated_at``.
    """
    persistent = True

    def __init__(self, client: Any, collection: str, owner: str, *, max_bytes: int = FIRESTORE_MAX_BYTES,
                 timeout: float = 10.0):
        self._col = client.collection(collection)
        self.owner = owner
        self._max_bytes = max_bytes
        self._timeout = timeout

    def _newest_first(self):
        return self._col.order_by("updated_at", direction="DESCENDING")

    def get(self, conv_id: str) -> Conversation | None:
        snap = self._col.document(conv_id).get(timeout=self._timeout)
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        return Conversation(
            id=conv_id, owner=self.owner, messages=list(data.get("messages") or []), title=data.get("title", ""),
            created_at=float(data.get("created_at") or 0), updated_at=float(data.get("updated_at") or 0),
        )

    def save(self, conv: Conversation) -> None:
        if conv.owner != self.owner:
            raise ValueError(f"This store only holds {self.owner}'s conversations")
        self._col.document(conv.id).set({
            "title": conv.title, "created_at": conv.created_at, "updated_at": conv.updated_at,
            "messages": fit_messages(conv.messages, self._max_bytes),
        }, timeout=self._timeout)

    def delete(self, conv_id: str) -> None:
        self._col.document(conv_id).delete(timeout=self._timeout)

    def list_recent(self, owner: str, limit: int) -> list[Summary]:
        if owner != self.owner:
            return []
        query = self._newest_first().limit(limit).select(["title", "updated_at"])
        out = []
        for snap in query.stream(timeout=self._timeout):
            data = snap.to_dict() or {}
            out.append(Summary(snap.id, data.get("title", ""), float(data.get("updated_at") or 0)))
        return out

    def prune(self, owner: str, keep: int) -> None:
        if owner != self.owner:
            return
        for snap in self._newest_first().offset(keep).select(["updated_at"]).stream(timeout=self._timeout):
            snap.reference.delete(timeout=self._timeout)


def make_firestore_client(service_account_info: dict):
    """A Firestore client for the service account's own project (imported lazily: the library is heavy)."""
    from google.cloud import firestore
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(service_account_info)
    return firestore.Client(project=service_account_info.get("project_id"), credentials=creds)
