"""Live check: can the app's service account read and write Kevin's conversation history in Firestore?

Run from the repo root with .streamlit/secrets.toml in place (after the one-time Firestore setup):
    python scripts/check_firestore.py

It writes one throwaway conversation, reads it back, lists the collection and deletes the test document.
Your real conversations are listed but never changed.
"""
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from workout_chat import access  # noqa: E402
from workout_chat.config import FIRESTORE_COLLECTION, KEVIN_HISTORY_LIMIT  # noqa: E402
from workout_chat.conversations import Conversation, FirestoreStore, make_firestore_client  # noqa: E402
from workout_chat.data import DataLoadError, service_account_info  # noqa: E402

# Most specific first: an API that is switched off also reports PermissionDenied.
HINTS = {
    "has not been used": "Enable the Cloud Firestore API for the project (step 2 of the setup guide).",
    "does not exist": "Create the (default) Firestore database in Native mode (step 3).",
    "NotFound": "Create the (default) Firestore database in Native mode (step 3).",
    "PermissionDenied": "Give the service account the 'Cloud Datastore User' role (step 4), then wait a minute.",
}


def main() -> int:
    try:
        info = service_account_info()
    except DataLoadError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"Project: {info.get('project_id')}   Service account: {info.get('client_email')}")
    if not info.get("project_id"):
        print("FAILED: the gcp_service_account secret has no project_id.")
        return 1

    test_id = f"healthcheck-{uuid.uuid4().hex[:8]}"
    try:
        store = FirestoreStore(make_firestore_client(info), FIRESTORE_COLLECTION, access.KEVIN)
        conv = Conversation(id=test_id, owner=access.KEVIN)
        conv.add({"role": "user", "content": "Firestore health check"})
        started = time.perf_counter()
        store.save(conv)
        got = store.get(test_id)
        elapsed = time.perf_counter() - started
        assert got is not None and got.messages == conv.messages, "read back something different"
        recent = store.list_recent(access.KEVIN, KEVIN_HISTORY_LIMIT + 1)
        store.delete(test_id)  # only the test doc; prune() is not used here on purpose
    except Exception as exc:  # noqa: BLE001 - report anything with a hint
        text = f"{type(exc).__name__}: {exc}"
        print(f"FAILED: {text.splitlines()[0][:300]}")
        for needle, hint in HINTS.items():
            if needle in text:
                print(f"Hint: {hint}")
                break
        return 1

    print(f"OK: wrote, read and deleted a test document in '{FIRESTORE_COLLECTION}' ({elapsed:.2f}s round trip).")
    saved = [s for s in recent if s.id != test_id]
    print(f"Saved conversations: {len(saved)}")
    for s in saved:
        print(f"  - {s.title or 'Untitled'}  ({time.strftime('%Y-%m-%d %H:%M', time.localtime(s.updated_at))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
