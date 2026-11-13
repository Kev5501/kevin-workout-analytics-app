"""Conversation stores: memory and Firestore (with a fake client) behave the same through one interface."""
import json

import pytest

from fakes import FakeFirestore
from workout_chat import conversations as cv
from workout_chat.conversations import Conversation, FirestoreStore, MemoryStore
from workout_chat.graph import TurnResult


def conv(cid, owner="kevin", updated=0.0, n=1):
    c = Conversation(id=cid, owner=owner, created_at=updated, updated_at=updated)
    for i in range(n):
        c.messages += [{"role": "user", "content": f"{cid} q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    c.title = f"title {cid}"
    return c


@pytest.fixture(params=["memory", "firestore"])
def store(request):
    if request.param == "memory":
        return MemoryStore()
    return FirestoreStore(FakeFirestore(), "kevin_conversations", "kevin")


# ---- shared behaviour -------------------------------------------------------------------------
def test_save_and_get_round_trip(store):
    c = conv("a", updated=5)
    store.save(c)
    got = store.get("a")
    assert (got.id, got.owner, got.title, got.messages, got.updated_at) == ("a", "kevin", "title a", c.messages, 5)
    assert store.get("missing") is None


def test_saved_copy_is_independent_of_the_callers_object(store):
    c = conv("a")
    store.save(c)
    c.messages.append({"role": "user", "content": "later"})
    assert len(store.get("a").messages) == 2


def test_list_recent_is_newest_first_and_limited(store):
    for i in range(12):
        store.save(conv(f"c{i}", updated=float(i)))
    recent = store.list_recent("kevin", 10)
    assert [s.id for s in recent] == [f"c{i}" for i in range(11, 1, -1)]
    assert recent[0].title == "title c11" and recent[0].updated_at == 11.0


def test_prune_keeps_only_the_newest(store):
    for i in range(13):
        store.save(conv(f"c{i}", updated=float(i)))
    store.prune("kevin", 10)
    assert {s.id for s in store.list_recent("kevin", 50)} == {f"c{i}" for i in range(3, 13)}
    assert store.get("c0") is None and store.get("c12") is not None


def test_owners_are_isolated(store):
    store.save(conv("k1", updated=1))
    assert store.list_recent("guest", 10) == []
    store.prune("guest", 0)
    assert store.get("k1") is not None


def test_record_reply_appends_saves_and_prunes(store):
    for i in range(10):
        store.save(conv(f"old{i}", updated=float(i)))
    snapshot = Conversation(id="new", owner="kevin")
    snapshot.add({"role": "user", "content": "How many sets?"})
    store.save(snapshot)
    cv.record_reply(store, snapshot, {"role": "assistant", "content": "6"}, keep=10)
    assert [m["content"] for m in store.get("new").messages] == ["How many sets?", "6"]
    assert store.get("old0") is None and len(store.list_recent("kevin", 50)) == 10


def test_record_reply_uses_the_snapshot_if_the_first_save_was_lost(store):
    snapshot = Conversation(id="lost", owner="kevin")
    snapshot.add({"role": "user", "content": "q"})
    cv.record_reply(store, snapshot, {"role": "assistant", "content": "a"}, keep=None)
    assert [m["content"] for m in store.get("lost").messages] == ["q", "a"]


# ---- memory store -----------------------------------------------------------------------------
def test_memory_store_caps_its_size():
    store = MemoryStore(max_items=3)
    for i in range(5):
        store.save(conv(f"c{i}", owner="guest", updated=float(i)))
    assert [s.id for s in store.list_recent("guest", 10)] == ["c4", "c3", "c2"]


# ---- firestore store --------------------------------------------------------------------------
def test_firestore_listing_reads_only_title_and_time():
    db = FakeFirestore()
    store = FirestoreStore(db, "kc", "kevin")
    store.save(conv("a", updated=1, n=3))
    snap = db.collection("kc").order_by("updated_at").select(["title", "updated_at"]).stream()[0]
    assert set(snap.to_dict()) == {"title", "updated_at"}
    assert store.list_recent("kevin", 10)[0].title == "title a"


def test_firestore_delete_removes_one_conversation():
    store = FirestoreStore(FakeFirestore(), "kc", "kevin")
    store.save(conv("a"))
    store.save(conv("b"))
    store.delete("a")
    assert store.get("a") is None and store.get("b") is not None


def test_firestore_refuses_other_owners():
    store = FirestoreStore(FakeFirestore(), "kc", "kevin")
    with pytest.raises(ValueError):
        store.save(conv("g", owner="guest"))


def test_firestore_drops_old_charts_to_fit_the_document_limit():
    db = FakeFirestore()
    store = FirestoreStore(db, "kc", "kevin", max_bytes=5_000)
    c = Conversation(id="big", owner="kevin")
    for i in range(3):
        c.add({"role": "user", "content": f"plot {i}"})
        c.add({"role": "assistant", "content": "chart", "figure_json": "x" * 2_000})
    store.save(c)
    saved = db.data["kc"]["big"]["messages"]
    assert len(json.dumps(saved)) <= 5_000
    assert [bool(m.get("figure_json")) for m in saved if m["role"] == "assistant"] == [False, True, True]
    assert saved[1]["figure_dropped"] is True
    assert c.messages[1]["figure_json"]                              # the caller's copy is untouched


def test_firestore_errors_surface_to_the_caller():
    store = FirestoreStore(FakeFirestore(fail=RuntimeError("PermissionDenied")), "kc", "kevin")
    with pytest.raises(RuntimeError):
        store.list_recent("kevin", 1)


# ---- helpers ----------------------------------------------------------------------------------
def test_title_is_the_first_question_trimmed():
    c = Conversation(id="x", owner="kevin")
    c.add({"role": "user", "content": "  How   much did Kevin bench " + "x" * 100})
    c.add({"role": "user", "content": "second"})
    assert c.title.startswith("How much did Kevin bench") and len(c.title) == cv.TITLE_CHARS
    assert c.title.endswith("…")


def test_fit_messages_drops_oldest_messages_only_as_a_last_resort():
    msgs = [{"role": "user", "content": "x" * 400} for _ in range(5)]
    fitted = cv.fit_messages(msgs, 1_000)
    assert 1 <= len(fitted) < 5 and fitted[-1] == msgs[-1]
    assert cv.fit_messages(msgs, 10**6) == msgs


def test_history_pairs_keeps_the_last_turns():
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "code": "SELECT 1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    assert cv.history_pairs(msgs, 2) == [("user", "q2"), ("assistant", "a2")]
    assert cv.history_pairs(msgs, 0) == []


def test_assistant_message_builds_the_debug_line():
    result = TurnResult(answer="6 sets", route="sql", code="SELECT 1", language="sql",
                        standalone_question="How many bench sets?", models_used={"coder": "g1"}, error="slow")
    msg = cv.assistant_message(result, "and bench?")
    assert msg["content"] == "6 sets" and msg["code"] == "SELECT 1"
    assert msg["debug"] == "route: sql · models: coder=g1 · understood as: How many bench sets? · detail: slow"
    assert cv.assistant_message(TurnResult(answer="hi"), "q")["debug"] == ""


@pytest.mark.parametrize("ago,text", [(5, "just now"), (120, "2m ago"), (7_200, "2h ago"), (200_000, "2d ago")])
def test_relative_time(ago, text):
    assert cv.relative_time(1_000_000 - ago, now=1_000_000) == text
