"""UI tests with Streamlit's AppTest. Data and models are faked; everything else (login, limits, rendering) is real."""
import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fakes import FakeFirestore
from workout_chat import access, config, conversations, data, graph as graph_module, jobs
from workout_chat.graph import TurnResult

APP = Path(__file__).resolve().parents[1] / "app.py"
KEVIN_PW, GUEST_PW = "kev-pw", "guest-pw"
PENDING_NOTE = "You can close this page; the answer will be here when you come back."


class Backend(list):
    """The (thread_id, question) calls the fake graph received, plus handles on the other fakes."""

    def __init__(self):
        super().__init__()
        self.histories = []          # the ``history`` passed with each call
        self.db = FakeFirestore()    # Kevin's "Firestore"
        self.hold = False            # True: background turns wait until release()
        self.held = []

    # Executor protocol: run turns inline (deterministic), or hold them to simulate a slow model.
    def submit(self, fn, *args):
        if self.hold:
            self.held.append((fn, args))
        else:
            fn(*args)

    def release(self):
        self.hold = False
        while self.held:
            fn, args = self.held.pop(0)
            fn(*args)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_SHEET_URL", "https://example.com/sheet")
    monkeypatch.setenv("APP_PASSWORD", KEVIN_PW)
    monkeypatch.setenv("GUEST_PASSWORD", GUEST_PW)
    monkeypatch.setenv("GCP_SERVICE_ACCOUNT_JSON", json.dumps({"type": "service_account"}))
    for name in ("OPENROUTER_API_KEY",):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_backend(monkeypatch, sample_df):
    import streamlit as st
    st.cache_resource.clear()
    calls = Backend()
    monkeypatch.setattr(data, "load_dataframe", lambda: sample_df)
    monkeypatch.setattr(graph_module, "build_graph", lambda df, **kw: object())
    monkeypatch.setattr(jobs, "_default_executor", lambda: calls)
    monkeypatch.setattr(conversations, "make_firestore_client", lambda info: calls.db)

    def fake_run_turn(graph, thread_id, question, on_step=None, history=None):
        calls.append((thread_id, question))
        calls.histories.append(list(history or []))
        if on_step:
            on_step("Reading the question...")
        return TurnResult(answer=f"Answer to: {question}", route="sql", code="SELECT 1", language="sql",
                          models_used={"coder": "g1"})

    monkeypatch.setattr(graph_module, "run_turn", fake_run_turn)
    yield calls
    st.cache_resource.clear()


def start():
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    return at


def login(at, who, password):
    at.radio[0].set_value(who).run()
    at.text_input[0].set_value(password)
    at.button[0].click().run()
    return at


def ask(at, text):
    at.chat_input[0].set_value(text).run()
    return at


def texts(at):
    return [m.value for m in at.markdown]


# ---- login ----------------------------------------------------------------------------------
def test_asks_who_you_are_before_anything_else(fake_backend):
    at = start()
    assert at.radio[0].label == "Who are you?" and list(at.radio[0].options) == ["A guest", "Kevin"]
    assert at.radio[0].value is None
    assert not at.chat_input and not at.text_input


def test_password_box_appears_after_choosing_a_role_and_hides_the_text(fake_backend):
    at = start()
    at.radio[0].set_value("Kevin").run()
    assert len(at.text_input) == 1 and at.text_input[0].proto.type == 1  # PASSWORD
    assert not at.chat_input


@pytest.mark.parametrize("who,password", [
    ("Kevin", "wrong"), ("Kevin", GUEST_PW), ("A guest", KEVIN_PW), ("A guest", ""),
])
def test_wrong_password_is_rejected(fake_backend, monkeypatch, who, password):
    monkeypatch.setattr("time.sleep", lambda s: None)
    at = login(start(), who, password)
    assert at.error[0].value == "Incorrect password" and not at.chat_input


def test_kevin_gets_in_with_his_password_and_sees_the_greeting(fake_backend):
    at = login(start(), "Kevin", KEVIN_PW)
    assert at.chat_input and "Signed in as Kevin" in at.sidebar.caption[0].value
    assert any("Hi! I'm connected" in t for t in texts(at))
    assert not any("Guests can ask" in t for t in texts(at))


def test_workout_level_columns_are_in_a_collapsed_dropdown(fake_backend):
    at = login(start(), "Kevin", KEVIN_PW)
    assert len(at.expander) == 1 and "Workout-level" in at.expander[0].label
    assert at.expander[0].proto.expanded is False
    assert any("`Comment`" in m.value for m in at.expander[0].markdown)
    assert not any("`Comment`" in t and "`Reps`" in t for t in texts(at))      # not in the main table
    assert any("`Reps`" in t for t in texts(at)) and any("Try asking" in t for t in texts(at))


def test_guest_gets_in_with_the_guest_password(fake_backend):
    at = login(start(), "A guest", GUEST_PW)
    assert at.chat_input and "Guest: 0 of 5 questions used" in at.sidebar.caption[0].value
    assert any("Guests can ask 5 questions" in t for t in texts(at))


def test_missing_secret_shows_a_clear_error(fake_backend, monkeypatch):
    monkeypatch.delenv("GUEST_PASSWORD")
    at = start()
    assert "GUEST_PASSWORD" in at.error[0].value and not at.radio


# ---- asking questions -----------------------------------------------------------------------------
def test_guest_question_is_answered_and_counted(fake_backend):
    at = login(start(), "A guest", GUEST_PW)
    ask(at, "How many sets?")
    assert "Answer to: How many sets?" in texts(at)
    assert "Guest: 1 of 5 questions used" in at.sidebar.caption[0].value
    assert len(fake_backend) == 1
    assert not any(c.value.startswith("route:") for c in at.caption)      # debug info is Kevin-only


def test_guest_is_stopped_after_five_questions(fake_backend):
    at = login(start(), "A guest", GUEST_PW)
    for i in range(5):
        ask(at, f"q{i}")
    ask(at, "one more?")
    assert texts(at)[-1] == access.LIMIT_MESSAGE
    assert len(fake_backend) == 5                                        # the 6th never reached the models
    assert "Guest: 5 of 5 questions used" in at.sidebar.caption[0].value
    ask(at, "and another?")
    assert len(fake_backend) == 5


def test_new_chat_does_not_reset_the_guest_limit(fake_backend):
    at = login(start(), "A guest", GUEST_PW)
    for i in range(5):
        ask(at, f"q{i}")
    next(b for b in at.sidebar.button if b.label == "New chat").click().run()
    ask(at, "fresh start?")
    assert texts(at)[-1] == access.LIMIT_MESSAGE and len(fake_backend) == 5


def test_kevin_is_never_limited_and_sees_debug_info(fake_backend):
    at = login(start(), "Kevin", KEVIN_PW)
    for i in range(8):
        ask(at, f"q{i}")
    assert len(fake_backend) == 8 and access.LIMIT_MESSAGE not in texts(at)
    assert any(c.value.startswith("route: sql") for c in at.caption)
    assert any(b.label == "Refresh data" for b in at.sidebar.button)


def test_guests_do_not_get_the_refresh_button(fake_backend):
    at = login(start(), "A guest", GUEST_PW)
    assert not any(b.label == "Refresh data" for b in at.sidebar.button)


def test_follow_ups_reuse_the_same_thread_until_new_chat(fake_backend):
    at = login(start(), "Kevin", KEVIN_PW)
    ask(at, "first")
    ask(at, "second")
    assert fake_backend[0][0] == fake_backend[1][0]
    next(b for b in at.sidebar.button if b.label == "New chat").click().run()
    ask(at, "third")
    assert fake_backend[2][0] != fake_backend[0][0]


def test_the_code_is_shown_in_an_expander(fake_backend):
    at = login(start(), "Kevin", KEVIN_PW)
    ask(at, "q")
    assert any("SELECT 1" in c.value for c in at.code)


def test_data_load_failure_is_shown_not_crashed(fake_backend, monkeypatch):
    def boom():
        raise data.DataLoadError("Sheet not shared with the service account")

    monkeypatch.setattr(data, "load_dataframe", boom)
    at = login(start(), "Kevin", KEVIN_PW)
    assert "Sheet not shared" in at.error[0].value and not at.exception


# ---- surviving reloads (phone sleep) and Kevin's history ---------------------------------------
def reopen(conv_id):
    """A brand-new browser session (what a phone reload gives you) opened on ``?c=conv_id``."""
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.query_params["c"] = conv_id
    at.run()
    return at


def conversation_buttons(at):
    return [b for b in at.sidebar.button if b.label not in ("New chat", "Refresh data")]


def test_the_conversation_id_is_kept_in_the_url(fake_backend):
    at = login(start(), "Kevin", KEVIN_PW)
    ask(at, "How many sets?")
    assert at.query_params["c"] == fake_backend[0][0]


@pytest.mark.parametrize("who,password", [("Kevin", KEVIN_PW), ("A guest", GUEST_PW)])
def test_a_reloaded_page_gets_the_conversation_back_after_logging_in(fake_backend, who, password):
    at = login(start(), who, password)
    ask(at, "How many sets?")
    conv_id = fake_backend[0][0]
    again = login(reopen(conv_id), who, password)
    assert "How many sets?" in texts(again) and "Answer to: How many sets?" in texts(again)
    ask(again, "and squats?")
    assert fake_backend[1][0] == conv_id                                  # same conversation continues


def test_reloading_while_the_answer_is_running_shows_progress_then_the_answer(fake_backend):
    fake_backend.hold = True
    at = login(start(), "Kevin", KEVIN_PW)
    ask(at, "How many sets?")
    assert any(c.value == PENDING_NOTE for c in at.caption)
    assert at.chat_input[0].proto.disabled                                # one question at a time
    conv_id = at.query_params["c"]

    again = login(reopen(conv_id), "Kevin", KEVIN_PW)                     # phone woke up and reloaded
    assert "How many sets?" in texts(again) and any(c.value == PENDING_NOTE for c in again.caption)

    fake_backend.release()                                               # the model finally answers
    again.run()
    assert "Answer to: How many sets?" in texts(again)
    assert not any(c.value == PENDING_NOTE for c in again.caption) and not again.chat_input[0].proto.disabled


def test_the_original_page_also_gets_the_answer_when_it_finishes(fake_backend):
    fake_backend.hold = True
    at = login(start(), "A guest", GUEST_PW)
    ask(at, "How many sets?")
    fake_backend.release()
    at.run()
    assert texts(at).count("Answer to: How many sets?") == 1


def test_a_guest_cannot_open_kevins_conversation(fake_backend):
    at = login(start(), "Kevin", KEVIN_PW)
    ask(at, "Kevin's private question")
    kevin_id = fake_backend[0][0]
    guest = login(reopen(kevin_id), "A guest", GUEST_PW)
    assert not any("private" in t for t in texts(guest))
    assert guest.query_params["c"] != kevin_id
    ask(guest, "guest question")
    assert fake_backend[1][0] != kevin_id                                 # never shares Kevin's LangGraph thread


def test_follow_ups_send_the_stored_history_along(fake_backend):
    at = login(start(), "Kevin", KEVIN_PW)
    ask(at, "first")
    ask(at, "second")
    assert fake_backend.histories == [[], [("user", "first"), ("assistant", "Answer to: first")]]


def test_kevins_sidebar_lists_his_ten_most_recent_conversations(fake_backend):
    at = login(start(), "Kevin", KEVIN_PW)
    for i in range(12):
        ask(at, f"question {i}")
        next(b for b in at.sidebar.button if b.label == "New chat").click().run()
    labels = [b.label for b in conversation_buttons(at)]
    assert len(labels) == 10 and labels[0].startswith("question 11") and labels[-1].startswith("question 2")
    assert len(fake_backend.db.data[config.FIRESTORE_COLLECTION]) == 10   # older ones are pruned from storage

    next(b for b in conversation_buttons(at) if b.label.startswith("question 5")).click().run()
    assert "Answer to: question 5" in texts(at) and "Answer to: question 11" not in texts(at)
    ask(at, "follow-up")
    assert conversation_buttons(at)[0].label.startswith("question 5")    # most recent moves to the top


def test_kevins_history_survives_an_app_restart(fake_backend):
    import streamlit as st
    at = login(start(), "Kevin", KEVIN_PW)
    ask(at, "before the restart")
    st.cache_resource.clear()                                             # memory, jobs and checkpointer are gone
    again = login(start(), "Kevin", KEVIN_PW)
    assert conversation_buttons(again)[0].label.startswith("before the restart")


def test_guests_get_no_history_sidebar_and_never_touch_firestore(fake_backend):
    at = login(start(), "A guest", GUEST_PW)
    ask(at, "q")
    assert conversation_buttons(at) == [] and fake_backend.db.calls == []


def test_firestore_outage_keeps_the_chat_working_with_a_note(fake_backend, monkeypatch):
    fake_backend.db.fail = RuntimeError("403 Cloud Firestore API has not been used in project")
    at = login(start(), "Kevin", KEVIN_PW)
    assert any("History isn't being saved" in c.value for c in at.sidebar.caption)
    ask(at, "still works?")
    assert "Answer to: still works?" in texts(at) and not at.exception
