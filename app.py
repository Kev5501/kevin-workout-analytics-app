"""Streamlit entry point: login -> greeting -> chat over Kevin's workout log.

Answers survive a phone going to sleep: each turn runs in a background thread (jobs.py) that saves its reply
to a conversation store (conversations.py), and the current conversation id lives in the URL (?c=...). A page
that reloads finds the conversation again after login and shows the reply, or progress while it is running.
Kevin's conversations are kept in Firestore (newest KEVIN_HISTORY_LIMIT, listed in the sidebar); guests' stay
in process memory.
"""
import copy
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import plotly.io as pio  # noqa: E402
import streamlit as st  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from workout_chat import access, config, conversations, data, graph as graph_module, jobs  # noqa: E402
from workout_chat.conversations import Conversation  # noqa: E402
from workout_chat.schema import build_greeting, workout_count  # noqa: E402

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Kevin's Workout Chat", page_icon="🏋️", layout="centered")

POLL_SECONDS = 2


# ---- shared across all sessions of this app process --------------------------------------
@st.cache_resource
def get_saver() -> InMemorySaver:
    return InMemorySaver()


@st.cache_resource
def get_daily_counter() -> access.DailyCounter:
    return access.DailyCounter()


@st.cache_resource
def get_runner() -> jobs.JobRunner:
    return jobs.JobRunner()


@st.cache_resource
def get_memory_store() -> conversations.MemoryStore:
    return conversations.MemoryStore()


@st.cache_resource(show_spinner=False)
def get_firestore_store() -> conversations.FirestoreStore:
    """Kevin's persistent history. Raises if Firestore is unusable (failures are not cached)."""
    client = conversations.make_firestore_client(data.service_account_info())
    store = conversations.FirestoreStore(client, config.FIRESTORE_COLLECTION, access.KEVIN)
    store.list_recent(access.KEVIN, 1)  # fail now (API off, role missing) rather than mid-conversation
    return store


@st.cache_resource(ttl=config.DATA_CACHE_TTL_SECONDS, show_spinner="Loading Kevin's workout log...")
def get_app():
    df = data.load_dataframe()
    return df, graph_module.build_graph(df, checkpointer=get_saver())


def store_for(role: str):
    """Firestore for Kevin (falls back to memory for the rest of the session if it fails); memory for guests."""
    if role == access.KEVIN and "history_error" not in st.session_state:
        try:
            return get_firestore_store()
        except Exception as exc:  # noqa: BLE001 - history is a nice-to-have; the chat must keep working
            logger.exception("Firestore unavailable; Kevin's history falls back to memory")
            st.session_state.history_error = f"{type(exc).__name__}: {exc}"
    return get_memory_store()


# ---- login -----------------------------------------------------------------------------------
def login_screen() -> None:
    st.title("💪 Kevin's GainsGPT 💪")
    label = st.radio("Who are you?", list(access.ROLE_CHOICES), index=None)
    if label is None:
        st.stop()
    role = access.ROLE_CHOICES[label]
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if access.verify_password(
            role, password,
            kevin_password=config.get_secret("APP_PASSWORD"),
            guest_password=config.get_secret("GUEST_PASSWORD"),
        ):
            st.session_state.role = role
            st.rerun()
        time.sleep(1)  # slow down guessing
        st.error("Incorrect password")
    st.stop()


# ---- conversations ---------------------------------------------------------------------------
def use_conversation(conv: Conversation) -> Conversation:
    st.session_state.conv = conv
    if st.query_params.get("c") != conv.id:
        st.query_params["c"] = conv.id
    return conv


def open_conversation(store, role: str) -> Conversation:
    """The conversation named in the URL if this role may see it, otherwise a fresh one."""
    conv_id = st.query_params.get("c")
    conv = None
    if conv_id:
        try:
            conv = store.get(conv_id)
        except Exception:  # noqa: BLE001
            logger.exception("Could not load conversation %s", conv_id)
            st.warning("Couldn't load that conversation, so here's a new one.")
        if conv is not None and conv.owner != role:
            conv = None  # e.g. a guest holding a link to one of Kevin's conversations
    return use_conversation(conv or Conversation(id=uuid.uuid4().hex, owner=role))


def merge_finished_reply(conv: Conversation, job: jobs.TurnJob | None) -> bool:
    """Add a finished job's reply if the conversation is still waiting for it. True if something changed."""
    if job is None or job.running or job.message is None or not conv.messages:
        return False
    last = conv.messages[-1]
    if last.get("role") == "user" and last.get("content") == job.question:
        conv.add(job.message)
        return True
    return False


def save_quietly(store, conv: Conversation) -> None:
    try:
        store.save(conv)
    except Exception:  # noqa: BLE001 - the background job retries the save with the reply
        logger.exception("Saving conversation %s failed", conv.id)


def new_chat(role: str) -> None:
    use_conversation(Conversation(id=uuid.uuid4().hex, owner=role))


def greeting_message(df, role: str) -> dict:
    loaded_at = df.attrs.get("loaded_at", "")
    cached = st.session_state.get("greeting")
    if cached is None or cached[0] != loaded_at:
        greeting = build_greeting(df)
        after = greeting.outro
        if role == access.GUEST:
            after += f"\n\n*Guests can ask {config.GUEST_QUESTION_LIMIT} questions.*"
        msg = {
            "role": "assistant", "content": greeting.intro,
            "details_title": greeting.details_title, "details": greeting.details, "content_after": after,
        }
        cached = st.session_state.greeting = (loaded_at, msg)
    return cached[1]


def show(msg: dict, is_kevin: bool) -> None:
    st.markdown(msg["content"])
    if msg.get("details"):
        with st.expander(msg["details_title"]):
            st.markdown(msg["details"])
    if msg.get("content_after"):
        st.markdown(msg["content_after"])
    if msg.get("figure_json"):
        st.plotly_chart(pio.from_json(msg["figure_json"]), use_container_width=True)
    elif msg.get("figure_dropped"):
        st.caption("(Chart not saved: too large for the history store.)")
    if msg.get("code"):
        with st.expander("Show the code that was run"):
            st.code(msg["code"], language=msg.get("language") or "python")
    if is_kevin and msg.get("debug"):
        st.caption(msg["debug"])


def recent_conversations(store, current: Conversation) -> None:
    st.divider()
    st.markdown("**Recent conversations**")
    if not store.persistent:
        st.caption("History isn't being saved right now (Firestore unavailable).")
    if st.session_state.get("recent") is None:
        try:
            st.session_state.recent = store.list_recent(access.KEVIN, config.KEVIN_HISTORY_LIMIT)
        except Exception:  # noqa: BLE001
            logger.exception("Listing conversations failed")
            st.session_state.recent = []
    if not st.session_state.recent:
        st.caption("No saved conversations yet.")
    for item in st.session_state.recent:
        label = f"{item.title or 'Untitled'} · {conversations.relative_time(item.updated_at)}"
        kind = "primary" if item.id == current.id else "secondary"
        if st.button(label, key=f"conv-{item.id}", type=kind, width="stretch"):
            st.query_params["c"] = item.id
            st.session_state.pop("conv", None)
            st.rerun()


@st.fragment(run_every=POLL_SECONDS)
def pending_reply(conv_id: str) -> None:
    """Progress of the running turn; reruns the whole page once the reply is in."""
    job = get_runner().get(conv_id)
    if job is None or not job.running:
        st.rerun()
    with st.chat_message("assistant"):
        st.status(f"{job.label} ({int(time.time() - job.started_at)} s)", state="running")
        st.caption("You can close this page; the answer will be here when you come back.")


def ask(question: str, conv: Conversation, store, role: str, graph) -> None:
    conv.add({"role": "user", "content": question})
    verdict = access.check_question(
        role, st.session_state.questions_asked, get_daily_counter(),
        session_limit=config.GUEST_QUESTION_LIMIT, daily_limit=config.GUEST_DAILY_LIMIT,
    )
    if not verdict.allowed:
        conv.add({"role": "assistant", "content": access.LIMIT_MESSAGE})
        save_quietly(store, conv)
        st.rerun()

    st.session_state.questions_asked = verdict.used
    save_quietly(store, conv)
    st.session_state.recent = None

    turns = int(config.get_model_config().settings.get("history_turns", 6))
    history = conversations.history_pairs(conv.messages[:-1], turns * 2)
    snapshot = copy.deepcopy(conv)
    keep = config.KEVIN_HISTORY_LIMIT if role == access.KEVIN else None

    def work(on_step):
        result = graph_module.run_turn(graph, conv.id, question, on_step=on_step, history=history)
        return conversations.assistant_message(result, question)

    get_runner().submit(conv.id, question, work, lambda msg: conversations.record_reply(store, snapshot, msg, keep))
    st.rerun()


# ---- main -------------------------------------------------------------------------------------
missing = config.missing_secrets()
if missing:
    st.error(f"Missing secrets: {', '.join(missing)}. Add them in .streamlit/secrets.toml (or the Streamlit Cloud secrets box).")
    st.stop()

if "role" not in st.session_state:
    login_screen()

role = st.session_state.role
is_kevin = role == access.KEVIN

try:
    df, graph = get_app()
except data.DataLoadError as exc:
    st.error(f"Could not load the workout log: {exc}")
    st.stop()

if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0  # not reset by "New chat", so guests cannot reset their limit

store = store_for(role)
conv = st.session_state.get("conv") or open_conversation(store, role)
runner = get_runner()
job = runner.get(conv.id)
if merge_finished_reply(conv, job):
    st.session_state.recent = None
busy = bool(job and job.running)

with st.sidebar:
    if is_kevin:
        st.caption("Signed in as Kevin (no question limit)")
    else:
        st.caption(f"Guest: {st.session_state.questions_asked} of {config.GUEST_QUESTION_LIMIT} questions used")
    st.caption(f"{len(df):,} sets · {workout_count(df):,} workouts · loaded {df.attrs.get('loaded_at', '')[:16].replace('T', ' ')} UTC")
    if st.button("New chat"):
        new_chat(role)
        st.rerun()
    if is_kevin and st.button("Refresh data"):
        get_app.clear()
        st.rerun()
    if is_kevin:
        recent_conversations(store, conv)

st.title("🏋️ Kevin's Workout Chat")
with st.chat_message("assistant"):
    show(greeting_message(df, role), is_kevin)
for msg in conv.messages:
    with st.chat_message(msg["role"]):
        show(msg, is_kevin)

if busy:
    pending_reply(conv.id)

question = st.chat_input("Ask about Kevin's workouts...", disabled=busy)
if question and not busy:
    ask(question, conv, store, role, graph)
