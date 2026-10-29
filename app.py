"""Streamlit entry point: login -> greeting -> chat over Kevin's workout log."""
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import plotly.io as pio  # noqa: E402
import streamlit as st  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from workout_chat import access, config, data, graph as graph_module  # noqa: E402
from workout_chat.schema import build_greeting  # noqa: E402

st.set_page_config(page_title="Kevin's Workout Chat", page_icon="🏋️", layout="centered")


# ---- shared across all sessions of this app process --------------------------------------
@st.cache_resource
def get_saver() -> InMemorySaver:
    return InMemorySaver()


@st.cache_resource
def get_daily_counter() -> access.DailyCounter:
    return access.DailyCounter()


@st.cache_resource(ttl=config.DATA_CACHE_TTL_SECONDS, show_spinner="Loading Kevin's workout log...")
def get_app():
    df = data.load_dataframe()
    return df, graph_module.build_graph(df, checkpointer=get_saver())


# ---- login -----------------------------------------------------------------------------------
def login_screen() -> None:
    st.title("🏋️ Kevin's Workout Chat")
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


def new_chat(df) -> None:
    st.session_state.thread_id = uuid.uuid4().hex
    greeting = build_greeting(df)
    after = greeting.outro
    if st.session_state.role == access.GUEST:
        after += f"\n\n*Guests can ask {config.GUEST_QUESTION_LIMIT} questions.*"
    st.session_state.messages = [{
        "role": "assistant", "content": greeting.intro,
        "details_title": greeting.details_title, "details": greeting.details, "content_after": after,
    }]


def show(msg: dict, is_kevin: bool) -> None:
    st.markdown(msg["content"])
    if msg.get("details"):
        with st.expander(msg["details_title"]):
            st.markdown(msg["details"])
    if msg.get("content_after"):
        st.markdown(msg["content_after"])
    if msg.get("figure_json"):
        st.plotly_chart(pio.from_json(msg["figure_json"]), use_container_width=True)
    if msg.get("code"):
        with st.expander("Show the code that was run"):
            st.code(msg["code"], language=msg.get("language") or "python")
    if is_kevin and msg.get("debug"):
        st.caption(msg["debug"])


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

if "messages" not in st.session_state:
    st.session_state.questions_asked = 0  # not reset by "New chat", so guests cannot reset their limit
    new_chat(df)

with st.sidebar:
    if is_kevin:
        st.caption("Signed in as Kevin (no question limit)")
    else:
        st.caption(f"Guest: {st.session_state.questions_asked} of {config.GUEST_QUESTION_LIMIT} questions used")
    st.caption(f"{len(df):,} sets · {df['Date'].nunique():,} workouts · loaded {df.attrs.get('loaded_at', '')[:16].replace('T', ' ')} UTC")
    if st.button("New chat"):
        new_chat(df)
        st.rerun()
    if is_kevin and st.button("Refresh data"):
        get_app.clear()
        st.rerun()

st.title("🏋️ Kevin's Workout Chat")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        show(msg, is_kevin)

question = st.chat_input("Ask about Kevin's workouts...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    verdict = access.check_question(
        role, st.session_state.questions_asked, get_daily_counter(),
        session_limit=config.GUEST_QUESTION_LIMIT, daily_limit=config.GUEST_DAILY_LIMIT,
    )
    if not verdict.allowed:
        st.session_state.messages.append({"role": "assistant", "content": access.LIMIT_MESSAGE})
        st.rerun()

    st.session_state.questions_asked = verdict.used
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.status("Working...", expanded=False) as status:
            result = graph_module.run_turn(
                graph, st.session_state.thread_id, question, on_step=lambda label: status.update(label=label)
            )
            status.update(label="Done", state="complete")
    debug = ""
    if result.route:
        debug = f"route: {result.route}"
        if result.models_used:
            debug += " · models: " + ", ".join(f"{k}={v}" for k, v in result.models_used.items())
        if result.standalone_question and result.standalone_question != question:
            debug += f" · understood as: {result.standalone_question}"
        if result.error:
            debug += f" · detail: {result.error}"
    st.session_state.messages.append({
        "role": "assistant", "content": result.answer, "figure_json": result.figure_json,
        "code": result.code, "language": result.language, "debug": debug,
    })
    st.rerun()
