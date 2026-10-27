"""Graph state and small helpers for reading the conversation."""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


def replace(old, new):
    """Reducer: the newest value wins (used for per-turn scratch channels)."""
    return new


class ChatState(TypedDict, total=False):
    # Conversation memory: only the visitor's questions and the final answers (persisted per thread).
    messages: Annotated[list[AnyMessage], add_messages]
    # The tool-call round trip of the CURRENT turn (replaced every turn, never accumulates).
    tool_messages: Annotated[list[AnyMessage], replace]

    question: str               # the visitor's latest message
    standalone_question: str    # same question, with follow-up context filled in
    route: str                  # "sql" | "python" | ... | "clarify" | "off_topic" | "error"
    route_confidence: float
    needs_chart: bool
    is_followup: bool

    answer: str
    code: str
    language: str
    figure_json: Optional[str]
    error: str

    router_model: str
    router_approximate: bool
    rewriter_model: str
    coder_model: str
    synth_model: str


def previous_user_questions(state: ChatState, n: int = 2) -> list[str]:
    """The ``n`` user questions before the current one (the current one is the last human message)."""
    humans = [str(m.content) for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    return humans[:-1][-n:]


def history_text(state: ChatState, max_messages: int, answer_chars: int = 300) -> str:
    """Recent conversation (excluding the current question) as plain text for prompts."""
    msgs = list(state.get("messages", []))[:-1][-max_messages:]
    lines = []
    for m in msgs:
        who = "Visitor" if isinstance(m, HumanMessage) else "Assistant"
        text = str(m.content).strip()
        lines.append(f"{who}: {text if who == 'Visitor' else text[:answer_chars]}")
    return "\n".join(lines)
