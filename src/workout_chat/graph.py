"""The LangGraph workflow.

START -> start -> route --+--> off_topic ------------------------------> END
                          +--> clarify ---------------------------------> END
                          +--> (rewrite) -> prepare -> tools -> synthesize -> END
                          +--> END   (router unavailable: friendly error)

`tools` is a LangGraph ToolNode running whichever tool the router picked (see tools/__init__.py).
Memory: the checkpointer keeps the conversation per thread_id, so follow-ups work.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .config import ModelConfig
from .nodes.replies import make_clarify_node, make_off_topic_node
from .nodes.rewrite import make_rewrite_node
from .nodes.router import (
    ROUTE_CLARIFY, ROUTE_ERROR, ROUTE_OFF_TOPIC, make_prepare_node, make_route_node, next_after_route,
)
from .nodes.synthesize import make_synthesize_node
from .state import ChatState
from .tools import TOOLS, ToolContext, ToolSpec, build_tools

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500

# The graph reports a node when it FINISHES, so each label below describes the step that starts next.
INITIAL_LABEL = "Reading the question..."
LABEL_AFTER = {
    "start": "Working out what is being asked...",
    "rewrite": "Writing and running code...",
    "prepare": "Writing and running code...",
    "tools": "Writing the answer...",
}
LABEL_FOLLOWUP = "Resolving the follow-up..."
LABEL_CLARIFY = "Preparing a question..."


def _label_after(node: str, update: dict | None) -> str | None:
    """What is running now that ``node`` has finished (None when nothing slow follows)."""
    if node != "route":
        return LABEL_AFTER.get(node)
    update = update or {}
    route = update.get("route")
    if route in (ROUTE_ERROR, ROUTE_OFF_TOPIC):
        return None  # ends the turn, or a fixed reply with no model call
    if route == ROUTE_CLARIFY:
        return LABEL_CLARIFY
    return LABEL_FOLLOWUP if update.get("is_followup") else LABEL_AFTER["prepare"]


def _start_node(state: ChatState) -> dict:
    """Reset everything that belongs to a single turn (the checkpointer keeps old values otherwise)."""
    question = str(state["messages"][-1].content).strip()[:MAX_QUESTION_CHARS]
    return {
        "question": question, "standalone_question": question, "route": "", "route_confidence": 0.0,
        "needs_chart": False, "is_followup": False, "answer": "", "code": "", "language": "", "figure_json": None,
        "error": "", "router_model": "", "router_approximate": False, "rewriter_model": "", "coder_model": "",
        "synth_model": "", "tool_messages": [],
    }


def build_graph(
    df: pd.DataFrame,
    *,
    config: ModelConfig | None = None,
    checkpointer=None,
    specs: tuple[ToolSpec, ...] = TOOLS,
):
    ctx = ToolContext.from_df(df, config)
    tool_routes = {spec.route for spec in specs}

    graph = StateGraph(ChatState)
    graph.add_node("start", _start_node)
    graph.add_node("route", make_route_node(ctx, specs))
    graph.add_node("rewrite", make_rewrite_node(ctx))
    graph.add_node("prepare", make_prepare_node(specs))
    graph.add_node("tools", ToolNode(build_tools(ctx, specs), messages_key="tool_messages"))
    graph.add_node("synthesize", make_synthesize_node(ctx))
    graph.add_node("off_topic", make_off_topic_node(ctx))
    graph.add_node("clarify", make_clarify_node(ctx))

    graph.add_edge(START, "start")
    graph.add_edge("start", "route")
    graph.add_conditional_edges(
        "route", next_after_route(tool_routes),
        {"off_topic": "off_topic", "clarify": "clarify", "rewrite": "rewrite", "prepare": "prepare", "end": END},
    )
    graph.add_edge("rewrite", "prepare")
    graph.add_edge("prepare", "tools")
    graph.add_edge("tools", "synthesize")
    for terminal in ("synthesize", "off_topic", "clarify"):
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


# --------------------------------------------------------------------------
# Running one turn
# --------------------------------------------------------------------------
GENERIC_ERROR = "Sorry, something went wrong on my side. Please try again."


@dataclass
class TurnResult:
    answer: str
    route: str = ""
    code: str = ""
    language: str = ""
    figure_json: Optional[str] = None
    standalone_question: str = ""
    models_used: dict = field(default_factory=dict)   # node -> model key (shown to Kevin only)
    error: str = ""                                   # technical detail (shown to Kevin only)


def run_turn(graph, thread_id: str, question: str, on_step: Callable[[str], None] | None = None) -> TurnResult:
    """Run one question through the graph on conversation ``thread_id``. Never raises."""
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        if on_step:
            on_step(INITIAL_LABEL)
        for chunk in graph.stream({"messages": [HumanMessage(question)]}, cfg, stream_mode="updates"):
            for node, update in chunk.items():
                label = _label_after(node, update)
                if on_step and label:
                    on_step(label)
        values = graph.get_state(cfg).values
    except Exception as exc:  # noqa: BLE001 - the UI must always get an answer
        logger.exception("Turn failed")
        return TurnResult(answer=GENERIC_ERROR, route=ROUTE_ERROR, error=f"{type(exc).__name__}: {exc}")

    models = {
        "router": values.get("router_model", "") + (" (fallback)" if values.get("router_approximate") else ""),
        "rewriter": values.get("rewriter_model", ""),
        "coder": values.get("coder_model", ""),
        "synthesizer": values.get("synth_model", ""),
    }
    return TurnResult(
        answer=values.get("answer") or GENERIC_ERROR,
        route=values.get("route", ""),
        code=values.get("code", ""),
        language=values.get("language", ""),
        figure_json=values.get("figure_json"),
        standalone_question=values.get("standalone_question", ""),
        models_used={k: v for k, v in models.items() if v},
        error=values.get("error", ""),
    )
