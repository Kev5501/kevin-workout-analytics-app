"""Router: one typed-decision call decides what kind of question this is."""
from __future__ import annotations

import logging
import uuid
from typing import Callable, Sequence

from langchain_core.messages import AIMessage

from .. import decisions, prompts
from ..config import ConfigError
from ..decisions import Choice, Noul, Question
from ..llm import AllModelsFailed
from ..state import ChatState, previous_user_questions
from ..tools.base import ToolContext, ToolSpec

logger = logging.getLogger(__name__)

ROUTE_OFF_TOPIC = "off_topic"
ROUTE_CLARIFY = "clarify"
ROUTE_ERROR = "error"
YES = 0.5  # a Noul answer at or above this counts as "yes"


def build_questions(specs: Sequence[ToolSpec]) -> dict[str, Question]:
    """The router's questions. The route options come from the tool registry, so new tools appear automatically."""
    options = {spec.route: spec.route_description for spec in specs}
    options[ROUTE_CLARIFY] = prompts.CLARIFY_DESCRIPTION
    options[ROUTE_OFF_TOPIC] = prompts.OFF_TOPIC_DESCRIPTION
    return {
        "route": Choice(prompts.ROUTE_INSTRUCTIONS.format(domain=prompts.router_domain()), options),
        "needs_chart": Noul(prompts.CHART_INSTRUCTIONS, "A chart, plot or graph is requested", "No chart is requested"),
        "is_followup": Noul(prompts.FOLLOWUP_INSTRUCTIONS, "It refers back to earlier questions", "It stands on its own"),
    }


def make_route_node(ctx: ToolContext, specs: Sequence[ToolSpec]) -> Callable[[ChatState], dict]:
    questions = build_questions(specs)
    tool_routes = {spec.route for spec in specs}
    threshold = float(ctx.settings.get("router_confidence_threshold", 0.5))

    def route(state: ChatState) -> dict:
        previous = previous_user_questions(state, 2)
        payload = {"current_question": state["question"], "previous_questions": previous}
        try:
            result = decisions.decide(payload, questions, node="router", config=ctx.config)
        except (AllModelsFailed, ConfigError) as exc:
            message = exc.user_message if isinstance(exc, AllModelsFailed) else "The AI models are not configured correctly."
            logger.error("Router failed: %s", exc)
            return {"route": ROUTE_ERROR, "answer": message, "error": str(exc)}

        choice = result["route"]
        route_name = choice.value
        if choice.confidence < threshold:
            route_name = ROUTE_CLARIFY  # not sure what was asked: ask, don't guess
        needs_chart = result["needs_chart"].value >= YES
        if route_name == "sql" and needs_chart and "python" in tool_routes:
            route_name = "python"       # SQL cannot draw charts
        return {
            "route": route_name,
            "route_confidence": choice.confidence,
            "needs_chart": needs_chart,
            "is_followup": bool(previous) and result["is_followup"].value >= YES,
            "router_model": result.model_key,
            "router_approximate": result.approximate,
        }

    return route


def next_after_route(tool_routes: set[str]) -> Callable[[ChatState], str]:
    def choose(state: ChatState) -> str:
        route = state.get("route", ROUTE_ERROR)
        if route == ROUTE_ERROR:
            return "end"
        if route == ROUTE_OFF_TOPIC:
            return "off_topic"
        if route == ROUTE_CLARIFY or route not in tool_routes:
            return "clarify"
        return "rewrite" if state.get("is_followup") else "prepare"

    return choose


def make_prepare_node(specs: Sequence[ToolSpec]) -> Callable[[ChatState], dict]:
    by_route = {spec.route: spec for spec in specs}

    def prepare(state: ChatState) -> dict:
        spec = by_route[state["route"]]
        call = {
            "name": spec.name,
            "args": {"question": state["standalone_question"], "needs_chart": bool(state.get("needs_chart"))},
            "id": uuid.uuid4().hex,
        }
        return {"tool_messages": [AIMessage(content="", tool_calls=[call])]}

    return prepare
