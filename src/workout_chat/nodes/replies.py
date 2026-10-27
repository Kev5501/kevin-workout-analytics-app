"""Replies that need no tool: off-topic redirect and clarifying question."""
from __future__ import annotations

import logging
from typing import Callable

from langchain_core.messages import AIMessage

from .. import llm, prompts
from ..config import ConfigError
from ..llm import AllModelsFailed, text_of
from ..state import ChatState, history_text
from ..tools.base import ToolContext

logger = logging.getLogger(__name__)


def make_off_topic_node(ctx: ToolContext) -> Callable[[ChatState], dict]:
    example = f"How has Kevin's {ctx.top_exercise} progressed over the last 3 months?"

    def off_topic(state: ChatState) -> dict:
        answer = prompts.OFF_TOPIC_TEXT.format(example=example)  # fixed text: no model call, no tokens spent
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    return off_topic


def make_clarify_node(ctx: ToolContext) -> Callable[[ChatState], dict]:
    max_messages = int(ctx.settings.get("history_turns", 6)) * 2

    def clarify(state: ChatState) -> dict:
        system = prompts.CLARIFY_SYSTEM.format(domain=prompts.router_domain())
        human = prompts.CLARIFY_HUMAN.format(history=history_text(state, max_messages), question=state["question"])
        model = ""
        try:
            reply = llm.invoke_chat("rewriter", [("system", system), ("human", human)], config=ctx.config)
            answer, model = text_of(reply.value).strip() or prompts.CLARIFY_FALLBACK, reply.model_key
        except (AllModelsFailed, ConfigError) as exc:
            logger.warning("Clarify fell back to the fixed text: %s", exc)
            answer = prompts.CLARIFY_FALLBACK
        return {"answer": answer, "rewriter_model": model, "messages": [AIMessage(content=answer)]}

    return clarify
