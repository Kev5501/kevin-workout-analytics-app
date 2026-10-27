"""Turn a follow-up ("and for squats?") into a standalone question."""
from __future__ import annotations

import logging
from typing import Callable

from .. import llm, prompts
from ..config import ConfigError
from ..llm import AllModelsFailed, text_of
from ..state import ChatState, history_text
from ..tools.base import ToolContext

logger = logging.getLogger(__name__)


def make_rewrite_node(ctx: ToolContext) -> Callable[[ChatState], dict]:
    max_messages = int(ctx.settings.get("history_turns", 6)) * 2

    def rewrite(state: ChatState) -> dict:
        question = state["question"]
        human = prompts.REWRITE_HUMAN.format(history=history_text(state, max_messages), question=question)
        try:
            reply = llm.invoke_chat("rewriter", [("system", prompts.REWRITE_SYSTEM), ("human", human)], config=ctx.config)
        except (AllModelsFailed, ConfigError) as exc:
            logger.warning("Rewrite failed, using the original question: %s", exc)
            return {"standalone_question": question}
        standalone = text_of(reply.value).strip().strip('"') or question
        return {"standalone_question": standalone[:500], "rewriter_model": reply.model_key}

    return rewrite
