"""End node: turn the tool's result into a clear answer."""
from __future__ import annotations

import logging
from typing import Callable

from langchain_core.messages import AIMessage

from .. import llm, prompts
from ..config import ConfigError
from ..llm import AllModelsFailed, text_of
from ..state import ChatState
from ..tools.base import ToolContext

logger = logging.getLogger(__name__)
MAX_CODE_IN_PROMPT = 1500


def _raw_result_answer(artifact: dict) -> str:
    return (
        "I ran the analysis but couldn't write it up because the AI models are busy right now. "
        "Here is the raw result:\n\n```\n" + artifact.get("text", "") + "\n```"
    )


def make_synthesize_node(ctx: ToolContext) -> Callable[[ChatState], dict]:
    def synthesize(state: ChatState) -> dict:
        message = state["tool_messages"][-1]
        artifact = message.artifact if isinstance(getattr(message, "artifact", None), dict) else {}
        update = {
            "code": artifact.get("code", ""),
            "language": artifact.get("language", ""),
            "figure_json": artifact.get("figure_json"),
            "coder_model": artifact.get("model_key") or "",
        }

        if not artifact.get("ok"):
            update["error"] = artifact.get("error") or str(message.content)
            update["figure_json"] = None
            # AI unavailable -> say so; code just didn't work -> ask for a rephrase. Neither needs another model call.
            answer = artifact["error"] if artifact.get("llm_failed") else prompts.TOOL_FAILED_ANSWER
            return {**update, "answer": answer, "messages": [AIMessage(content=answer)]}

        human = prompts.SYNTH_HUMAN.format(
            question=state["standalone_question"],
            language=artifact.get("language", ""),
            chart="yes" if artifact.get("figure_json") else "no",
            code=(artifact.get("code") or "")[:MAX_CODE_IN_PROMPT],
            result=artifact.get("text", ""),
        )
        try:
            reply = llm.invoke_chat("synthesizer", [("system", prompts.SYNTH_SYSTEM), ("human", human)], config=ctx.config)
            answer, update["synth_model"] = text_of(reply.value).strip(), reply.model_key
        except (AllModelsFailed, ConfigError) as exc:
            logger.warning("Synthesizer failed, returning the raw result: %s", exc)
            answer = ""
        if not answer:
            answer = _raw_result_answer(artifact)
        return {**update, "answer": answer, "messages": [AIMessage(content=answer)]}

    return synthesize
