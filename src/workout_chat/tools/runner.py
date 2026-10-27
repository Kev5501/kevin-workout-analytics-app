"""Generate code with the coder model, run it in the sandbox, and retry with the error on failure."""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

import pandas as pd

from .. import llm, prompts
from ..config import ConfigError
from ..llm import AllModelsFailed, text_of
from ..sandbox import SandboxResult
from .base import ToolContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeLanguage:
    name: str                                   # "sql" | "python"
    fence: tuple[str, ...]                      # accepted code-fence labels
    node: str                                   # models.toml node (falls back to "coder")
    system_prompt: Callable[[ToolContext, bool], str]
    run: Callable[[str, pd.DataFrame], SandboxResult]


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    code: str = ""
    text: str = ""
    error: str | None = None
    figure_json: str | None = None
    attempts: int = 0
    model_key: str | None = None
    llm_failed: bool = False                    # the AI models were unavailable (not a code problem)
    language: str = ""
    history: list[str] = field(default_factory=list)  # one line per failed attempt (for debugging)


_FENCE = re.compile(r"```(?P<label>[A-Za-z0-9_+-]*)[ \t]*\r?\n(?P<body>.*?)```", re.S)
_OPEN_FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n(?P<body>.*)$", re.S)


def extract_code(text: str, labels: tuple[str, ...]) -> str:
    """The code from a model reply: the first fenced block (preferring the right language), else the raw text."""
    blocks = list(_FENCE.finditer(text))
    if blocks:
        preferred = [b for b in blocks if b.group("label").lower() in labels]
        return (preferred or blocks)[0].group("body").strip()
    unterminated = _OPEN_FENCE.search(text)
    if unterminated:
        return unterminated.group("body").strip()
    return text.strip()


def generate_and_run(ctx: ToolContext, lang: CodeLanguage, question: str, needs_chart: bool = False) -> ToolOutcome:
    max_attempts = int(ctx.settings.get("max_code_retries", 2)) + 1
    system = lang.system_prompt(ctx, needs_chart)
    previous_code, previous_error = "", ""
    failures: list[str] = []
    model_key = None

    for attempt in range(1, max_attempts + 1):
        if previous_error:
            human = prompts.CODE_RETRY.format(
                question=question, fence=lang.fence[0], code=previous_code, error=previous_error
            )
        else:
            human = prompts.CODE_FIRST.format(question=question)
        try:
            reply = llm.invoke_chat(lang.node, [("system", system), ("human", human)], default_node="coder", config=ctx.config)
        except AllModelsFailed as exc:
            return ToolOutcome(False, error=exc.user_message, attempts=attempt - 1, llm_failed=True,
                               language=lang.name, history=failures)
        except ConfigError as exc:
            logger.error("Model configuration problem: %s", exc)
            return ToolOutcome(False, error="The AI models are not configured correctly.", attempts=attempt - 1,
                               llm_failed=True, language=lang.name, history=failures)

        model_key = reply.model_key
        code = extract_code(text_of(reply.value), lang.fence)
        result = lang.run(code, ctx.df) if code else SandboxResult(False, error="The reply contained no code.")
        if result.ok:
            return ToolOutcome(True, code=code, text=result.text, figure_json=result.figure_json, attempts=attempt,
                               model_key=model_key, language=lang.name, history=failures)
        previous_code, previous_error = code, result.error or "Unknown error"
        failures.append(f"attempt {attempt}: {previous_error}")
        logger.info("%s attempt %d failed: %s", lang.name, attempt, previous_error)

    return ToolOutcome(False, code=previous_code, error=previous_error, attempts=max_attempts,
                       model_key=model_key, language=lang.name, history=failures)


def tool_content(outcome: ToolOutcome) -> str:
    """The text the ToolMessage carries."""
    if outcome.ok:
        chart = "yes" if outcome.figure_json else "no"
        return f"STATUS: success ({outcome.language}, attempt {outcome.attempts})\nCHART: {chart}\nRESULT:\n{outcome.text}"
    return f"STATUS: failed after {outcome.attempts} attempt(s)\nLAST ERROR: {outcome.error}"


def tool_artifact(outcome: ToolOutcome) -> dict:
    return asdict(outcome)
