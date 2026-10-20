"""Chat-model factory and calling helpers.

Models come from models.toml (see config.py). A node's chain is tried in order:
if one model fails (rate limit, outage, bad key, retired model), the next one is
used, and every failure is recorded so the UI and logs can say what happened.

Decision-kind models (typed questions, e.g. mercury-decide) live in decisions.py.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .config import ConfigError, ModelConfig, ModelSpec, get_model_config, get_secret

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
APP_TITLE = "Workout Analytics Chat"
REQUEST_TIMEOUT_SECONDS = 60

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
ERROR_KINDS = ("rate_limit", "auth", "unavailable", "network", "bad_response", "other")


class ModelCallError(RuntimeError):
    """One model failed. ``kind`` is one of ERROR_KINDS."""

    def __init__(self, model_key: str, kind: str, detail: str):
        super().__init__(f"{model_key}: {kind}: {detail}")
        self.model_key, self.kind, self.detail = model_key, kind, detail


class AllModelsFailed(RuntimeError):
    """Every model in a node's chain failed. ``user_message`` is safe to show in the chat."""

    def __init__(self, node: str, failures: list[ModelCallError]):
        self.node, self.failures = node, list(failures)
        summary = "; ".join(str(f) for f in self.failures) or "no models were tried"
        super().__init__(f"All models failed for node '{node}': {summary}")

    @property
    def user_message(self) -> str:
        kinds = {f.kind for f in self.failures}
        if "rate_limit" in kinds:
            return ("Sorry, the free AI models are at their rate or daily limit right now. "
                    "Please try again in a few minutes.")
        if kinds == {"auth"}:
            return "Sorry, an AI model rejected its API key or account. The app owner needs to check the secrets."
        return "Sorry, I couldn't reach the AI models just now. Please try again in a moment."


def _status_code(exc: BaseException) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "code"):
            value = getattr(obj, attr, None)
            if isinstance(value, int) and 100 <= value < 600:
                return value
    return None


def classify_error(exc: BaseException) -> str:
    """Map any provider exception to one of ERROR_KINDS (providers raise many different types)."""
    status = _status_code(exc)
    text = f"{type(exc).__name__} {exc}".lower()
    if status == 429 or any(s in text for s in ("rate limit", "ratelimit", "resource_exhausted", "quota", "too many requests")):
        return "rate_limit"
    if status in (401, 402, 403) or any(s in text for s in ("api key", "unauthorized", "permission denied", "insufficient credit")):
        return "auth"
    if status in (408, 504) or isinstance(exc, (TimeoutError, ConnectionError)) or any(
        s in text for s in ("timeout", "timed out", "connection")
    ):
        return "network"
    if (status is not None and (status == 404 or status >= 500)) or any(
        s in text for s in ("not found", "no endpoints", "model_unavailable", "overloaded", "unavailable")
    ):
        return "unavailable"
    return "other"


def _short(exc: BaseException, limit: int = 300) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}"[:limit]


# --------------------------------------------------------------------------
# Building models
# --------------------------------------------------------------------------
_MODEL_CACHE: dict[tuple, Any] = {}


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def build_chat_model(spec: ModelSpec, settings: dict[str, Any]) -> Any:
    """Create (or reuse) the LangChain chat model for ``spec``."""
    if spec.kind != "chat":
        raise ConfigError(f"Model '{spec.key}' has kind '{spec.kind}', not 'chat'.")

    kwargs: dict[str, Any] = {
        "temperature": settings.get("temperature", 0.0),
        "max_retries": 1,  # fail fast: the fallback chain is the retry strategy
        "timeout": REQUEST_TIMEOUT_SECONDS,
    }
    kwargs.update(spec.params)  # models.toml extras win (e.g. a per-model temperature)

    cache_key = (spec.key, spec.provider, spec.model, json.dumps(kwargs, sort_keys=True, default=str))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if spec.provider == "google_genai":
        model = ChatGoogleGenerativeAI(model=spec.model, google_api_key=get_secret("GEMINI_API_KEY"), **kwargs)
    elif spec.provider == "openrouter":
        headers = {"X-OpenRouter-Title": APP_TITLE, **kwargs.pop("default_headers", {})}
        model = ChatOpenAI(
            model=spec.model,
            api_key=get_secret("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
            default_headers=headers,
            **kwargs,
        )
    else:
        raise ConfigError(f"Unsupported provider '{spec.provider}' for model '{spec.key}'.")

    _MODEL_CACHE[cache_key] = model
    return model


# --------------------------------------------------------------------------
# Calling models
# --------------------------------------------------------------------------
def text_of(message: Any) -> str:
    """Plain text of a chat response (some models return a list of content blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b if isinstance(b, str) else b.get("text", "") for b in content if isinstance(b, (str, dict))]
        return "".join(parts)
    return str(content)


def call_chat_model(spec: ModelSpec, messages: Any, *, schema: Any = None, settings: dict[str, Any] | None = None) -> Any:
    """Call ONE chat model. Returns an AIMessage, or a ``schema`` instance when ``schema`` is given.

    Any failure is raised as ModelCallError so callers can fall back to the next model.
    """
    settings = settings if settings is not None else get_model_config().settings
    try:
        model = build_chat_model(spec, settings)
        runnable = model.with_structured_output(schema) if schema is not None else model
        result = runnable.invoke(messages)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - providers raise many exception types
        raise ModelCallError(spec.key, classify_error(exc), _short(exc)) from exc
    if result is None:
        raise ModelCallError(spec.key, "bad_response", "model returned no structured output")
    return result


@dataclass(frozen=True)
class ChatResult:
    value: Any                              # AIMessage, or a schema instance
    model_key: str                          # which model answered
    failures: tuple[ModelCallError, ...] = ()  # models that failed first


def invoke_chat(
    node: str,
    messages: Any,
    *,
    schema: Any = None,
    default_node: str | None = None,
    config: ModelConfig | None = None,
) -> ChatResult:
    """Run ``messages`` through the chat models configured for ``node``, in priority order.

    ``default_node`` lets a tool use its own chain when models.toml defines one, and
    otherwise share another node's (e.g. generate_sql_code -> coder).
    """
    cfg = config or get_model_config()
    chain_node = node if node in cfg.nodes else default_node
    if chain_node is None or chain_node not in cfg.nodes:
        raise ConfigError(f"models.toml has no [nodes] entry for '{node}'" + (f" or '{default_node}'." if default_node else "."))

    specs = [s for s in cfg.chain(chain_node) if s.kind == "chat"]
    if not specs:
        raise ConfigError(f"Node '{chain_node}' has no usable chat models (decision-kind models can't write text).")

    failures: list[ModelCallError] = []
    for spec in specs:
        try:
            value = call_chat_model(spec, messages, schema=schema, settings=cfg.settings)
            return ChatResult(value, spec.key, tuple(failures))
        except ModelCallError as err:
            logger.warning("Model call failed (node=%s): %s", node, err)
            failures.append(err)
    raise AllModelsFailed(node, failures)
