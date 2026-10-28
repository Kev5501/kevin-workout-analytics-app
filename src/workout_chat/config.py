"""Central configuration: secrets, constants and the models.toml registry.

This is the only module that reads ``st.secrets``. Everything else imports from
here, so the same code runs in Streamlit, in tests and in scripts (secrets fall
back to environment variables outside Streamlit).
"""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODELS_FILE = ROOT / "models.toml"

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SHEET_WORKSHEET = "Cleaned_Data"
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
DATA_CACHE_TTL_SECONDS = 600

# Guest limits. Per session: questions one guest session may ask. Daily: questions ALL guests together may ask
# per UTC day (a safety net, because a page refresh starts a new session). Set the daily limit to 0 to disable it.
GUEST_QUESTION_LIMIT = 5
GUEST_DAILY_LIMIT = 10

# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------
# APP_PASSWORD is Kevin's password (no question limit); GUEST_PASSWORD is for visitors (limited).
REQUIRED_SECRETS = ("GEMINI_API_KEY", "GOOGLE_SHEET_URL", "APP_PASSWORD", "GUEST_PASSWORD", "gcp_service_account")
OPTIONAL_SECRETS = ("OPENROUTER_API_KEY",)

# Which secret each provider needs. A model is skipped if its secret is missing.
PROVIDER_SECRETS = {
    "google_genai": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class ConfigError(ValueError):
    """Raised for invalid models.toml content or an unusable model chain."""


def _streamlit_secret(name: str) -> Any:
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets file, not running under Streamlit, etc.
        pass
    return None


def get_secret(name: str, default: Any = None) -> Any:
    """Read a secret from st.secrets, then the environment.

    ``gcp_service_account`` may also come from the ``GCP_SERVICE_ACCOUNT_JSON``
    environment variable (handy in Codespaces). Nested secret tables are
    returned as plain dicts.
    """
    value = _streamlit_secret(name)
    if value is None:
        value = os.environ.get(name)
    if value is None and name == "gcp_service_account":
        raw = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
        if raw:
            value = json.loads(raw)
    if value is None:
        return default
    if hasattr(value, "items"):
        return dict(value)
    return value


def has_secret(name: str) -> bool:
    return bool(get_secret(name))


def missing_secrets() -> list[str]:
    """Names of required secrets that are not set."""
    return [name for name in REQUIRED_SECRETS if not has_secret(name)]


# --------------------------------------------------------------------------
# models.toml
# --------------------------------------------------------------------------
# Required nodes. [nodes] may contain extra keys (e.g. a model for one specific tool);
# they get the same validation and are available through ModelConfig.chain().
NODE_NAMES = ("router", "rewriter", "coder", "synthesizer")

# "chat"     = normal chat/completions model (text or structured output)
# "decision" = typed-question decision model (JSON in, JSON out, e.g. mercury-decide)
MODEL_KINDS = ("chat", "decision")

DEFAULT_SETTINGS: dict[str, Any] = {
    "temperature": 0.0,
    "max_code_retries": 2,
    "history_turns": 6,
    "router_confidence_threshold": 0.5,
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    model: str
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    settings: dict[str, Any]
    models: dict[str, ModelSpec]
    nodes: dict[str, tuple[str, ...]]

    def chain(self, node: str) -> list[ModelSpec]:
        """Models for ``node`` in priority order, minus any whose API key is missing."""
        if node not in self.nodes:
            raise ConfigError(f"Unknown node '{node}'. Known nodes: {', '.join(self.nodes)}")
        specs = [self.models[key] for key in self.nodes[node]]
        available = [s for s in specs if has_secret(PROVIDER_SECRETS[s.provider])]
        if not available:
            needed = sorted({PROVIDER_SECRETS[s.provider] for s in specs})
            raise ConfigError(
                f"No usable model for node '{node}'. Set at least one of these secrets: {', '.join(needed)}"
            )
        return available


def load_model_config(path: Path | str = MODELS_FILE) -> ModelConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    settings = {**DEFAULT_SETTINGS, **raw.get("settings", {})}

    models: dict[str, ModelSpec] = {}
    for key, entry in raw.get("models", {}).items():
        for field in ("provider", "model"):
            if field not in entry:
                raise ConfigError(f"[models.{key}] is missing '{field}'")
        if entry["provider"] not in PROVIDER_SECRETS:
            raise ConfigError(
                f"[models.{key}] has unknown provider '{entry['provider']}'. "
                f"Known providers: {', '.join(PROVIDER_SECRETS)}"
            )
        kind = entry.get("kind", "chat")
        if kind not in MODEL_KINDS:
            raise ConfigError(f"[models.{key}] has unknown kind '{kind}'. Known kinds: {', '.join(MODEL_KINDS)}")
        params = {k: v for k, v in entry.items() if k not in ("provider", "model", "kind")}
        models[key] = ModelSpec(key, entry["provider"], entry["model"], kind, params)

    nodes_raw = raw.get("nodes", {})
    for node in NODE_NAMES:
        if not nodes_raw.get(node):
            raise ConfigError(f"[nodes] is missing a non-empty list for '{node}'")
    nodes: dict[str, tuple[str, ...]] = {}
    for node, keys in nodes_raw.items():
        if not keys:
            raise ConfigError(f"[nodes] '{node}' must list at least one model")
        unknown = [k for k in keys if k not in models]
        if unknown:
            raise ConfigError(f"[nodes] '{node}' references undefined models: {', '.join(unknown)}")
        nodes[node] = tuple(keys)

    return ModelConfig(settings=settings, models=models, nodes=nodes)


@lru_cache(maxsize=1)
def get_model_config() -> ModelConfig:
    return load_model_config()
