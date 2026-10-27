"""Shared types for tools. A tool = a ToolSpec; the router's choices are built from the registry."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import pandas as pd
from langchain_core.tools import BaseTool

from ..config import ModelConfig, get_model_config
from ..schema import build_schema_context


@dataclass(frozen=True, eq=False)
class ToolContext:
    df: pd.DataFrame
    schema_context: str
    settings: Mapping[str, Any]
    config: ModelConfig
    today: str
    latest_date: str
    top_exercise: str

    @classmethod
    def from_df(cls, df: pd.DataFrame, config: ModelConfig | None = None) -> "ToolContext":
        cfg = config or get_model_config()
        top = df["Exercise"].value_counts().index[0] if df["Exercise"].notna().any() else "bench press"
        return cls(
            df=df,
            schema_context=build_schema_context(df),
            settings=cfg.settings,
            config=cfg,
            today=datetime.now(timezone.utc).date().isoformat(),
            latest_date=df["Date"].max().date().isoformat(),
            top_exercise=str(top),
        )


@dataclass(frozen=True)
class ToolSpec:
    name: str                                  # LangChain tool name; also the models.toml node for its model
    route: str                                 # the router option that selects this tool
    route_description: str                     # how the router tells when to pick it
    build: Callable[[ToolContext], BaseTool]
