"""Tool registry. To add a tool: write a ToolSpec (see sql_tool.py) and append it to TOOLS.

The router's choices are built from this list, and the graph's tool node runs whatever the router
picks, so nothing else needs to change. Give the tool its own model in models.toml by adding a
[nodes] entry named after the tool (otherwise it uses the `coder` chain).
"""
from __future__ import annotations

from .base import ToolContext, ToolSpec
from .python_tool import PYTHON_TOOL
from .sql_tool import SQL_TOOL

TOOLS: tuple[ToolSpec, ...] = (SQL_TOOL, PYTHON_TOOL)


def build_tools(ctx: ToolContext, specs: tuple[ToolSpec, ...] = TOOLS):
    return [spec.build(ctx) for spec in specs]


__all__ = ["TOOLS", "ToolContext", "ToolSpec", "build_tools"]
