from __future__ import annotations

from langchain_core.tools import tool

from .. import prompts, sandbox
from ..schema import SQL_TABLE_NAME
from .base import ToolContext, ToolSpec
from .runner import CodeLanguage, generate_and_run, tool_artifact, tool_content


def _system(ctx: ToolContext, needs_chart: bool) -> str:
    return prompts.SQL_SYSTEM.format(
        table=SQL_TABLE_NAME, today=ctx.today, latest=ctx.latest_date, schema=ctx.schema_context
    )


SQL = CodeLanguage(name="sql", fence=("sql", "duckdb"), node="generate_sql_code", system_prompt=_system, run=sandbox.run_sql)


def _build(ctx: ToolContext):
    @tool("generate_sql_code", response_format="content_and_artifact")
    def generate_sql_code(question: str, needs_chart: bool = False) -> tuple[str, dict]:
        """Write and run one read-only DuckDB SQL query over Kevin's workout table to answer a lookup, filter,
        count, sum, average, ranking or grouping question. Returns the query result."""
        outcome = generate_and_run(ctx, SQL, question, needs_chart)
        return tool_content(outcome), tool_artifact(outcome)

    return generate_sql_code


SQL_TOOL = ToolSpec(
    name="generate_sql_code",
    route="sql",
    route_description=(
        "Look up, filter, count, sum, average, rank or group rows to answer with numbers or a short table. "
        "Examples: total sets of an exercise, heaviest weight, average RPE by gym day, number of workouts last month."
    ),
    build=_build,
)
