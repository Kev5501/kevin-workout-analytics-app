from __future__ import annotations

from langchain_core.tools import tool

from .. import prompts, sandbox
from .base import ToolContext, ToolSpec
from .runner import CodeLanguage, generate_and_run, tool_artifact, tool_content


def _system(ctx: ToolContext, needs_chart: bool) -> str:
    return prompts.PYTHON_SYSTEM.format(
        chart_rule=prompts.CHART_REQUIRED if needs_chart else prompts.CHART_OPTIONAL,
        today=ctx.today, latest=ctx.latest_date, schema=ctx.schema_context,
    )


PYTHON = CodeLanguage(name="python", fence=("python", "py"), node="generate_python_code", system_prompt=_system,
                      run=sandbox.run_python)


def _build(ctx: ToolContext):
    @tool("generate_python_code", response_format="content_and_artifact")
    def generate_python_code(question: str, needs_chart: bool = False) -> tuple[str, dict]:
        """Write and run pandas/plotly code over Kevin's workout data for trends, progression, comparisons,
        correlations, multi-step calculations and charts. Returns the result and, if useful, a chart."""
        outcome = generate_and_run(ctx, PYTHON, question, needs_chart)
        return tool_content(outcome), tool_artifact(outcome)

    return generate_python_code


PYTHON_TOOL = ToolSpec(
    name="generate_python_code",
    route="python",
    route_description=(
        "Needs trends over time, progression, comparisons, correlations, multi-step calculations, or any chart, plot "
        "or graph. Examples: how has bench press progressed, plot body weight over the year, does a high RPE relate "
        "to the workout rating."
    ),
    build=_build,
)
