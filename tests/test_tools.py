import pytest
from langchain_core.messages import ToolMessage

from fakes import FakeLLM, write_cfg
from workout_chat import llm
from workout_chat.llm import AllModelsFailed, ModelCallError
from workout_chat.tools import TOOLS, ToolContext, build_tools
from workout_chat.tools.runner import extract_code, generate_and_run
from workout_chat.tools.sql_tool import SQL
from workout_chat.tools.python_tool import PYTHON


@pytest.fixture
def ctx(tmp_path, sample_df):
    return ToolContext.from_df(sample_df, write_cfg(tmp_path))


def patch_llm(monkeypatch, replies):
    fake = FakeLLM(replies)
    monkeypatch.setattr(llm, "invoke_chat", fake)
    return fake


# ---- code extraction ---------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("Here you go:\n```SQL\nSELECT 1;\n```\nHope it helps", "SELECT 1;"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("```python\nx = 1\n```\n```sql\nSELECT 2\n```", "SELECT 2"),     # prefers the right language
    ("```sql\nSELECT 1", "SELECT 1"),                                  # unterminated fence
    ("SELECT 1", "SELECT 1"),                                          # no fence at all
    ("   ", ""),
])
def test_extract_code(text, expected):
    assert extract_code(text, ("sql", "duckdb")) == expected


# ---- the generate -> run -> retry loop ------------------------------------------------
def test_success_on_first_attempt(ctx, monkeypatch):
    fake = patch_llm(monkeypatch, {"generate_sql_code": "```sql\nSELECT COUNT(*) AS n FROM workouts\n```"})
    out = generate_and_run(ctx, SQL, "How many sets?")
    assert out.ok and out.attempts == 1 and "6" in out.text and out.language == "sql"
    assert fake.calls[0]["node"] == "generate_sql_code" and fake.calls[0]["default_node"] == "coder"
    assert "Question: How many sets?" in fake.calls[0]["human"]


def test_retry_feeds_back_the_code_and_the_error(ctx, monkeypatch):
    fake = patch_llm(monkeypatch, {"generate_sql_code": [
        "```sql\nSELECT Wieght FROM workouts\n```",
        "```sql\nSELECT MAX(Weight) AS heaviest FROM workouts\n```",
    ]})
    out = generate_and_run(ctx, SQL, "Heaviest set?")
    assert out.ok and out.attempts == 2 and "heaviest" in out.text
    retry_prompt = fake.calls[1]["human"]
    assert "Wieght" in retry_prompt and "Previous code" in retry_prompt and "Error:" in retry_prompt


def test_gives_up_after_max_retries(ctx, monkeypatch):
    fake = patch_llm(monkeypatch, {"generate_sql_code": "```sql\nSELECT nope FROM workouts\n```"})
    out = generate_and_run(ctx, SQL, "q")
    assert not out.ok and out.attempts == 3 and len(fake.calls) == 3  # 1 try + max_code_retries (2)
    assert out.error and out.code and len(out.history) == 3 and not out.llm_failed


def test_reply_without_code_counts_as_a_failed_attempt(ctx, monkeypatch):
    patch_llm(monkeypatch, {"generate_sql_code": ["", "```sql\nSELECT 1 AS one\n```"]})
    out = generate_and_run(ctx, SQL, "q")
    assert out.ok and out.attempts == 2


def test_all_models_failing_is_reported_as_llm_failed(ctx, monkeypatch):
    err = AllModelsFailed("generate_sql_code", [ModelCallError("g1", "rate_limit", "quota")])
    patch_llm(monkeypatch, {"generate_sql_code": err})
    out = generate_and_run(ctx, SQL, "q")
    assert not out.ok and out.llm_failed and "rate or daily limit" in out.error


def test_python_tool_runs_in_the_sandbox_and_can_chart(ctx, monkeypatch):
    code = "```python\nresult = df.groupby('Exercise')['Weight'].max().reset_index()\nfig = px.bar(result, x='Exercise', y='Weight')\n```"
    fake = patch_llm(monkeypatch, {"generate_python_code": code})
    out = generate_and_run(ctx, PYTHON, "Plot max weight per exercise", needs_chart=True)
    assert out.ok and out.figure_json and "Squat" in out.text
    assert "MUST also create a Plotly figure" in fake.calls[0]["system"]


def test_chart_is_optional_when_not_requested(ctx, monkeypatch):
    fake = patch_llm(monkeypatch, {"generate_python_code": "```python\nresult = 1\n```"})
    generate_and_run(ctx, PYTHON, "q", needs_chart=False)
    assert "ONLY if a chart clearly helps" in fake.calls[0]["system"]


def test_blocked_python_is_retried_like_any_error(ctx, monkeypatch):
    fake = patch_llm(monkeypatch, {"generate_python_code": ["```python\nimport os\nresult = 1\n```", "```python\nresult = 2\n```"]})
    out = generate_and_run(ctx, PYTHON, "q")
    assert out.ok and out.attempts == 2 and "Imports are not allowed" in fake.calls[1]["human"]


def test_prompt_contains_schema_dates_and_kevin(ctx, monkeypatch):
    fake = patch_llm(monkeypatch, {"generate_sql_code": "```sql\nSELECT 1\n```"})
    generate_and_run(ctx, SQL, "q")
    system = fake.calls[0]["system"]
    assert "Kevin's workout log" in system and ctx.today in system and ctx.latest_date in system
    assert "Exact values" in system and "Comment" in system and "SECRET" not in system


# ---- tools through LangChain's tool interface -----------------------------------------
def test_registry_is_consistent():
    assert [t.name for t in TOOLS] == ["generate_sql_code", "generate_python_code"]
    assert len({t.route for t in TOOLS}) == len(TOOLS)
    assert all(t.route_description.strip() for t in TOOLS)


def test_tool_call_returns_content_and_artifact(ctx, monkeypatch):
    patch_llm(monkeypatch, {"generate_sql_code": "```sql\nSELECT COUNT(*) AS n FROM workouts\n```"})
    sql_tool = next(t for t in build_tools(ctx) if t.name == "generate_sql_code")
    msg = sql_tool.invoke({"name": sql_tool.name, "args": {"question": "How many sets?"}, "id": "1", "type": "tool_call"})
    assert isinstance(msg, ToolMessage) and "STATUS: success" in msg.content
    assert msg.artifact["ok"] and msg.artifact["code"].startswith("SELECT") and msg.artifact["language"] == "sql"


def test_failed_tool_call_still_returns_a_message(ctx, monkeypatch):
    patch_llm(monkeypatch, {"generate_sql_code": "```sql\nSELECT nope FROM workouts\n```"})
    sql_tool = next(t for t in build_tools(ctx) if t.name == "generate_sql_code")
    msg = sql_tool.invoke({"name": sql_tool.name, "args": {"question": "q"}, "id": "1", "type": "tool_call"})
    assert "STATUS: failed after 3 attempt(s)" in msg.content and msg.artifact["ok"] is False
