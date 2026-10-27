import json

import pytest
from langchain_core.tools import tool

from fakes import FakeLLM, decision, write_cfg
from workout_chat import decisions, llm
from workout_chat.graph import GENERIC_ERROR, MAX_QUESTION_CHARS, build_graph, run_turn
from workout_chat.llm import AllModelsFailed, ModelCallError
from workout_chat.prompts import CLARIFY_FALLBACK, TOOL_FAILED_ANSWER
from workout_chat.tools import TOOLS, ToolSpec

SQL_OK = "```sql\nSELECT COUNT(*) AS n FROM workouts\n```"
PY_CHART = "```python\nresult = df.groupby('Exercise')['Weight'].max().reset_index()\nfig = px.bar(result, x='Exercise', y='Weight', title='Max weight')\n```"


@pytest.fixture
def graph(tmp_path, sample_df):
    return build_graph(sample_df, config=write_cfg(tmp_path))


def script(monkeypatch, decisions_list, replies):
    """Fake the router (one decision per turn) and the language models."""
    queue = list(decisions_list)
    seen = []

    def fake_decide(payload, questions, **kw):
        seen.append(payload)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(decisions, "decide", fake_decide)
    fake = FakeLLM(replies)
    monkeypatch.setattr(llm, "invoke_chat", fake)
    return fake, seen


# ---- the main paths -------------------------------------------------------------------
def test_sql_question_end_to_end(graph, monkeypatch):
    fake, _ = script(monkeypatch, [decision("sql")], {"generate_sql_code": SQL_OK, "synthesizer": "Kevin logged 6 sets."})
    steps = []
    res = run_turn(graph, "t1", "How many sets has Kevin done?", on_step=steps.append)
    assert res.answer == "Kevin logged 6 sets." and res.route == "sql" and res.language == "sql"
    assert res.code.startswith("SELECT COUNT") and res.figure_json is None
    assert res.models_used == {"router": "d1", "coder": "g1", "synthesizer": "g1"}
    assert fake.kinds() == ["generate_sql_code", "synthesizer"]
    assert "6" in fake.human_for("synthesizer")[0]                      # the real result reached the writer
    assert steps[0].startswith("Reading") and "Writing the answer..." in steps


def test_python_question_returns_a_chart(graph, monkeypatch):
    script(monkeypatch, [decision("python", chart=0.9)], {"generate_python_code": PY_CHART, "synthesizer": "Squat is heaviest."})
    res = run_turn(graph, "t1", "Plot max weight per exercise")
    assert res.route == "python" and res.language == "python"
    assert json.loads(res.figure_json)["layout"]["title"]["text"] == "Max weight"


def test_off_topic_gets_a_fixed_reply_and_spends_no_tokens(graph, monkeypatch):
    fake, _ = script(monkeypatch, [decision("off_topic", 0.95)], {})
    res = run_turn(graph, "t1", "What's the capital of France?")
    assert "only answer questions about Kevin's workout log" in res.answer and "Bench Press" in res.answer
    assert res.route == "off_topic" and fake.calls == [] and res.code == ""


def test_vague_question_gets_a_clarifying_question(graph, monkeypatch):
    fake, _ = script(monkeypatch, [decision("clarify", 0.8)], {"clarify": "Which exercise do you mean: bench or squat?"})
    res = run_turn(graph, "t1", "How is Kevin doing?")
    assert res.answer == "Which exercise do you mean: bench or squat?" and res.route == "clarify"
    assert fake.kinds() == ["clarify"]


def test_clarify_falls_back_to_fixed_text_if_models_are_down(graph, monkeypatch):
    err = AllModelsFailed("rewriter", [ModelCallError("g1", "rate_limit", "x")])
    script(monkeypatch, [decision("clarify")], {"clarify": err})
    assert run_turn(graph, "t1", "hm?").answer == CLARIFY_FALLBACK


# ---- memory and follow-ups -------------------------------------------------------------------
def test_follow_up_is_rewritten_and_the_tool_gets_the_standalone_question(graph, monkeypatch):
    fake, seen = script(
        monkeypatch,
        [decision("sql"), decision("sql", followup=0.9)],
        {"generate_sql_code": SQL_OK, "synthesizer": "ok", "rewrite": "How many sets of Squat has Kevin done?"},
    )
    run_turn(graph, "t1", "How many sets of bench?")
    res = run_turn(graph, "t1", "and for squats?")
    assert res.standalone_question == "How many sets of Squat has Kevin done?"
    assert "How many sets of bench?" in fake.human_for("rewrite")[0]      # history was used
    assert "Question: How many sets of Squat has Kevin done?" in fake.human_for("generate_sql_code")[-1]
    assert seen[1]["previous_questions"] == ["How many sets of bench?"]


def test_follow_up_survives_a_failed_rewrite(graph, monkeypatch):
    err = AllModelsFailed("rewriter", [ModelCallError("g1", "network", "x")])
    script(monkeypatch, [decision("sql"), decision("sql", followup=0.9)],
           {"generate_sql_code": SQL_OK, "synthesizer": "ok", "rewrite": err})
    run_turn(graph, "t1", "first?")
    assert run_turn(graph, "t1", "and now?").standalone_question == "and now?"


def test_conversation_memory_keeps_only_questions_and_answers(graph, monkeypatch):
    script(monkeypatch, [decision("sql")], {"generate_sql_code": SQL_OK, "synthesizer": "Answer."})
    run_turn(graph, "t1", "one?")
    run_turn(graph, "t1", "two?")
    msgs = graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert [(m.type, m.content) for m in msgs] == [("human", "one?"), ("ai", "Answer."), ("human", "two?"), ("ai", "Answer.")]


def test_threads_are_isolated(graph, monkeypatch):
    script(monkeypatch, [decision("sql")], {"generate_sql_code": SQL_OK, "synthesizer": "A"})
    run_turn(graph, "alice", "q1")
    run_turn(graph, "bob", "q2")
    assert len(graph.get_state({"configurable": {"thread_id": "alice"}}).values["messages"]) == 2


def test_nothing_leaks_from_the_previous_turn(graph, monkeypatch):
    script(monkeypatch, [decision("python", chart=0.9), decision("sql")],
           {"generate_python_code": PY_CHART, "generate_sql_code": SQL_OK, "synthesizer": "A"})
    first = run_turn(graph, "t1", "Plot it")
    second = run_turn(graph, "t1", "How many sets?")
    assert first.figure_json and second.figure_json is None
    assert second.language == "sql" and second.code.startswith("SELECT")


# ---- failure handling ---------------------------------------------------------------------------------
def test_router_outage_returns_a_friendly_message(graph, monkeypatch):
    err = AllModelsFailed("router", [ModelCallError("d1", "rate_limit", "x"), ModelCallError("g1", "rate_limit", "y")])

    def boom(*a, **k):
        raise err

    monkeypatch.setattr(decisions, "decide", boom)
    monkeypatch.setattr(llm, "invoke_chat", FakeLLM({}))
    res = run_turn(graph, "t1", "q")
    assert res.route == "error" and "rate or daily limit" in res.answer


def test_code_that_never_works_asks_for_a_rephrase_without_calling_the_writer(graph, monkeypatch):
    fake, _ = script(monkeypatch, [decision("sql")], {"generate_sql_code": "```sql\nSELECT nope FROM workouts\n```"})
    res = run_turn(graph, "t1", "q")
    assert res.answer == TOOL_FAILED_ANSWER and res.error and "synthesizer" not in fake.kinds()
    assert fake.kinds().count("generate_sql_code") == 3


def test_models_down_during_code_generation_says_so(graph, monkeypatch):
    err = AllModelsFailed("generate_sql_code", [ModelCallError("g1", "rate_limit", "x")])
    script(monkeypatch, [decision("sql")], {"generate_sql_code": err})
    assert "rate or daily limit" in run_turn(graph, "t1", "q").answer


def test_writer_outage_still_returns_the_raw_result(graph, monkeypatch):
    err = AllModelsFailed("synthesizer", [ModelCallError("g1", "rate_limit", "x")])
    script(monkeypatch, [decision("sql")], {"generate_sql_code": SQL_OK, "synthesizer": err})
    res = run_turn(graph, "t1", "q")
    assert "couldn't write it up" in res.answer and "6" in res.answer and res.code


def test_unexpected_bug_never_crashes_the_ui(graph, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(decisions, "decide", boom)
    res = run_turn(graph, "t1", "q")
    assert res.answer == GENERIC_ERROR and "kaboom" in res.error


def test_very_long_questions_are_truncated(graph, monkeypatch):
    _, seen = script(monkeypatch, [decision("sql")], {"generate_sql_code": SQL_OK, "synthesizer": "A"})
    run_turn(graph, "t1", "x" * 5000)
    assert len(seen[0]["current_question"]) == MAX_QUESTION_CHARS


# ---- extensibility ---------------------------------------------------------------------------------------
def test_a_new_tool_works_without_touching_the_graph(tmp_path, sample_df, monkeypatch):
    def build(ctx):
        @tool("lookup_pr", response_format="content_and_artifact")
        def lookup_pr(question: str, needs_chart: bool = False):
            """Look up a personal record."""
            return "STATUS: success", {"ok": True, "text": "PR: 315 lbs", "code": "n/a", "language": "custom", "figure_json": None}
        return lookup_pr

    extra = ToolSpec("lookup_pr", "pr", "Personal record lookups.", build)
    graph = build_graph(sample_df, config=write_cfg(tmp_path), specs=TOOLS + (extra,))
    fake, _ = script(monkeypatch, [decision("pr")], {"synthesizer": "Kevin's PR is 315 lbs."})
    res = run_turn(graph, "t1", "What is Kevin's PR?")
    assert res.answer == "Kevin's PR is 315 lbs." and res.language == "custom"
    assert "PR: 315 lbs" in fake.human_for("synthesizer")[0]
