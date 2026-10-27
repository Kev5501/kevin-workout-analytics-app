import pytest
from langchain_core.messages import AIMessage, HumanMessage

from fakes import decision, write_cfg
from workout_chat import decisions
from workout_chat.config import ConfigError
from workout_chat.llm import AllModelsFailed, ModelCallError
from workout_chat.nodes import router
from workout_chat.nodes.router import build_questions, make_prepare_node, make_route_node, next_after_route
from workout_chat.tools import TOOLS, ToolContext, ToolSpec


@pytest.fixture
def ctx(tmp_path, sample_df):
    return ToolContext.from_df(sample_df, write_cfg(tmp_path))


def state(question="How many sets?", history=()):
    msgs = []
    for q, a in history:
        msgs += [HumanMessage(q), AIMessage(a)]
    return {"messages": msgs + [HumanMessage(question)], "question": question}


def run_route(ctx, monkeypatch, result, st=None):
    seen = {}

    def fake_decide(payload, questions, **kw):
        seen.update(payload=payload, questions=questions, kw=kw)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(decisions, "decide", fake_decide)
    update = make_route_node(ctx, TOOLS)(st or state())
    return update, seen


# ---- questions ---------------------------------------------------------------------
def test_questions_are_built_from_the_tool_registry():
    q = build_questions(TOOLS)
    assert set(q) == {"route", "needs_chart", "is_followup"}
    assert set(q["route"].options) == {"sql", "python", "clarify", "off_topic"}
    assert "Kevin" in q["route"].instructions and "rating" in q["route"].instructions
    decisions.validate_questions(q)  # valid for the Decisions API


def test_a_new_tool_appears_in_the_router_automatically():
    extra = ToolSpec("do_stats", "stats", "Advanced statistics.", build=lambda ctx: None)
    q = build_questions(TOOLS + (extra,))
    assert q["route"].options["stats"] == "Advanced statistics."


# ---- routing decisions ------------------------------------------------------------------
def test_confident_sql_route(ctx, monkeypatch):
    update, seen = run_route(ctx, monkeypatch, decision("sql", 0.9))
    assert update["route"] == "sql" and update["router_model"] == "d1" and update["needs_chart"] is False
    assert seen["kw"]["node"] == "router"


def test_payload_has_current_and_previous_questions(ctx, monkeypatch):
    st = state("and for squats?", history=[("bench max?", "225"), ("rpe avg?", "8"), ("total sets?", "50")])
    _, seen = run_route(ctx, monkeypatch, decision(), st)
    assert seen["payload"] == {"current_question": "and for squats?", "previous_questions": ["rpe avg?", "total sets?"]}


def test_low_confidence_asks_for_clarification(ctx, monkeypatch):
    update, _ = run_route(ctx, monkeypatch, decision("sql", confidence=0.3))
    assert update["route"] == "clarify"


def test_low_confidence_off_topic_also_clarifies(ctx, monkeypatch):
    assert run_route(ctx, monkeypatch, decision("off_topic", confidence=0.4))[0]["route"] == "clarify"


def test_confident_off_topic(ctx, monkeypatch):
    assert run_route(ctx, monkeypatch, decision("off_topic", 0.95))[0]["route"] == "off_topic"


def test_sql_plus_chart_is_switched_to_python(ctx, monkeypatch):
    update, _ = run_route(ctx, monkeypatch, decision("sql", 0.9, chart=0.9))
    assert update["route"] == "python" and update["needs_chart"] is True


def test_followup_needs_history(ctx, monkeypatch):
    assert run_route(ctx, monkeypatch, decision(followup=0.9))[0]["is_followup"] is False
    st = state("and squats?", history=[("bench?", "225")])
    assert run_route(ctx, monkeypatch, decision(followup=0.9), st)[0]["is_followup"] is True


def test_fallback_model_is_reported(ctx, monkeypatch):
    update, _ = run_route(ctx, monkeypatch, decision(model="g1", approximate=True))
    assert update["router_model"] == "g1" and update["router_approximate"] is True


def test_router_outage_becomes_a_friendly_error(ctx, monkeypatch):
    err = AllModelsFailed("router", [ModelCallError("d1", "rate_limit", "x")])
    update, _ = run_route(ctx, monkeypatch, err)
    assert update["route"] == "error" and "rate or daily limit" in update["answer"]


def test_router_config_problem_becomes_a_friendly_error(ctx, monkeypatch):
    update, _ = run_route(ctx, monkeypatch, ConfigError("no keys"))
    assert update["route"] == "error" and "not configured" in update["answer"]


# ---- edges and tool call --------------------------------------------------------------
@pytest.mark.parametrize("st,expected", [
    ({"route": "error"}, "end"),
    ({"route": "off_topic"}, "off_topic"),
    ({"route": "clarify"}, "clarify"),
    ({"route": "mystery"}, "clarify"),
    ({"route": "sql", "is_followup": False}, "prepare"),
    ({"route": "python", "is_followup": True}, "rewrite"),
])
def test_next_after_route(st, expected):
    assert next_after_route({"sql", "python"})(st) == expected


def test_prepare_builds_a_tool_call_with_the_standalone_question():
    update = make_prepare_node(TOOLS)({"route": "python", "standalone_question": "Plot X", "needs_chart": True})
    call = update["tool_messages"][0].tool_calls[0]
    assert call["name"] == "generate_python_code"
    assert call["args"] == {"question": "Plot X", "needs_chart": True}
