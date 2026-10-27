"""Test doubles for the model calls. Everything else (sandbox, ToolNode, graph, memory) is real."""
from langchain_core.messages import AIMessage

from workout_chat import decisions, llm
from workout_chat.config import load_model_config
from workout_chat.decisions import Answer, DecisionResult

MODELS_TOML = """
[models.d1]
provider = "openrouter"
model = "decide"
kind = "decision"
[models.g1]
provider = "google_genai"
model = "gem"
[nodes]
router = ["d1", "g1"]
rewriter = ["g1"]
coder = ["g1"]
synthesizer = ["g1"]
"""


def write_cfg(tmp_path, extra_settings=""):
    p = tmp_path / "models.toml"
    p.write_text(f"[settings]\n{extra_settings}\n" + MODELS_TOML)
    return load_model_config(p)


def decision(route="sql", confidence=0.9, chart=0.05, followup=0.05, model="d1", approximate=False):
    """A DecisionResult as the router would receive it."""
    return DecisionResult(
        answers={
            "route": Answer("choice", route, confidence, {route: confidence}),
            "needs_chart": Answer("noul", chart, max(chart, 1 - chart), {}),
            "is_followup": Answer("noul", followup, max(followup, 1 - followup), {}),
        },
        model_key=model, approximate=approximate,
    )


def chat_result(text, model="g1"):
    return llm.ChatResult(AIMessage(content=text), model)


class FakeLLM:
    """Replaces llm.invoke_chat. ``replies`` maps a node name to a text, a list of texts (one per call) or a callable."""

    def __init__(self, replies):
        self.replies, self.calls = dict(replies), []

    def __call__(self, node, messages, *, schema=None, default_node=None, config=None):
        system = messages[0][1] if isinstance(messages[0], tuple) else str(messages[0].content)
        human = messages[-1][1] if isinstance(messages[-1], tuple) else str(messages[-1].content)
        kind = node
        if node == "rewriter":
            kind = "clarify" if "clarifying" in system else "rewrite"
        self.calls.append({"node": node, "kind": kind, "system": system, "human": human, "default_node": default_node})
        reply = self.replies.get(kind, self.replies.get(default_node or "", ""))
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(human)
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        return chat_result(reply)

    def kinds(self):
        return [c["kind"] for c in self.calls]

    def human_for(self, kind):
        return [c["human"] for c in self.calls if c["kind"] == kind]
