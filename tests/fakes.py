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

    def __call__(self, node, messages, *, schema=None, default_node=None, effort=None, config=None):
        system = messages[0][1] if isinstance(messages[0], tuple) else str(messages[0].content)
        human = messages[-1][1] if isinstance(messages[-1], tuple) else str(messages[-1].content)
        kind = node
        if node == "rewriter":
            kind = "clarify" if "clarifying" in system else "rewrite"
        self.calls.append({"node": node, "kind": kind, "system": system, "human": human, "default_node": default_node,
                           "effort": effort})
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


# ---- Firestore -------------------------------------------------------------------------------
class FakeFirestore:
    """The slice of google.cloud.firestore.Client that conversations.FirestoreStore uses, in memory.

    ``fail`` makes every call raise (to test the fallback); ``calls`` records (op, collection) pairs.
    """

    def __init__(self, fail: Exception | None = None):
        self.data: dict[str, dict[str, dict]] = {}
        self.fail = fail
        self.calls: list[tuple] = []

    def collection(self, name):
        return _FakeQuery(self, name)

    def _check(self, op, name):
        self.calls.append((op, name))
        if self.fail:
            raise self.fail


class _FakeSnap:
    def __init__(self, ref, data, fields=None):
        self.reference, self.id, self.exists = ref, ref.id, data is not None
        self._data = None if data is None else {k: v for k, v in data.items() if fields is None or k in fields}

    def to_dict(self):
        return None if self._data is None else dict(self._data)


class _FakeDoc:
    def __init__(self, db, col, doc_id):
        self.db, self.col, self.id = db, col, doc_id

    def get(self, timeout=None):
        self.db._check("get", self.col)
        return _FakeSnap(self, self.db.data.get(self.col, {}).get(self.id))

    def set(self, data, timeout=None):
        self.db._check("set", self.col)
        self.db.data.setdefault(self.col, {})[self.id] = dict(data)

    def delete(self, timeout=None):
        self.db._check("delete", self.col)
        self.db.data.get(self.col, {}).pop(self.id, None)


class _FakeQuery:
    def __init__(self, db, col, order=None, desc=False, lim=None, skip=0, fields=None):
        self.db, self.col = db, col
        self._o, self._desc, self._lim, self._skip, self._fields = order, desc, lim, skip, fields

    def _with(self, **kw):
        args = dict(order=self._o, desc=self._desc, lim=self._lim, skip=self._skip, fields=self._fields)
        args.update(kw)
        return _FakeQuery(self.db, self.col, **args)

    def document(self, doc_id):
        return _FakeDoc(self.db, self.col, doc_id)

    def order_by(self, field, direction="ASCENDING"):
        return self._with(order=field, desc=direction == "DESCENDING")

    def limit(self, n):
        return self._with(lim=n)

    def offset(self, n):
        return self._with(skip=n)

    def select(self, fields):
        return self._with(fields=list(fields))

    def stream(self, timeout=None):
        self.db._check("stream", self.col)
        items = list(self.db.data.get(self.col, {}).items())
        if self._o:
            items.sort(key=lambda kv: kv[1].get(self._o, 0), reverse=self._desc)
        items = items[self._skip:]
        if self._lim is not None:
            items = items[: self._lim]
        return [_FakeSnap(_FakeDoc(self.db, self.col, k), v, self._fields) for k, v in items]
