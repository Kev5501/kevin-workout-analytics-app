import pytest
from types import SimpleNamespace

from workout_chat import config, llm
from workout_chat.config import ConfigError, load_model_config
from workout_chat.llm import AllModelsFailed, ModelCallError, classify_error, invoke_chat, text_of

MODELS = """
[models.g1]
provider = "google_genai"
model = "gem-1"
[models.g2]
provider = "google_genai"
model = "gem-2"
temperature = 0.9
[models.o1]
provider = "openrouter"
model = "or/one:free"
[models.d1]
provider = "openrouter"
model = "or/decide:free"
kind = "decision"
"""


def make_cfg(tmp_path, nodes):
    lines = "\n".join(f"{k} = {v!r}".replace("'", '"') for k, v in nodes.items())
    p = tmp_path / "models.toml"
    p.write_text(MODELS + f"\n[nodes]\n{lines}\n")
    return load_model_config(p)


CORE = {"router": ["g1"], "rewriter": ["g1"], "coder": ["o1", "g1"], "synthesizer": ["g1"]}


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for name in (*config.REQUIRED_SECRETS, *config.OPTIONAL_SECRETS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    llm.clear_model_cache()


class Err(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


# ---- error classification -----------------------------------------------------
@pytest.mark.parametrize("exc,kind", [
    (Err("boom", 429), "rate_limit"),
    (Err("429 RESOURCE_EXHAUSTED: quota exceeded"), "rate_limit"),
    (Err("bad", 401), "auth"),
    (Err("API key not valid"), "auth"),
    (Err("payment", 402), "auth"),
    (Err("model not found", 404), "unavailable"),
    (Err("upstream", 503), "unavailable"),
    (TimeoutError("slow"), "network"),
    (Err("Connection reset"), "network"),
    (Err("something odd"), "other"),
])
def test_classify_error(exc, kind):
    assert classify_error(exc) == kind


def test_classify_uses_nested_response_status():
    exc = Exception("failed")
    exc.response = SimpleNamespace(status_code=429)
    assert classify_error(exc) == "rate_limit"


# ---- building models --------------------------------------------------------
def test_build_google_model_passes_key_and_settings(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", lambda **kw: seen.update(kw) or "G")
    cfg = make_cfg(tmp_path, CORE)
    assert llm.build_chat_model(cfg.models["g1"], {"temperature": 0.0}) == "G"
    assert seen["model"] == "gem-1" and seen["google_api_key"] == "gem-key"
    assert seen["temperature"] == 0.0 and seen["max_retries"] == 1


def test_model_params_override_settings(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", lambda **kw: seen.update(kw) or "G")
    cfg = make_cfg(tmp_path, CORE)
    llm.build_chat_model(cfg.models["g2"], {"temperature": 0.0})
    assert seen["temperature"] == 0.9


def test_build_openrouter_model_uses_base_url_and_title_header(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(llm, "ChatOpenAI", lambda **kw: seen.update(kw) or "O")
    cfg = make_cfg(tmp_path, CORE)
    llm.build_chat_model(cfg.models["o1"], {"temperature": 0.0})
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["api_key"] == "or-key" and seen["model"] == "or/one:free"
    assert seen["default_headers"]["X-OpenRouter-Title"] == llm.APP_TITLE


def test_models_are_cached(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", lambda **kw: calls.append(1) or object())
    cfg = make_cfg(tmp_path, CORE)
    a = llm.build_chat_model(cfg.models["g1"], {"temperature": 0.0})
    b = llm.build_chat_model(cfg.models["g1"], {"temperature": 0.0})
    assert a is b and len(calls) == 1


def test_decision_models_cannot_be_built_as_chat(tmp_path):
    cfg = make_cfg(tmp_path, CORE)
    with pytest.raises(ConfigError, match="not 'chat'"):
        llm.build_chat_model(cfg.models["d1"], {})


def test_real_constructors_accept_our_kwargs(tmp_path):
    """No network: just proves the argument names match the installed LangChain classes."""
    cfg = make_cfg(tmp_path, CORE)
    assert llm.build_chat_model(cfg.models["g1"], {"temperature": 0.0}) is not None
    assert llm.build_chat_model(cfg.models["o1"], {"temperature": 0.0}) is not None


# ---- invoke_chat fallback -------------------------------------------------------
class FakeModel:
    def __init__(self, behaviour):
        self.behaviour, self.schema = behaviour, None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour


def patch_models(monkeypatch, by_key):
    monkeypatch.setattr(llm, "build_chat_model", lambda spec, settings: by_key[spec.key])


def test_invoke_chat_falls_back_and_records_failures(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, CORE)
    patch_models(monkeypatch, {"o1": FakeModel(Err("slow down", 429)), "g1": FakeModel("hello")})
    res = invoke_chat("coder", [("human", "hi")], config=cfg)
    assert res.value == "hello" and res.model_key == "g1"
    assert [f.model_key for f in res.failures] == ["o1"] and res.failures[0].kind == "rate_limit"


def test_invoke_chat_uses_first_model_when_it_works(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, CORE)
    patch_models(monkeypatch, {"o1": FakeModel("first"), "g1": FakeModel("second")})
    res = invoke_chat("coder", "hi", config=cfg)
    assert res.model_key == "o1" and res.failures == ()


def test_all_failed_raises_with_friendly_message(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, CORE)
    patch_models(monkeypatch, {"o1": FakeModel(Err("x", 429)), "g1": FakeModel(Err("y", 503))})
    with pytest.raises(AllModelsFailed) as info:
        invoke_chat("coder", "hi", config=cfg)
    assert len(info.value.failures) == 2
    assert "rate or daily limit" in info.value.user_message


def test_auth_only_failures_get_owner_message(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, {**CORE, "coder": ["o1"]})
    patch_models(monkeypatch, {"o1": FakeModel(Err("bad key", 401))})
    with pytest.raises(AllModelsFailed) as info:
        invoke_chat("coder", "hi", config=cfg)
    assert "app owner" in info.value.user_message


def test_schema_is_applied_and_none_result_counts_as_failure(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, CORE)
    good = FakeModel({"ok": True})
    patch_models(monkeypatch, {"o1": FakeModel(None), "g1": good})
    res = invoke_chat("coder", "hi", schema=dict, config=cfg)
    assert good.schema is dict and res.model_key == "g1"
    assert res.failures[0].kind == "bad_response"


def test_decision_models_are_skipped_for_text_nodes(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, {**CORE, "rewriter": ["d1", "g1"]})
    patch_models(monkeypatch, {"g1": FakeModel("ok")})
    assert invoke_chat("rewriter", "hi", config=cfg).model_key == "g1"


def test_node_with_only_decision_models_is_a_config_error(tmp_path):
    cfg = make_cfg(tmp_path, {**CORE, "rewriter": ["d1"]})
    with pytest.raises(ConfigError, match="no usable chat models"):
        invoke_chat("rewriter", "hi", config=cfg)


def test_tool_node_uses_own_chain_or_default(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, {**CORE, "generate_sql_code": ["g1"]})
    patch_models(monkeypatch, {"o1": FakeModel("coder"), "g1": FakeModel("sql-tool")})
    assert invoke_chat("generate_sql_code", "q", default_node="coder", config=cfg).value == "sql-tool"
    assert invoke_chat("generate_python_code", "q", default_node="coder", config=cfg).value == "coder"
    with pytest.raises(ConfigError, match="no \\[nodes\\] entry"):
        invoke_chat("generate_python_code", "q", config=cfg)


def test_missing_key_skips_model_in_chain(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    cfg = make_cfg(tmp_path, CORE)
    patch_models(monkeypatch, {"g1": FakeModel("gemini")})
    assert invoke_chat("coder", "hi", config=cfg).model_key == "g1"


# ---- text_of -----------------------------------------------------------------
def test_text_of_handles_strings_and_content_blocks():
    assert text_of(SimpleNamespace(content="plain")) == "plain"
    assert text_of(SimpleNamespace(content=[{"type": "text", "text": "a"}, "b", {"type": "x"}])) == "ab"
    assert text_of("raw") == "raw"
