import pytest

from workout_chat import config
from workout_chat.config import ConfigError, load_model_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (*config.REQUIRED_SECRETS, *config.OPTIONAL_SECRETS):
        monkeypatch.delenv(name, raising=False)


def write_toml(tmp_path, models="", nodes=None, extra=""):
    nodes = nodes or {n: ["a"] for n in config.NODE_NAMES}
    node_lines = "\n".join(f"{n} = {list(keys)!r}".replace("'", '"') for n, keys in nodes.items())
    text = (
        models
        or '[models.a]\nprovider = "google_genai"\nmodel = "m-a"\n[models.b]\nprovider = "openrouter"\nmodel = "m-b"\n'
    ) + f"\n[nodes]\n{node_lines}\n{extra}"
    path = tmp_path / "models.toml"
    path.write_text(text)
    return path


def test_repo_models_toml_is_valid(monkeypatch):
    """Whatever models you pick, the repo's models.toml must stay internally consistent."""
    cfg = load_model_config()
    assert set(config.NODE_NAMES) <= set(cfg.nodes)
    assert 0 < cfg.settings["router_confidence_threshold"] < 1
    for node, keys in cfg.nodes.items():
        assert keys and all(k in cfg.models for k in keys), node
    # with every provider key present, every node resolves to a non-empty chain
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    for node in cfg.nodes:
        assert cfg.chain(node)


def test_chain_skips_models_without_secrets(tmp_path, monkeypatch):
    nodes = {n: ["b", "a"] for n in config.NODE_NAMES}
    cfg = load_model_config(write_toml(tmp_path, nodes=nodes))
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert [s.key for s in cfg.chain("router")] == ["a"]  # no OpenRouter key -> falls back
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    assert [s.key for s in cfg.chain("router")] == ["b", "a"]


def test_chain_with_no_usable_model_names_the_secrets(tmp_path):
    cfg = load_model_config(write_toml(tmp_path))
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        cfg.chain("synthesizer")


def test_unknown_node_in_chain_rejected(tmp_path):
    cfg = load_model_config(write_toml(tmp_path))
    with pytest.raises(ConfigError, match="Unknown node"):
        cfg.chain("nope")


def test_undefined_model_reference_rejected(tmp_path):
    nodes = {n: ["a"] for n in config.NODE_NAMES} | {"coder": ["nope"]}
    with pytest.raises(ConfigError, match="nope"):
        load_model_config(write_toml(tmp_path, nodes=nodes))


def test_unknown_provider_rejected(tmp_path):
    with pytest.raises(ConfigError, match="mystery"):
        load_model_config(write_toml(tmp_path, models='[models.a]\nprovider = "mystery"\nmodel = "x"\n'))


def test_missing_core_node_rejected(tmp_path):
    with pytest.raises(ConfigError, match="rewriter"):
        load_model_config(write_toml(tmp_path, nodes={"router": ["a"]}))


def test_extra_model_keys_become_params(tmp_path):
    models = '[models.a]\nprovider = "openrouter"\nmodel = "x"\nmax_tokens = 512\n'
    spec = load_model_config(write_toml(tmp_path, models=models)).models["a"]
    assert spec.params == {"max_tokens": 512}
    assert spec.kind == "chat"  # default


def test_decision_kind_parsed_and_not_a_param(tmp_path):
    models = '[models.a]\nprovider = "openrouter"\nmodel = "x"\nkind = "decision"\n'
    spec = load_model_config(write_toml(tmp_path, models=models)).models["a"]
    assert spec.kind == "decision" and spec.params == {}


def test_invalid_kind_rejected(tmp_path):
    models = '[models.a]\nprovider = "openrouter"\nmodel = "x"\nkind = "banana"\n'
    with pytest.raises(ConfigError, match="banana"):
        load_model_config(write_toml(tmp_path, models=models))


def test_extra_nodes_are_loaded_and_validated(tmp_path):
    cfg = load_model_config(write_toml(tmp_path, extra='generate_sql_code = ["b", "a"]\n'))
    assert cfg.nodes["generate_sql_code"] == ("b", "a")
    with pytest.raises(ConfigError, match="ghost"):
        load_model_config(write_toml(tmp_path, extra='generate_sql_code = ["ghost"]\n'))


def test_reasoning_defaults(tmp_path):
    cfg = load_model_config(write_toml(tmp_path))
    assert cfg.settings["coder_reasoning_effort"] == "low"
    assert cfg.settings["escalate_reasoning_on_retry"] is True


def test_repo_models_toml_reasoning_effort_is_a_known_level():
    assert load_model_config().settings["coder_reasoning_effort"] in config.REASONING_LEVELS


def test_invalid_reasoning_effort_is_rejected(tmp_path):
    path = write_toml(tmp_path)
    path.write_text('[settings]\ncoder_reasoning_effort = "extreme"\n' + path.read_text())
    with pytest.raises(ConfigError, match="coder_reasoning_effort"):
        load_model_config(path)


@pytest.mark.parametrize("effort,steps,expected", [
    ("low", 0, "low"), ("low", 1, "medium"), ("low", 2, "high"), ("low", 5, "high"),
    ("minimal", 1, "low"), ("minimal", 2, "medium"), ("high", 1, "high"),
])
def test_escalate_reasoning(effort, steps, expected):
    assert config.escalate_reasoning(effort, steps) == expected


def test_send_reasoning_effort_is_a_model_flag_not_a_provider_param(tmp_path):
    models = (
        '[models.a]\nprovider = "openrouter"\nmodel = "x"\nsend_reasoning_effort = false\n'
        '[models.b]\nprovider = "openrouter"\nmodel = "y"\n'
    )
    cfg = load_model_config(write_toml(tmp_path, models=models, nodes={n: ["a"] for n in config.NODE_NAMES}))
    assert cfg.models["a"].send_reasoning_effort is False and cfg.models["a"].params == {}
    assert cfg.models["b"].send_reasoning_effort is True


def test_send_reasoning_effort_must_be_a_bool(tmp_path):
    models = '[models.a]\nprovider = "openrouter"\nmodel = "x"\nsend_reasoning_effort = "no"\n'
    with pytest.raises(ConfigError, match="send_reasoning_effort"):
        load_model_config(write_toml(tmp_path, models=models))
