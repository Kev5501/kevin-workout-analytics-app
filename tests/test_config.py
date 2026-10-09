import pytest

from workout_chat import config
from workout_chat.config import ConfigError, load_model_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (*config.REQUIRED_SECRETS, *config.OPTIONAL_SECRETS):
        monkeypatch.delenv(name, raising=False)


def test_default_models_toml_is_valid():
    cfg = load_model_config()
    assert set(cfg.nodes) == set(config.NODE_NAMES)
    assert cfg.nodes["router"][0] == "jev"
    assert cfg.settings["router_confidence_threshold"] == 0.5


def test_chain_skips_models_without_secrets(monkeypatch):
    cfg = load_model_config()
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert [s.key for s in cfg.chain("router")] == ["gemini_flash"]  # no Jev key -> falls back
    monkeypatch.setenv("TYPESAFE_API_KEY", "y")
    assert [s.key for s in cfg.chain("router")] == ["jev", "gemini_flash"]


def test_chain_with_no_usable_model_names_the_secrets():
    cfg = load_model_config()
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        cfg.chain("synthesizer")


def _write(tmp_path, text):
    p = tmp_path / "models.toml"
    p.write_text(text)
    return p


GOOD_NODES = 'router=["m"]\nrewriter=["m"]\ncoder=["m"]\nsynthesizer=["m"]\n'


def test_undefined_model_reference_rejected(tmp_path):
    p = _write(tmp_path, '[models.m]\nprovider="google_genai"\nmodel="x"\n[nodes]\n' + GOOD_NODES.replace('coder=["m"]', 'coder=["nope"]'))
    with pytest.raises(ConfigError, match="nope"):
        load_model_config(p)


def test_unknown_provider_rejected(tmp_path):
    p = _write(tmp_path, '[models.m]\nprovider="mystery"\nmodel="x"\n[nodes]\n' + GOOD_NODES)
    with pytest.raises(ConfigError, match="mystery"):
        load_model_config(p)


def test_missing_node_rejected(tmp_path):
    p = _write(tmp_path, '[models.m]\nprovider="google_genai"\nmodel="x"\n[nodes]\nrouter=["m"]\n')
    with pytest.raises(ConfigError, match="rewriter"):
        load_model_config(p)


def test_extra_model_keys_become_params(tmp_path):
    p = _write(tmp_path, '[models.m]\nprovider="openrouter"\nmodel="x"\nmax_tokens=512\n[nodes]\n' + GOOD_NODES)
    assert load_model_config(p).models["m"].params == {"max_tokens": 512}
