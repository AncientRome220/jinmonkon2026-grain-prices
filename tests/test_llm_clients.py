from __future__ import annotations

import io
import json

import pytest

from grain_prices import cli
from grain_prices.llm import clients
from grain_prices.llm.canonical_runner import experiment_key


def fake_urlopen(captured: dict, payload: dict):
    def urlopen(req, timeout):
        captured["body"] = json.loads(req.data)
        captured["timeout"] = timeout
        return io.BytesIO(json.dumps(payload).encode("utf-8"))
    return urlopen


def test_anthropic_omits_temperature_and_reads_the_text_block_after_thinking(monkeypatch):
    # #038: current Claude models reject temperature and put a thinking block first.
    captured: dict = {}
    payload = {"stop_reason": "end_turn", "content": [
        {"type": "thinking", "thinking": ""},
        {"type": "text", "text": '{"any_price": false, "observations": []}'},
    ]}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(clients.urllib.request, "urlopen", fake_urlopen(captured, payload))
    client = clients.build_client("anthropic", "claude-fable-5-1", None, timeout=900)

    answer = client.call("system", "user")

    assert answer == {"any_price": False, "observations": []}
    assert "temperature" not in captured["body"]
    assert captured["body"]["max_tokens"] == 16000
    assert captured["timeout"] == 900


def test_anthropic_joins_text_blocks(monkeypatch):
    # A short preamble block before the JSON must not turn a valid answer into an error.
    payload = {"stop_reason": "end_turn", "content": [
        {"type": "thinking", "thinking": ""},
        {"type": "text", "text": "Answer:\n"},
        {"type": "text", "text": '{"any_price": true, "observations": []}'},
    ]}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(clients.urllib.request, "urlopen", fake_urlopen({}, payload))

    answer = clients.build_client("anthropic", "claude-fable-5-1", None).call("system", "user")

    assert answer == {"any_price": True, "observations": []}


@pytest.mark.parametrize("stop_reason",
                         ["refusal", "max_tokens", "model_context_window_exceeded", None])
def test_anthropic_refusal_or_truncation_is_an_error_not_an_answer(monkeypatch, stop_reason):
    payload = {"stop_reason": stop_reason, "content": [{"type": "text", "text": "{}"}]}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(clients.urllib.request, "urlopen", fake_urlopen({}, payload))
    client = clients.build_client("anthropic", "claude-fable-5-1", None)

    with pytest.raises(RuntimeError, match=f"stop_reason={stop_reason}"):
        client.call("system", "user")


@pytest.mark.parametrize("temperature, sent", [(None, False), (0.0, True)])
def test_openai_compatible_sends_temperature_only_when_set(monkeypatch, temperature, sent):
    captured: dict = {}
    payload = {"choices": [{"message": {"content": '{"any_price": false}'}}]}
    monkeypatch.setenv("OLLAMA_API_KEY", "test")
    monkeypatch.setattr(clients.urllib.request, "urlopen", fake_urlopen(captured, payload))
    client = clients.build_client("openai-compatible", "gemma4:26b", temperature,
                                  "http://localhost:11434/v1", "OLLAMA_API_KEY", 1200)

    client.call("system", "user")

    assert ("temperature" in captured["body"]) is sent
    assert captured["timeout"] == 1200


def test_cli_temperature_default_means_unset_and_keeps_its_own_resume_key():
    args = cli.parser().parse_args([
        "score-llm", "--provider", "anthropic", "--model", "m", "--temperature", "default",
    ])
    assert args.temperature is None
    assert cli.parser().parse_args([
        "score-llm", "--provider", "gemini", "--model", "m",
    ]).temperature == 0.0
    # Rows scored at temperature 0 must not be resumed into an unset-temperature run.
    assert experiment_key("w", 0, "p", "m", None) != experiment_key("w", 0, "p", "m", 0.0)


def test_dotenv_provider_keys_win_and_nothing_else_is_loaded(tmp_path, monkeypatch):
    # #038: `.env` keys reach the CLI and beat a stale OS-level key (a user-scope
    # GEMINI_API_KEY was invalid while `.env` held the valid one), but corpus paths
    # there must not outrank local.toml.
    from grain_prices.config import load_provider_keys
    env = tmp_path / ".env"
    env.write_bytes(
        b"# comment\n"
        b"GRAIN_PRICES_DDB_DIR=D:/elsewhere\n"
        b"OPENAI_API_KEY=\"sk-test\" # quoted, then a comment\n"
        b"export ANTHROPIC_API_KEY=ant-test  # project key\n"
        b"GEMINI_API_KEY=from-file\n"
        b"EMPTY_API_KEY=\n"
    )
    # setenv first so that monkeypatch records each variable and undoes what the
    # loader writes; delenv alone records nothing when the variable is absent.
    for name in ("GRAIN_PRICES_DDB_DIR", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.setenv("EMPTY_API_KEY", "kept")
    monkeypatch.setenv("GEMINI_API_KEY", "stale-from-shell")

    loaded = load_provider_keys(env)

    assert loaded == ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    assert clients.os.environ["OPENAI_API_KEY"] == "sk-test"
    assert clients.os.environ["ANTHROPIC_API_KEY"] == "ant-test"
    assert clients.os.environ["GEMINI_API_KEY"] == "from-file"
    assert clients.os.environ["EMPTY_API_KEY"] == "kept"
    assert "GRAIN_PRICES_DDB_DIR" not in clients.os.environ
    assert load_provider_keys(tmp_path / "missing.env") == []


@pytest.mark.parametrize("provider, base_url, has_unique_items", [
    ("openai", "", False),
    ("openai-compatible", "", False),  # no base URL: also api.openai.com
    ("openai-compatible", "http://localhost:11434/v1", True),
    ("openai", "http://localhost:11434/v1", True),
])
def test_openai_strict_schema_drops_unique_items_only_for_openai(monkeypatch, provider, base_url,
                                                                 has_unique_items):
    # #038: api.openai.com strict mode rejects `uniqueItems` (400); Ollama runs keep
    # the schema unchanged so their rows stay comparable. Decided by endpoint host.
    from grain_prices.canonical.contracts import CANONICAL_PREDICTION_SCHEMA
    captured: dict = {}
    payload = {"choices": [{"message": {"content": '{"any_price": false, "observations": []}'}}]}
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(clients.urllib.request, "urlopen", fake_urlopen(captured, payload))
    client = clients.build_client(provider, "m", None, base_url)

    client.call("system", "user", CANONICAL_PREDICTION_SCHEMA)

    sent = json.dumps(captured["body"]["response_format"])
    assert ("uniqueItems" in sent) is has_unique_items
    assert "uniqueItems" in json.dumps(CANONICAL_PREDICTION_SCHEMA)


def raising_urlopen(status: int, body: bytes):
    def urlopen(req, timeout):
        raise clients.urllib.error.HTTPError(req.full_url, status, "Bad Request", {},
                                             io.BytesIO(body))
    return urlopen


def test_anthropic_http_error_carries_the_provider_message_not_the_key(monkeypatch):
    # #040: a spent balance surfaced only as "HTTP Error 400: Bad Request", 1,407 times.
    body = json.dumps({"type": "error", "error": {
        "type": "invalid_request_error",
        "message": "Your credit balance is too low to access the Anthropic API.",
    }}).encode("utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setattr(clients.urllib.request, "urlopen", raising_urlopen(400, body))
    client = clients.build_client("anthropic", "claude-fable-5-1", None)

    with pytest.raises(clients.ProviderError) as caught:
        client.call("system", "user")

    error = caught.value
    assert str(error) == ("anthropic HTTP 400 invalid_request_error: "
                          "Your credit balance is too low to access the Anthropic API.")
    assert error.fatal
    assert "sk-ant-secret" not in str(error)


@pytest.mark.parametrize("status, body, fatal, fragment", [
    (401, b'{"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}',
     True, "authentication_error: invalid x-api-key"),
    (429, b'{"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota",'
          b' "code": "insufficient_quota"}}', True, "insufficient_quota: You exceeded"),
    (429, b'{"error": {"message": "Rate limit reached", "type": "requests", "code": "rate_limit_exceeded"}}',
     False, "rate_limit_exceeded: Rate limit reached"),
    (404, b'{"error": "model \\"qwen\\" not found"}', False, 'HTTP 404: model "qwen" not found'),
    (500, b"<html>upstream error</html>", False, "HTTP 500: <html>upstream error</html>"),
])
def test_openai_compatible_http_errors_are_classified(monkeypatch, status, body, fatal, fragment):
    # OpenAI nests {"error": {...}}, Ollama sends {"error": "..."}, a proxy may send HTML.
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(clients.urllib.request, "urlopen", raising_urlopen(status, body))
    client = clients.build_client("openai-compatible", "m", 0.0, "http://localhost:11434/v1")

    with pytest.raises(clients.ProviderError) as caught:
        client.call("system", "user")

    assert caught.value.fatal is fatal
    assert fragment in str(caught.value)
