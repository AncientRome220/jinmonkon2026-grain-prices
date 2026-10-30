from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Protocol


class LLMClient(Protocol):
    def call(self, system: str, user: str, schema: dict | None = None) -> dict: ...


# A 400 or 429 with one of these in the message fails for every request until the
# account is fixed, like 401/403 (#040: a spent Anthropic balance returns 400).
_ACCOUNT_MESSAGE = re.compile(r"credit balance|billing|insufficient_quota|exceeded your current quota",
                              re.IGNORECASE)
_BODY_LIMIT = 4096


class ProviderError(RuntimeError):
    """An HTTP error carrying the provider's own explanation, not just the status line."""

    def __init__(self, provider: str, status: int, error_type: str, message: str):
        self.provider, self.status = provider, status
        self.error_type, self.message = error_type, message
        head = " ".join(part for part in (provider, f"HTTP {status}", error_type) if part)
        super().__init__(f"{head}: {message}" if message else head)

    @property
    def fatal(self) -> bool:
        """True when retrying the next window cannot succeed either."""
        if self.status in {401, 402, 403}:
            return True
        return self.status in {400, 429} and bool(
            _ACCOUNT_MESSAGE.search(f"{self.error_type} {self.message}")
        )


def _provider_error(provider: str, exc: urllib.error.HTTPError) -> ProviderError:
    # Only the body is read; the request headers hold the key and are never quoted.
    try:
        raw = exc.read(_BODY_LIMIT).decode("utf-8", "replace")
    except Exception:
        raw = ""
    error_type, message = "", raw.strip() or str(exc.reason or "")
    try:
        detail = json.loads(raw).get("error")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, dict):
        # Anthropic: {"error": {"type", "message"}}; OpenAI: {"error": {"type", "code", "message"}}
        error_type = str(detail.get("code") or detail.get("type") or "")
        message = str(detail.get("message") or message)
    elif isinstance(detail, str):  # Ollama: {"error": "..."}
        message = detail
    return ProviderError(provider, exc.code, error_type, message)


def _post_json(provider: str, req: urllib.request.Request, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise _provider_error(provider, exc) from None


def _loads_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}") + 1
    value = json.loads(text[start:end] if start >= 0 and end > start else text)
    if not isinstance(value, dict):
        raise ValueError("provider response is not a JSON object")
    return value


def _gemini_response_json_schema(value):
    """Return the supported JSON Schema subset without weakening local validation."""
    if isinstance(value, list):
        return [_gemini_response_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    converted = {
        key: _gemini_response_json_schema(item)
        for key, item in value.items()
        if key != "uniqueItems"
    }
    schema_type = converted.get("type")
    if isinstance(schema_type, list):
        converted.pop("type")
        converted["anyOf"] = [{"type": item} for item in schema_type]
    return converted


def _without_unique_items(value):
    """OpenAI strict mode rejects ``uniqueItems``; local validation still enforces it."""
    if isinstance(value, list):
        return [_without_unique_items(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {key: _without_unique_items(item) for key, item in value.items() if key != "uniqueItems"}


@dataclass
class GeminiClient:
    model: str
    temperature: float | None = 0.0
    max_output_tokens: int | None = None
    thinking_level: str | None = None

    def __post_init__(self):
        if self.thinking_level is not None:
            if self.thinking_level not in {"low", "medium", "high"}:
                raise ValueError("thinking level must be low, medium or high")
            if self.temperature is not None:
                raise ValueError("Gemini thinking-level runs require unset temperature")
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required")
        if self.max_output_tokens is not None:
            from google.genai import types
            self.client = genai.Client(api_key=key, http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)))
        else:
            self.client = genai.Client(api_key=key)

    def call(self, system: str, user: str, schema: dict | None = None) -> dict:
        from google.genai import types
        thinking = ({"thinking_config": types.ThinkingConfig(thinking_level=self.thinking_level)}
                    if self.thinking_level is not None else
                    {"thinking_config": types.ThinkingConfig(thinking_budget=8192)}
                    if self.max_output_tokens is not None else {})
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            **({} if self.max_output_tokens is None else {"max_output_tokens": self.max_output_tokens}),
            **thinking,
            response_json_schema=(
                _gemini_response_json_schema(schema) if schema else None
            ),
            # None leaves sampling at the provider default (#038).
            **({} if self.temperature is None else {"temperature": self.temperature}),
        )
        response = self.client.models.generate_content(
            model=self.model, contents=user, config=config,
        )
        return _loads_object(response.text or "")


@dataclass
class OpenAICompatibleClient:
    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = 0.0
    timeout: float = 180.0
    # Only for api.openai.com: other endpoints (Ollama) keep receiving the schema
    # byte-for-byte as before, so runs already scored there stay comparable (#038).
    # The Anthropic client sends no schema at all (prompt only); the Claude runs of
    # 2026-10-18 were scored that way, so it is recorded in #038 rather than changed.
    drop_unique_items: bool = False
    max_output_tokens: int | None = None

    def call(self, system: str, user: str, schema: dict | None = None) -> dict:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is required")
        if schema and self.drop_unique_items:
            schema = _without_unique_items(schema)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": (
                {"type": "json_schema", "json_schema": {"name": "canonical_prediction", "strict": True, "schema": schema}}
                if schema else {"type": "json_object"}
            ),
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            body["max_completion_tokens"] = self.max_output_tokens
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        payload = _post_json("openai-compatible", req, self.timeout)
        return _loads_object(payload["choices"][0]["message"]["content"])


@dataclass
class AnthropicClient:
    model: str
    temperature: float | None = 0.0
    timeout: float = 180.0
    # Room for thinking, which current models cannot switch off and which counts
    # against max_tokens; 1,800 cut answers off mid-JSON (#038).
    max_tokens: int = 16000

    def call(self, system: str, user: str, schema: dict | None = None) -> dict:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required")
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        payload = _post_json("anthropic", req, self.timeout)
        # No server-side refusal fallback: a different model answering would
        # silently mix two models into one comparison row (#038).
        # Anything but a normal stop (refusal, max_tokens, context window exceeded, ...)
        # is a failed window rather than a possibly truncated answer.
        stop_reason = payload.get("stop_reason")
        if stop_reason not in {"end_turn", "stop_sequence"}:
            raise RuntimeError(f"anthropic stop_reason={stop_reason}")
        # Thinking blocks come first on current models; the answer is in the text blocks.
        text = "".join(block["text"] for block in payload.get("content", [])
                       if block.get("type") == "text")
        return _loads_object(text)


def build_client(provider: str, model: str, temperature: float | None, base_url: str = "",
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 180.0,
                 max_output_tokens: int | None = None, thinking_level: str | None = None) -> LLMClient:
    if provider == "gemini":
        return GeminiClient(model, temperature, max_output_tokens, thinking_level)
    if thinking_level is not None:
        raise ValueError("thinking_level is supported only for Gemini")
    if provider == "anthropic":
        return AnthropicClient(model, temperature, timeout, max_tokens=max_output_tokens or 16000)
    if provider in {"openai", "openai-compatible"}:
        url = base_url or "https://api.openai.com/v1"
        # Decided by the endpoint, not the provider name: both names reach
        # api.openai.com when no base URL is given.
        return OpenAICompatibleClient(model, url, api_key_env, temperature, timeout,
                                      drop_unique_items=urlparse(url).hostname == "api.openai.com",
                                      max_output_tokens=max_output_tokens)
    raise ValueError(f"unsupported provider: {provider}")
