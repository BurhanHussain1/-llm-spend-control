"""Every provider returns the same Completion shape, and fails the same way.

The real adapters are exercised against mocked HTTP responses, so this suite
never touches the network and needs no API keys.
"""

import httpx
import pytest
import respx

from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import Completion, ProviderError, split_system_prompt
from app.providers.mock import DEGENERATE_MARKER, MockProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider
from app.registry import ModelRegistry
from app.schemas import Message

PROMPT = [
    Message(role="system", content="You are terse."),
    Message(role="user", content="Summarize the Q3 report."),
]


@pytest.fixture
def registry():
    return ModelRegistry.load()


# --- shared contract ---------------------------------------------------------


def test_split_system_prompt_separates_system_turns():
    system, conversation = split_system_prompt(PROMPT)

    assert system == "You are terse."
    assert [message.role for message in conversation] == ["user"]


def test_split_system_prompt_handles_no_system_turn():
    system, conversation = split_system_prompt([Message(role="user", content="hi")])

    assert system is None
    assert len(conversation) == 1


# --- mock provider -----------------------------------------------------------


async def test_mock_provider_is_deterministic(registry):
    provider = MockProvider(registry)

    first = await provider.complete("mock-cheap", PROMPT, max_output_tokens=512)
    second = await provider.complete("mock-cheap", PROMPT, max_output_tokens=512)

    assert first == second


async def test_mock_provider_returns_realistic_counts_and_latency(registry):
    provider = MockProvider(registry)

    result = await provider.complete("mock-mid", PROMPT, max_output_tokens=512)

    assert isinstance(result, Completion)
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    # Within the +/-20% band of mock-mid's 2200ms typical latency.
    assert 1760 <= result.latency_ms <= 2640


async def test_stronger_mock_tiers_produce_longer_answers(registry):
    """The quality judge needs cheap and strong answers to actually differ."""
    provider = MockProvider(registry)

    cheap = await provider.complete("mock-cheap", PROMPT, max_output_tokens=4096)
    strong = await provider.complete("mock-strong", PROMPT, max_output_tokens=4096)

    assert strong.output_tokens > cheap.output_tokens


async def test_mock_provider_respects_the_output_cap(registry):
    provider = MockProvider(registry)

    result = await provider.complete("mock-strong", PROMPT, max_output_tokens=5)

    assert result.output_tokens == 5


async def test_mock_provider_can_produce_degenerate_output(registry):
    """Exercises the empty-answer path that triggers escalation in Phase 6."""
    provider = MockProvider(registry)
    prompt = [Message(role="user", content=f"{DEGENERATE_MARKER} summarize this")]

    result = await provider.complete("mock-cheap", prompt, max_output_tokens=512)

    assert result.text == ""


async def test_mock_provider_rejects_non_mock_models(registry):
    provider = MockProvider(registry)

    with pytest.raises(ProviderError, match="not a mock model"):
        await provider.complete("claude-opus-5", PROMPT, max_output_tokens=512)


# --- Anthropic ---------------------------------------------------------------

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def anthropic_body(text: str = "Revenue rose 4%.", stop_reason: str = "end_turn"):
    return {
        "id": "msg_01abc",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 42, "output_tokens": 7},
    }


@respx.mock
async def test_anthropic_normalizes_a_response():
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=anthropic_body())
    )
    provider = AnthropicProvider(api_key="test-key")

    result = await provider.complete("claude-haiku-4-5", PROMPT, max_output_tokens=512)

    assert result.text == "Revenue rose 4%."
    assert (result.input_tokens, result.output_tokens) == (42, 7)
    assert result.latency_ms >= 0
    assert result.provider_metadata["stop_reason"] == "end_turn"


@respx.mock
async def test_anthropic_sends_the_system_prompt_out_of_band():
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=anthropic_body())
    )
    provider = AnthropicProvider(api_key="test-key")

    await provider.complete("claude-haiku-4-5", PROMPT, max_output_tokens=512)

    sent = route.calls.last.request
    import json

    payload = json.loads(sent.content)
    assert payload["system"] == "You are terse."
    assert [m["role"] for m in payload["messages"]] == ["user"]


@respx.mock
async def test_anthropic_refusal_yields_empty_text_not_a_crash():
    """A refusal is a successful HTTP call with no usable content."""
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200, json=anthropic_body(text="", stop_reason="refusal")
        )
    )
    provider = AnthropicProvider(api_key="test-key")

    result = await provider.complete("claude-haiku-4-5", PROMPT, max_output_tokens=512)

    assert result.text == ""
    assert result.provider_metadata["refused"] is True


@respx.mock
async def test_anthropic_errors_become_provider_errors():
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
    )
    provider = AnthropicProvider(api_key="test-key")

    with pytest.raises(ProviderError, match="anthropic"):
        await provider.complete("claude-haiku-4-5", PROMPT, max_output_tokens=512)


# --- OpenAI ------------------------------------------------------------------

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

OPENAI_BODY = {
    "id": "chatcmpl-01abc",
    "object": "chat.completion",
    "created": 1770000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Revenue rose 4%."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 40, "completion_tokens": 6, "total_tokens": 46},
}


@respx.mock
async def test_openai_normalizes_a_response():
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=OPENAI_BODY))
    provider = OpenAIProvider(api_key="test-key")

    result = await provider.complete("gpt-4o-mini", PROMPT, max_output_tokens=512)

    assert result.text == "Revenue rose 4%."
    assert (result.input_tokens, result.output_tokens) == (40, 6)
    assert result.provider_metadata["finish_reason"] == "stop"


@respx.mock
async def test_openai_errors_become_provider_errors():
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
    )
    provider = OpenAIProvider(api_key="test-key")

    with pytest.raises(ProviderError, match="openai"):
        await provider.complete("gpt-4o-mini", PROMPT, max_output_tokens=512)


# --- Ollama ------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"

OLLAMA_BODY = {
    "model": "llama3.1:8b",
    "message": {"role": "assistant", "content": "Revenue rose 4%."},
    "done_reason": "stop",
    "prompt_eval_count": 38,
    "eval_count": 6,
}


@respx.mock
async def test_ollama_normalizes_a_response():
    respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=OLLAMA_BODY))
    provider = OllamaProvider()

    result = await provider.complete("llama3.1:8b", PROMPT, max_output_tokens=512)

    assert result.text == "Revenue rose 4%."
    assert (result.input_tokens, result.output_tokens) == (38, 6)


@respx.mock
async def test_ollama_missing_counts_default_to_zero():
    """Ollama omits token counts on a cache hit; that must not raise."""
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            200, json={"message": {"content": "cached"}, "model": "llama3.1:8b"}
        )
    )
    provider = OllamaProvider()

    result = await provider.complete("llama3.1:8b", PROMPT, max_output_tokens=512)

    assert (result.input_tokens, result.output_tokens) == (0, 0)


@respx.mock
async def test_ollama_connection_failure_says_the_daemon_may_be_down():
    respx.post(OLLAMA_URL).mock(side_effect=httpx.ConnectError("refused"))
    provider = OllamaProvider()

    with pytest.raises(ProviderError, match="ollama serve"):
        await provider.complete("llama3.1:8b", PROMPT, max_output_tokens=512)


@respx.mock
async def test_ollama_http_errors_become_provider_errors():
    respx.post(OLLAMA_URL).mock(return_value=httpx.Response(500, text="boom"))
    provider = OllamaProvider()

    with pytest.raises(ProviderError, match="ollama"):
        await provider.complete("llama3.1:8b", PROMPT, max_output_tokens=512)
