"""Unit tests for `LLMClient`.

The underlying `ChatOllama` invocation is mocked out entirely — no network
or real model calls happen here. `time.sleep` is also patched so retry
tests run instantly.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from client_config import ClientConfig
from llm_client import MAX_ATTEMPTS, RETRY_DELAY_SECONDS, LLMClient
from llm_exceptions import EmptyResponseError, StructuredOutputValidationError


class _VoteSchema(BaseModel):
    target: str


def _fake_message(content, usage_metadata=None):
    return SimpleNamespace(content=content, usage_metadata=usage_metadata)


@pytest.fixture
def stub_config():
    config = MagicMock(spec=ClientConfig)
    config.resolve.return_value = {
        "model": "test-model",
        "temperature": 0.5,
        "num_predict": 256,
    }
    return config


@pytest.fixture(autouse=True)
def no_sleep():
    with patch("llm_client.time.sleep") as sleep_mock:
        yield sleep_mock


@pytest.fixture
def mock_chat_ollama():
    with patch("llm_client.ChatOllama") as chat_ollama_cls:
        instance = MagicMock()
        chat_ollama_cls.return_value = instance
        yield chat_ollama_cls, instance


def test_profile_resolution_applied_to_underlying_model_call(stub_config, mock_chat_ollama):
    chat_ollama_cls, instance = mock_chat_ollama
    instance.invoke.return_value = _fake_message("hello")

    client = LLMClient(config=stub_config)
    client.invoke("gm", messages=[("user", "hi")])

    stub_config.resolve.assert_called_once_with("gm")
    chat_ollama_cls.assert_called_once_with(
        model="test-model", temperature=0.5, num_predict=256
    )


def test_retry_and_succeed_on_empty_response(stub_config, mock_chat_ollama, no_sleep):
    _, instance = mock_chat_ollama
    instance.invoke.side_effect = [_fake_message(""), _fake_message("real answer")]

    client = LLMClient(config=stub_config)
    response = client.invoke("gm", messages=[("user", "hi")])

    assert response.text == "real answer"
    assert instance.invoke.call_count == 2
    no_sleep.assert_called_once_with(RETRY_DELAY_SECONDS)


def test_exhausts_retries_and_raises_on_repeated_empty_responses(
    stub_config, mock_chat_ollama, no_sleep
):
    _, instance = mock_chat_ollama
    instance.invoke.return_value = _fake_message("   ")

    client = LLMClient(config=stub_config)

    with pytest.raises(EmptyResponseError):
        client.invoke("gm", messages=[("user", "hi")])

    assert instance.invoke.call_count == MAX_ATTEMPTS
    assert no_sleep.call_count == MAX_ATTEMPTS - 1


def test_connection_error_fails_fast_without_retry(stub_config, mock_chat_ollama, no_sleep):
    _, instance = mock_chat_ollama
    instance.invoke.side_effect = ConnectionError("cannot reach ollama server")

    client = LLMClient(config=stub_config)

    with pytest.raises(ConnectionError):
        client.invoke("gm", messages=[("user", "hi")])

    assert instance.invoke.call_count == 1
    no_sleep.assert_not_called()


def test_missing_model_error_fails_fast_without_retry(stub_config, mock_chat_ollama, no_sleep):
    class ModelNotFoundError(Exception):
        pass

    _, instance = mock_chat_ollama
    instance.invoke.side_effect = ModelNotFoundError("model 'x' not found")

    client = LLMClient(config=stub_config)

    with pytest.raises(ModelNotFoundError):
        client.invoke("gm", messages=[("user", "hi")])

    assert instance.invoke.call_count == 1
    no_sleep.assert_not_called()


def test_structured_output_success(stub_config, mock_chat_ollama):
    _, instance = mock_chat_ollama
    parsed = _VoteSchema(target="player_2")
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = {
        "raw": _fake_message('{"target": "player_2"}'),
        "parsed": parsed,
        "parsing_error": None,
    }
    instance.with_structured_output.return_value = structured_llm

    client = LLMClient(config=stub_config)
    response = client.invoke("gm", messages=[("user", "vote")], structured_schema=_VoteSchema)

    instance.with_structured_output.assert_called_once_with(_VoteSchema, include_raw=True)
    assert response.structured == parsed
    assert response.text == '{"target": "player_2"}'


def test_structured_output_validation_failure_raises_immediately(
    stub_config, mock_chat_ollama, no_sleep
):
    _, instance = mock_chat_ollama
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = {
        "raw": _fake_message("not valid json"),
        "parsed": None,
        "parsing_error": ValueError("bad schema match"),
    }
    instance.with_structured_output.return_value = structured_llm

    client = LLMClient(config=stub_config)

    with pytest.raises(StructuredOutputValidationError):
        client.invoke("gm", messages=[("user", "vote")], structured_schema=_VoteSchema)

    assert structured_llm.invoke.call_count == 1
    no_sleep.assert_not_called()


def test_response_metadata_populated(stub_config, mock_chat_ollama):
    _, instance = mock_chat_ollama
    instance.invoke.return_value = _fake_message(
        "an answer",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )

    client = LLMClient(config=stub_config)
    response = client.invoke("gm", messages=[("user", "hi")])

    assert response.model == "test-model"
    assert isinstance(response.latency_seconds, float)
    assert response.latency_seconds >= 0
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.total_tokens == 15


def test_chat_ollama_instance_cached_per_profile(stub_config, mock_chat_ollama):
    chat_ollama_cls, instance = mock_chat_ollama
    instance.invoke.return_value = _fake_message("hello")

    client = LLMClient(config=stub_config)
    client.invoke("gm", messages=[("user", "hi")])
    client.invoke("gm", messages=[("user", "hi again")])

    chat_ollama_cls.assert_called_once()


def test_chat_ollama_cache_keyed_by_resolved_profile_not_agent_id(
    stub_config, mock_chat_ollama
):
    chat_ollama_cls, instance = mock_chat_ollama
    instance.invoke.return_value = _fake_message("hello")

    profiles = {
        "gm": {"model": "test-model", "temperature": 0.5, "num_predict": 256},
        "player_1": {"model": "test-model", "temperature": 0.5, "num_predict": 256},
        "player_2": {"model": "other-model", "temperature": 0.9, "num_predict": 128},
    }
    stub_config.resolve.side_effect = lambda agent_id: profiles[agent_id]

    client = LLMClient(config=stub_config)
    client.invoke("gm", messages=[("user", "hi")])
    client.invoke("player_1", messages=[("user", "hi")])
    client.invoke("player_2", messages=[("user", "hi")])

    # "gm" and "player_1" resolve to an identical profile, so they must share
    # one cached ChatOllama instance; "player_2" resolves to a distinct
    # profile and must get its own separate instance (2 total, not 3).
    assert chat_ollama_cls.call_count == 2
