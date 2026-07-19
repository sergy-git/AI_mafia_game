"""Synchronous LLM client layer on top of `langchain_ollama.ChatOllama`.

`LLMClient` is the single interface every LangGraph node (GM or player agent)
calls into. It resolves per-agent model configuration via `ClientConfig`,
caches one `ChatOllama` instance per resolved profile for the process
lifetime, and returns a `ClientResponse` with text, optional structured
output, latency, token counts, and the model actually used.

Calls are strictly synchronous and sequential (no async/concurrency).
Streaming is out of scope — calls return the full completion only.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from client_config import ClientConfig
from llm_exceptions import EmptyResponseError, StructuredOutputValidationError

# Retry policy: applies only to empty/blank model responses. Connection
# errors and missing/unpulled-model errors are never retried — they
# propagate immediately from the underlying `ChatOllama` call.
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0


@dataclass
class ClientResponse:
    """Result of an `LLMClient.invoke()` call."""

    text: str
    model: str
    latency_seconds: float
    structured: Optional[Any] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMClient:
    """Single, stable interface for calling per-agent LLMs via Ollama."""

    def __init__(self, config: Optional[ClientConfig] = None):
        self._config = config or ClientConfig()
        self._llm_cache: Dict[str, ChatOllama] = {}

    def invoke(
        self,
        agent_id: str,
        messages: List[BaseMessage],
        structured_schema: Optional[Type[BaseModel]] = None,
    ) -> ClientResponse:
        """Send `messages` to the model resolved for `agent_id`.

        Returns plain text by default. If `structured_schema` (a pydantic
        model) is given, the response is additionally parsed/validated
        against it and returned via `ClientResponse.structured`; `text`
        always holds the raw model output regardless of mode.

        Raises `EmptyResponseError` if the model returns only empty/blank
        text after retries, and `StructuredOutputValidationError`
        immediately (no retry) if structured output fails validation.
        Connection errors and missing-model errors from the underlying
        client propagate unmodified and are never retried.
        """
        profile = self._config.resolve(agent_id)
        model_name = profile.get("model")
        llm = self._get_llm(profile)

        last_text = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            start = time.monotonic()
            if structured_schema is not None:
                text, structured, usage = self._invoke_structured(
                    llm, messages, structured_schema
                )
            else:
                text, usage = self._invoke_plain(llm, messages)
                structured = None
            latency = time.monotonic() - start

            if text and text.strip():
                return ClientResponse(
                    text=text,
                    structured=structured,
                    latency_seconds=latency,
                    model=model_name,
                    prompt_tokens=usage.get("input_tokens") if usage else None,
                    completion_tokens=usage.get("output_tokens") if usage else None,
                    total_tokens=usage.get("total_tokens") if usage else None,
                )

            last_text = text
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)

        raise EmptyResponseError(
            f"Model '{model_name}' for agent '{agent_id}' returned an empty "
            f"response after {MAX_ATTEMPTS} attempts (last text: {last_text!r})."
        )

    def _get_llm(self, profile: Dict[str, Any]) -> ChatOllama:
        key = self._profile_key(profile)
        if key not in self._llm_cache:
            params = dict(profile)
            model = params.pop("model")
            self._llm_cache[key] = ChatOllama(model=model, **params)
        return self._llm_cache[key]

    @staticmethod
    def _profile_key(profile: Dict[str, Any]) -> str:
        return json.dumps(profile, sort_keys=True)

    @staticmethod
    def _invoke_plain(llm: ChatOllama, messages: List[BaseMessage]):
        response = llm.invoke(messages)
        content = response.content
        text = content if isinstance(content, str) else str(content)
        usage = getattr(response, "usage_metadata", None)
        return text, usage

    @staticmethod
    def _invoke_structured(
        llm: ChatOllama, messages: List[BaseMessage], schema: Type[BaseModel]
    ):
        structured_llm = llm.with_structured_output(schema, include_raw=True)
        result = structured_llm.invoke(messages)
        raw = result.get("raw")
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")

        content = raw.content if raw is not None else ""
        text = content if isinstance(content, str) else str(content)
        usage = getattr(raw, "usage_metadata", None) if raw is not None else None

        if parsing_error is not None or parsed is None:
            raise StructuredOutputValidationError(
                f"Structured output validation failed for schema "
                f"'{schema.__name__}': {parsing_error}"
            )

        return text, parsed, usage
