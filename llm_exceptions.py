"""Exceptions raised by `LLMClient`.

Connection errors and missing/unpulled-model errors raised by the underlying
`ChatOllama` call are intentionally left unwrapped and propagate as-is, since
those should fail fast without retry. The exceptions below cover the two
cases `LLMClient` itself decides to raise.
"""


class LLMClientError(Exception):
    """Base class for errors raised directly by `LLMClient`."""


class EmptyResponseError(LLMClientError):
    """Raised when the model keeps returning empty/blank text after retries."""


class StructuredOutputValidationError(LLMClientError):
    """Raised when a structured-output call fails schema validation."""
