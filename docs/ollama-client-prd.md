# PRD: Ollama Client Layer

## Problem Statement

The Mafia simulation will run many LLM calls per turn against locally hosted models via Ollama, from both the GM and every player agent. Today there is only `ollama_client.py`, a throwaway script written to confirm Ollama works at all on this machine — it hand-rolls HTTP/CLI calls, has no per-agent configuration, no resilience, no structured output, and no built-in timing/token metadata. It also doesn't integrate with LangChain/LangGraph, which the project's chosen stack depends on.

Before any game logic, GM, or LangGraph graph can be built, there needs to be a real, reusable client module that every node in the graph can call into with a stable, tested interface.

## Solution

Build a client layer, `LLMClient`, on top of `langchain_ollama.ChatOllama` so it's natively compatible with LangGraph nodes. It resolves per-agent model configuration from a config loader, `ClientConfig`. Every call is synchronous and non-streaming, and returns a `ClientResponse` containing response text (plus a validated structured payload, if requested) alongside latency, token counts, and the model used. Empty/blank responses (a known intermittent model quirk) are retried a bounded number of times; connection errors and missing/unpulled models fail fast.

## User Stories

1. As a LangGraph node author, I want a single client interface regardless of which agent or the GM is calling it, so that I don't special-case model selection in graph/node code.
2. As a developer configuring the game, I want to give individual agents (including the GM) different models, so that I can balance capability against local resource constraints.
3. As a developer configuring the game, I want agents without an explicit override to fall back to a default model, so that I don't have to configure every agent individually.
4. As a developer building graph nodes, I want to send a list of messages and get plain text back, so that I can append it to public or private history without the client needing to know game semantics.
5. As a future GM implementer, I want to optionally request a schema-constrained response instead of parsing free text, so that extracting structured data (like a vote) is reliable.
6. As a developer, I want every response to include latency, token counts, and the model used, so that I don't have to hand-instrument timing as `test_run.py` does today.
7. As a developer, I want empty/blank model responses automatically retried a bounded number of times, so that a known intermittent model quirk doesn't silently corrupt a turn.
8. As a developer, I want connection errors and missing/unpulled models to fail immediately without retries, so that hard misconfiguration surfaces right away instead of being masked.
9. As a developer, I want calls to be strictly synchronous and sequential, so that the client matches turn-based gameplay without concurrency complexity.
10. As a developer, I want to run the full test suite without a real Ollama server, so that tests are fast and CI-friendly.
11. As a developer, I want a separate manual script for exercising the client against a real local server, so that I can validate real-world behavior outside the automated suite.
12. As a developer, I want config profile resolution testable in isolation from any network/model calls, so that config bugs are caught without Ollama running.

## Implementation Decisions

- **Scope boundary:** This PRD covers only the LLM-calling client layer. The `MafiaState` graph schema, GM logic, node wiring (`think_node`/`speak_node`), and the `transcript.md` exporter described in [idea.md](idea.md) are out of scope and will be covered by future PRDs.
- **Foundation:** Built on `langchain_ollama.ChatOllama` rather than raw HTTP/CLI calls. `ollama_client.py` is a throwaway prototype and is not carried forward as a design reference.
- **`LLMClient` module:** Exposes a single synchronous call: `invoke(agent_id, messages, structured_schema=None)`. Internally it resolves the config profile for `agent_id` via `ClientConfig`, and caches/reuses one `ChatOllama` instance per resolved profile key for the process lifetime (avoiding recreating the object every call — independent of whether Ollama itself keeps that model loaded in memory). It invokes non-streaming; if `structured_schema` is given it requests schema-constrained output and validates the result, otherwise it returns plain text. The retry policy below wraps the call, and latency/token counts are captured into a `ClientResponse`.
- **Config resolution:** No separate role tier. `ClientConfig` resolves purely by `agent_id`: an explicit entry in `agents[agent_id]` overrides `default`; the GM is just another `agent_id` key. Config lives in a renamed file, `llm_config.json` (replacing the flat `model_config.json`), same JSON format, with a `default` profile plus an `agents` map of per-agent overrides (model tag, temperature, and other generation parameters such as `num_predict`).
- **`ClientResponse` shape:** `text` (`str`, always populated with the raw model output, even in structured mode), `structured` (populated only when a schema was requested), `latency_seconds`, token counts (when available from the underlying response), and `model` (the resolved model name actually used).
- **Concurrency:** Strictly synchronous and sequential — one call at a time. No async/concurrent invocation now.
- **Streaming:** Out of scope. Calls return the full completion only once generation finishes.
- **Structured output:** Opt-in per call via `structured_schema`, not the default. On schema-validation failure, the client raises a distinct exception (e.g. `StructuredOutputValidationError`) immediately — not retried, since it more likely reflects a prompting/schema mismatch than transient flakiness.
- **Retry policy:** Applies only to empty/blank model responses. Exactly 2 retries (3 attempts total) with a fixed 1-second delay between attempts. Connection errors and missing/unpulled-model errors are never retried; model availability is not checked eagerly at construction — it's only discovered when a call actually fails.
- **Metadata capture is mandatory:** every `LLMClient` call returns latency/token/model metadata as part of its normal return value, with no opt-out.

## Testing Decisions

- Good tests here assert observable behavior (given these config profiles and this mocked model response, the client returns this `ClientResponse`/raises this exception) rather than internal call sequencing.
- `LLMClient` gets automated unit tests with the underlying `ChatOllama` invocation mocked out. These cover: correct profile resolution being applied to the underlying model call, retry-and-succeed on an empty response, exhausting retries and raising on repeated empty responses, fail-fast (no retry) on connection errors and missing-model errors, structured-output success and immediate-raise-on-validation-failure paths, and that returned `ClientResponse` metadata (latency, tokens, model name) is populated.
- `ClientConfig` gets automated unit tests with no network/model mocking needed: profile loading from file, explicit per-agent override resolution, and fallback to `default` when no override exists.
- Real-server behavior (does Ollama actually respond correctly, is the configured model pulled, real-world latency) is intentionally left to a separate, clearly-labeled manual script in the spirit of today's `test_run.py`, and is not part of the automated test suite.

## Out of Scope

- `MafiaState` graph schema and any LangGraph graph/node wiring.
- GM logic: rule enforcement, phase transitions, vote counting, kill resolution.
- The `transcript.md` exporter.
- Async/concurrent call support.
- Streaming responses.
- Multi-machine or distributed inference; cloud LLM fallback.
- Migrating/deleting the existing `ollama_client.py` prototype (it may be removed or repurposed in a later, separate change).

## Further Notes

- The existing repo notes (captured from prior debugging) record that this model occasionally returns an empty response with an immediate stop token on raw completion-style calls, and that using a chat-style call format resolves most of that. This informs the retry policy above but should be re-validated once the client sits on `ChatOllama` rather than raw HTTP.
- Per-agent model configuration is a deliberate choice made with awareness that Ollama can only keep a limited number of models resident in memory at once, so switching models between calls may introduce reload latency. This tradeoff was accepted explicitly in favor of configuration flexibility; if reload latency becomes a practical problem, a future revision may need a smarter model-residency strategy.
