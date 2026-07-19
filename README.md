# AI Mafia: Multi-Agent LLM Simulation

> *Who's lying? Who's telling the truth? Does it even matter if you can't tell the difference?*

A closed room. A circle of LLM agents. One by one, they speak — accuse, defend, deceive.

None of them are human. All of them are trying to survive.

---

## Overview

**AI Mafia** is a multi-agent simulation of the party game Mafia (Werewolf) where every player is an LLM agent orchestrated by an LLM-driven Game Master. The project serves as an architectural stress-test for building complex, state-heavy, multi-agent frameworks using LangGraph with local LLMs via Ollama.

This project runs fully on-premise. No cloud. No oversight. Just agents doing what they do best.

## Current Status

**Phase:** LLMClient layer complete; LangGraph orchestration in progress.

### ✅ What Works
- **`LLMClient` layer** — Built on `langchain_ollama.ChatOllama`, fully implements [PRD](docs/ollama-client-prd.md):
  - Synchronous, sequential API: `invoke(agent_id, messages, structured_schema=None)`
  - Per-agent config resolution with fallback to defaults
  - Structured output (Pydantic schema) support with immediate validation
  - Empty-response retry logic (3 attempts, 1-second delay)
  - Fast-fail on connection/missing-model errors (no retries)
  - Full response metadata: latency, token counts, resolved model name
- **Automated test suite** — pytest with unit tests for config resolution and LLM behavior (mocked backend)
- **Manual real-server check** — Script to validate behavior against live Ollama instance

### 🚧 In Development
- LangGraph-based game state machine and orchestration
- Agent `think_node` and `speak_node` dual-path execution
- GM rule enforcement, phase transitions, vote counting, kill resolution
- Structured markdown transcript exporter
- LangGraph graph visualization and debugging

## Project Structure

- **docs/**
  - [game-rules.md](docs/game-rules.md) — Complete Mafia game rules (7 players: 5 Citizens, 2 Mafia)
  - [idea.md](docs/idea.md) — Conceptual architecture and coding requirements
  - [ollama-client-prd.md](docs/ollama-client-prd.md) — Detailed PRD for the `LLMClient` layer
- **LLM Client Layer**
  - [llm_client.py](llm_client.py) — `LLMClient` built on `langchain_ollama.ChatOllama`
  - [client_config.py](client_config.py) — Per-agent config resolution (`ClientConfig`)
  - [llm_exceptions.py](llm_exceptions.py) — Custom exceptions (`EmptyResponseError`, `StructuredOutputValidationError`)
  - [llm_config.json](llm_config.json) — Per-agent model configuration (default profile + agent overrides)
- **Tests**
  - [tests/test_client_config.py](tests/test_client_config.py) — Config resolution tests
  - [tests/test_llm_client.py](tests/test_llm_client.py) — LLMClient unit tests (mocked backend)
  - [tests/manual_llm_client_check.py](tests/manual_llm_client_check.py) — Manual real-server validation script
  - [pytest.ini](pytest.ini) — Pytest configuration
  - [run_tests.sh](run_tests.sh) — Test runner (runs pytest + manual checks)

## LLMClient API

The `LLMClient` provides a single, stable interface for all LLM calls:

```python
from client_config import ClientConfig
from llm_client import LLMClient
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize (uses llm_config.json by default)
client = LLMClient(ClientConfig())

# Plain-text call
response = client.invoke(
    agent_id="detective",
    messages=[
        SystemMessage(content="You are a detective in Mafia."),
        HumanMessage(content="Who looks suspicious?")
    ]
)
print(response.text)
print(f"Used model: {response.model}")
print(f"Latency: {response.latency_seconds:.2f}s")

# Structured output (with Pydantic schema)
from pydantic import BaseModel

class Vote(BaseModel):
    target: str
    confidence: float

response = client.invoke(
    agent_id="player1",
    messages=[HumanMessage(content="Cast your vote.")],
    structured_schema=Vote
)
print(response.structured)  # Parsed Vote instance
```

**Key features:**
- Per-agent config resolution with fallback to defaults (`llm_config.json`)
- Automatic retry on empty responses (3 attempts, 1-second delay)
- Fast-fail on connection/missing-model errors
- Full metadata: latency, token counts, resolved model name
- Structured output validation via Pydantic (optional)
- Strictly synchronous and sequential (no async/concurrency)

### Dual-Path Agent Execution
Agents operate on two distinct layers:
1. **Inner Thoughts Mode (Private State):** Strategic planning, suspicion tracking, role-aware reasoning
2. **Conversation Mode (Public State):** What agents actually say in the game

### Game State (`MafiaState`)
- `public_chat_history`: All spoken dialogue
- `private_memories`: Per-agent internal thoughts and history
- `alive_players`: Current roster
- `current_phase`: DAY_DISCUSSION, DAY_VOTING, or NIGHT_ACTION

### Transcript Export
Game output is exported as a beautifully structured `transcript.md` showing the complete divergence between what agents think vs. what they say.

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running Tests

```bash
# Run full test suite (automated + manual real-server check)
./run_tests.sh

# Run only automated tests (no real-server required)
python -m pytest

# Run with verbose output
python -m pytest -v
```

## Manual Real-Server Testing

To validate behavior against a live Ollama instance:

```bash
# Start Ollama
ollama pull minicpm-v4.5:8b
ollama serve

# In another terminal, run the manual check
python -m tests.manual_llm_client_check
```

## Dependencies

- **Ollama** (optional for tests): Local LLM serving engine
- **langchain-core** (>=0.3): LangChain core abstractions
- **langchain-ollama** (>=0.2): Ollama integration for LangChain
- **pydantic** (>=2): Schema validation for structured output
- **pytest** (>=8): Test framework
- **requests** (>=2.28): HTTP client
- Python 3.10+

See [requirements.txt](requirements.txt) for exact versions.

## Game Rules Quick Reference

**7 players:** 5 Citizens, 2 Mafia
- **Day phase:** Discussion and voting to eliminate a suspect
- **Night phase:** Mafia secretly decides on a target to eliminate
- **Win condition (Citizens):** Eliminate all Mafia
- **Win condition (Mafia):** Equal or outnumber Citizens

See [game-rules.md](docs/game-rules.md) for the full ruleset.
