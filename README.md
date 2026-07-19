# AI Mafia: Multi-Agent LLM Simulation

> *Who's lying? Who's telling the truth? Does it even matter if you can't tell the difference?*

A closed room. A circle of LLM agents. One by one, they speak — accuse, defend, deceive.

None of them are human. All of them are trying to survive.

---

## Overview

**AI Mafia** is a multi-agent simulation of the party game Mafia (Werewolf) where every player is an LLM agent orchestrated by an LLM-driven Game Master. The project serves as an architectural stress-test for building complex, state-heavy, multi-agent frameworks using LangGraph with local LLMs via Ollama.

This project runs fully on-premise. No cloud. No oversight. Just agents doing what they do best.

## Current Status

**Phase:** Early prototype. The core Ollama client layer is partially implemented; the full LangGraph game orchestration, agent nodes, and transcript exporter are in progress.

### What Works
- Basic Ollama client integration (`ollama_client.py`)
- Simple test harness (`test_run.py`)
- Model configuration loading

### What's Planned / In Development
- Structured `LLMClient` layer (per [PRD](docs/ollama-client-prd.md))
- LangGraph-based game state and orchestration
- Agent `think_node` and `speak_node` dual-path execution
- GM rule enforcement and phase management
- Structured markdown transcript exporter
- Comprehensive test suite

## Project Structure

- **docs/**
  - [game-rules.md](docs/game-rules.md) — Complete Mafia game rules (7 players: 5 Citizens, 2 Mafia)
  - [idea.md](docs/idea.md) — Conceptual architecture and coding requirements
  - [ollama-client-prd.md](docs/ollama-client-prd.md) — Detailed PRD for the `LLMClient` layer
- **ollama_client.py** — Throwaway Ollama HTTP/CLI integration (prototype; to be replaced by LLMClient)
- **model_config.json** — Simple model configuration (to be expanded to `llm_config.json` per PRD)
- **test_run.py** — Basic manual test script
- **requirements.txt** — Python dependencies (currently incomplete; LangGraph/langchain-ollama TBD)

## Key Design Patterns

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

# Start Ollama and pull the target model
ollama pull minicpm-v4.5:8b
ollama serve

# In another terminal, run the test harness
python test_run.py
```

## Dependencies

- **Ollama**: Local LLM serving engine
- **langchain-ollama**: LLM integration (LangChain)
- **langgraph**: Multi-agent orchestration framework
- **requests**: HTTP client
- Python 3.10+

## Game Rules Quick Reference

**7 players:** 5 Citizens, 2 Mafia
- **Day phase:** Discussion and voting to eliminate a suspect
- **Night phase:** Mafia secretly decides on a target to eliminate
- **Win condition (Citizens):** Eliminate all Mafia
- **Win condition (Mafia):** Equal or outnumber Citizens

See [game-rules.md](docs/game-rules.md) for the full ruleset.
