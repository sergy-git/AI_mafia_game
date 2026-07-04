# Agentic Mafia: Multi-Agent State Simulation

This document provides a conceptual overview and coding task description for a multi-agent simulation of the party game **Mafia** (also known as *Werewolf*). The primary goal of this project is to serve as an architectural "stress-test" and learning milestone for building complex, state-heavy, multi-agent frameworks (such as LangGraph) running local LLMs via Ollama.


---

## 1. The Core Idea & Paradigm

The project simulates a game of Mafia where every player is an LLM agent, orchestrated by a LLM-driven Game Master (GM). The game relies on **imperfect information**, **hidden states**, and **deception**.

To simulate human-like behavior, each agent operates on two distinct layers:

```text
┌────────────────────────────────────────────────────────┐
│                      AGENT PLAYER                      │
│                                                        │
│  ┌────────────────────────┐    ┌────────────────────┐  │
│  │  Inner Thoughts Mode   │───>│  Conversation Mode │  │
│  │ (Private Hidden State) │    │ (Public Messaging) │  │
│  └────────────────────────┘    └────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

1. **Inner Thoughts Mode (Private State):** The agent's internal monologue, raw strategic planning, long-term memory, and suspicion tracker. This layer knows the agent's real identity, analyzes historical data, and calculates risk. **This data is completely hidden from other players.**
2. **Conversation Mode (Public State):** What the agent actually says during the day phase. This can involve lying, strategic manipulation, forming alliances, or accusing others.

---

## 2. Game Architecture & Roles

### A. The Game Master (GM)

The backbone of the application. The GM can be written in pure Python logic combined with a lightweight LLM execution layer.

* **Responsibilities:** Enforces game rules, tracks the master state (who is alive, dead, or assigned to which role), alternates between Day and Night cycles, counts votes, and logs every step.
* **Visibility:** Has absolute visibility over the entire state (public and private).

### B. The Players (Multi-LLM Agents)

Each agent is configured with a distinct personality profile, system prompt, and secret role assignment.

* **Mafia:** Must coordinate silently during the Night phase to eliminate a Citizen, and blend in during the Day phase.
* **Citizens (Innocents):** Must deduce who the Mafia members are during the Day phase based on conversational behavior, voting patterns, and inconsistencies.
* **Special Roles (Optional - Detective/Doctor):** Add additional hidden actions during the Night phase.

---

## 3. The Coding Task

Your task is to implement a minimum viable simulation of this game using **LangGraph** and **Ollama**.

### Architectural Requirements

1. **Global State Definition (`MafiaState`)**

   The application state must be explicitly tracked inside a shared graph state containing:

   * `public_chat_history`: A running ledger of everything spoken out loud.
   * `private_memories`: A dictionary mapping `agent_id` to its history of private thoughts.
   * `alive_players`: A list of players currently active.
   * `current_phase`: `"DAY_DISCUSSION"`, `"DAY_VOTING"`, or `"NIGHT_ACTION"`.

2. **The Dual-Path Node Pattern**

   When it is a player's turn to speak, the framework must execute two steps sequentially:

   * **Node 1 (`think_node`):** Pass the `public_chat_history` + `private_memories[agent_id]` + `secret_role` to the Ollama model. Instruct it to output its strategic assessment. Append this output *only* to `private_memories[agent_id]`.
   * **Node 2 (`speak_node`):** Pass the same history *plus* the freshly generated inner thought. Instruct the model to formulate its public statement or vote. Append this statement to `public_chat_history`.

3. **Transcript and Debug Tool (pure Python)**

   Implement an exporter that listens to state changes or parses the final graph output, generating a beautifully structured markdown file (`transcript.md`). The log must display the complete divergence between what agents *thought* and what they *said*:

   ```markdown
   ### Turn 3: Day Discussion (Agent: Alex)
   * **Secret Role:** Mafia
   * **Inner Monologue:** "Ben is getting suspicious of me. If I defend myself too aggressively, Chloe will notice. I need to pivot the blame onto David because he voted erratically last round."
   * **Public Statement:** *"Hey everyone, I notice David has been pretty quiet today. David, why did you shift your vote away from Emily at the last second yesterday?"*
   ```

---

## 4. Technical Constraints & Setup

* **Local Environment:** The project must run entirely on-premise.
* **IDE:** VS Code.
* **LLM Engine:** Ollama (Recommended: `minicpm-v4.5:8b`).
* **Framework Stack:** `langgraph`, `langchain-ollama`, and native Python file I/O for logging.