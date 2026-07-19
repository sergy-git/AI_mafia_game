"""Manual, real-server exercise script for `LLMClient` — NOT part of the
automated test suite.

Requires a real local Ollama server (see repo notes: `systemctl is-active
ollama`) with the configured model already pulled. Run via the helper script
from the repo root:

    ./run_manual_check.sh

This is the spiritual successor to `test_run.py` for the new client layer:
it prints latency/token metadata and both plain-text and structured-output
behavior so you can eyeball real-world model behavior outside of pytest.
"""

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from llm_client import LLMClient


class VoteDecision(BaseModel):
    target: str
    reason: str


def main():
    client = LLMClient()

    print("--- plain text call ---")
    response = client.invoke(
        agent_id="default",
        messages=[HumanMessage(content="Say hello in one short sentence.")],
    )
    print(f"model:    {response.model}")
    print(f"latency:  {response.latency_seconds:.2f}s")
    print(f"tokens:   prompt={response.prompt_tokens} "
          f"completion={response.completion_tokens} total={response.total_tokens}")
    print(f"text:     {response.text!r}")

    print("\n--- structured output call ---")
    response = client.invoke(
        agent_id="default",
        messages=[
            HumanMessage(
                content=(
                    "You are voting to eliminate a player named 'player_2'. "
                    "Respond with your vote."
                )
            )
        ],
        structured_schema=VoteDecision,
    )
    print(f"model:      {response.model}")
    print(f"latency:    {response.latency_seconds:.2f}s")
    print(f"raw text:   {response.text!r}")
    print(f"structured: {response.structured!r}")


if __name__ == "__main__":
    main()
