import json
import time
import subprocess
from typing import Any, Dict

import requests


class OllamaRunner:
    def __init__(self, model: str, host: str = "http://localhost:11434", cli_path: str = "ollama"):
        self.model = model
        self.host = host.rstrip('/')
        self.cli_path = cli_path

    def warmup(self, prompt: str = "Hello", timeout: int = 30) -> Dict[str, Any]:
        """Send a tiny prompt to warm the model into memory."""
        return self.chat(prompt, max_tokens=8, timeout=timeout)

    def chat(self, prompt: str, max_tokens: int = 256, timeout: int = 60) -> Dict[str, Any]:
        """Return a dict with the raw response and a best-effort text field."""
        # Try HTTP API first
        try:
            resp = self._http_generate(prompt, max_tokens=max_tokens, timeout=timeout)
        except Exception:
            resp = self._cli_generate(prompt, max_tokens=max_tokens, timeout=timeout)

        text = self._extract_text(resp)
        return {"raw": resp, "text": text}

    def _http_generate(self, prompt: str, max_tokens: int = 256, timeout: int = 60) -> Any:
        # Use /api/chat (not the raw /api/generate completion endpoint) so the
        # model gets its proper instruct chat template. Without it, this
        # instruct-tuned model treats the prompt as free-form text to continue
        # rather than a question to answer, and occasionally emits an
        # immediate stop token (empty response).
        url = f"{self.host}/api/chat"
        # Ollama ignores a top-level "max_tokens" field; the token limit must be
        # passed as "num_predict" inside "options". Also disable streaming so we
        # get a single JSON response instead of a long series of NDJSON chunks.
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return r.text

    def _cli_generate(self, prompt: str, max_tokens: int = 256, timeout: int = 60) -> Any:
        # Fallback to calling the `ollama` CLI if HTTP API is not available.
        # The CLI flags may vary by version; we try a reasonable invocation.
        cmd = [self.cli_path, "generate", self.model, "--prompt", prompt, "--json", "--max-tokens", str(max_tokens)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"ollama CLI error: {proc.stderr.strip()}")
        try:
            return json.loads(proc.stdout)
        except Exception:
            return proc.stdout

    def _extract_text(self, resp: Any) -> str:
        # Heuristic extraction of human-readable text from various ollama responses.
        if isinstance(resp, str):
            return resp
        if not isinstance(resp, dict):
            try:
                return json.dumps(resp)
            except Exception:
                return str(resp)

        # Common places to look for text
        # 1) 'message.content' (ollama's /api/chat result field)
        message = resp.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

        # 2) 'response' key (ollama's /api/generate result field)
        if "response" in resp and isinstance(resp["response"], str):
            return resp["response"]

        # 3) 'text' key
        if "text" in resp and isinstance(resp["text"], str):
            return resp["text"]

        # 2) 'output' key (ollama sometimes returns list of outputs)
        out = resp.get("output")
        if isinstance(out, str):
            return out
        if isinstance(out, list):
            pieces = []
            for item in out:
                if isinstance(item, dict) and "content" in item:
                    pieces.append(item["content"])
                elif isinstance(item, str):
                    pieces.append(item)
            if pieces:
                return "\n".join(pieces)

        # 3) 'choices' pattern
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                for key in ("message", "text", "content"):
                    if key in first:
                        val = first[key]
                        if isinstance(val, dict) and "content" in val:
                            return val["content"]
                        if isinstance(val, str):
                            return val

        # 4) fallback to JSON dump
        try:
            return json.dumps(resp)
        except Exception:
            return str(resp)


if __name__ == "__main__":
    # Small self-test when run directly
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "minicpm-v4.5:8b"
    runner = OllamaRunner(model=model)
    print("Warming up...")
    print(runner.warmup("Hi"))
    print("Chat: ", runner.chat("What is 2+2?"))
