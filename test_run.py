import json
import time
from pathlib import Path

from ollama_client import OllamaRunner


def load_config(path: str = "model_config.json") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(p.read_text())


def main():
    cfg = load_config()
    model = cfg.get("model")
    if not model:
        raise ValueError("model not set in config")

    runner = OllamaRunner(model=model)

    print(f"Model: {model}")
    print("Warming up model...")
    t0 = time.time()
    runner.warmup("Hello, warmup")
    warmup_time = time.time() - t0
    print(f"Warmup time: {warmup_time:.2f}s")

    prompt = "Give a short runtime benchmark: what is 12345 * 2?"
    print(f"Sending prompt: {prompt}")
    t0 = time.time()
    resp = runner.chat(prompt, max_tokens=64)
    elapsed = time.time() - t0

    print(f"Elapsed chat time: {elapsed:.2f}s")
    print("Response text:")
    print(resp.get("text"))


if __name__ == "__main__":
    main()
