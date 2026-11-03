#!/usr/bin/env python3
"""Local command provider using the preserved strict Codex CLI adapter.

Uses the user's existing `codex login`; never reads or copies an auth file.
This is a public CLI integration, not the native-session evaluation transport.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/original/experiments/dual_model_review_20261020_v1"))


def main():
    import jsonschema
    from runtime.core import load_original
    module = load_original("experiments/dual_model_review_20261020_v1/subscription_codex.py", "qgvi_original_codex")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    adapter = module.SubscriptionCodexAdapter(timeout=570)
    if args.probe:
        print(json.dumps(adapter.probe()))
        return
    packet = json.load(sys.stdin)
    if packet["requested_model"] != "gpt-6.1-sol" or packet["reasoning_effort"] != "low":
        raise ValueError("This release freezes GPT-6.1 Sol / low")
    image = Path(packet["image_path"]).read_bytes()
    if hashlib.sha256(image).hexdigest() != packet["image_sha256"]:
        raise ValueError("Provider image changed")
    module.PROMPT = packet["system_prompt"]
    module.LABEL_SCHEMA = packet["output_schema"]
    module.query_prompt = lambda value: value
    def parse(value):
        result = json.loads(value)
        jsonschema.validate(result, module.LABEL_SCHEMA)
        return result
    module.parse_judgment = parse
    result = adapter.evaluate(json.dumps(packet["user_payload"], allow_nan=False), image,
        "image/png" if image.startswith(b"\x89PNG") else "image/jpeg",
        packet["requested_model"], packet["reasoning_effort"])
    if result.get("reported_model") not in (None, packet["requested_model"]):
        raise ValueError("Codex reported a different model")
    print(json.dumps({"response": result["judgment"],
        "reported_model": packet["requested_model"], "reasoning_effort": packet["reasoning_effort"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("QGVI Codex provider failed: " + getattr(exc, "kind", type(exc).__name__), file=sys.stderr)
        raise SystemExit(1)
