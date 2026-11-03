"""Explicit, isolated subscription transport for the paired speed experiment.

Every inference uses GPT-6.1 Sol at low effort. Speed labels are requested
settings, never silently inferred from completion latency. No API-key transport,
model substitution, automatic application retry, or speed fallback is allowed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "experiments/agentic_locator_effort_fast_20261025_v1/providers.py"
spec = importlib.util.spec_from_file_location("speed10_private_previous_provider", SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

GPT_MODEL = "gpt-6.1-sol"
SPEEDS = {"normal": ("default", False), "fast": ("priority", True),
          "ultrafast": ("ultrafast", True)}
ProviderError = base.ProviderError


def speed_request_args(argv, speed):
    if speed not in SPEEDS:
        raise ValueError("Expected normal, fast or ultrafast; no aliases or fallback")
    if not argv:
        raise ValueError("Expected Codex executable")
    if tuple(argv[1:]) in base.LOCAL_READINESS_COMMANDS:
        return list(argv)
    if argv[1:3] != ["--no-daemon", "exec"] or argv[-1] != "-":
        raise ValueError("Unsupported subscription subprocess route")
    if "--strict-config" not in argv or "--ignore-user-config" not in argv:
        raise ValueError("Inference requires strict isolated configuration")
    if any(arg.startswith(("service_tier=", "features.fast_mode=")) for arg in argv):
        raise ValueError("Ambiguous preexisting speed configuration")
    tier, fast = SPEEDS[speed]
    return list(argv[:-1]) + ["-c", f'service_tier="{tier}"', "-c",
                             "features.fast_mode=" + str(fast).lower(), "-"]


def served_tier_metadata(stdout):
    """Use only explicit completion metadata, never echoed requested settings."""
    seen = set()
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue  # The inherited strict parser handles malformed envelopes.
        if not isinstance(event, dict) or event.get("type") != "turn.completed":
            continue
        for key in ("served_service_tier", "service_tier"):
            value = event.get(key)
            if value is not None:
                if value not in {"default", "priority", "ultrafast"}:
                    raise ProviderError("invalid_served_tier_metadata")
                seen.add(value)
    if len(seen) > 1:
        raise ProviderError("conflicting_served_tier_metadata")
    return next(iter(seen), None)


def safe_transport_failure(result):
    """Classify provider failures without persisting provider text or reasoning."""
    failures = []
    for line in (result.stdout or "").splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(event, dict) and event.get("type") in {"error", "turn.failed"}:
            failures.append(json.dumps(event).lower())
    text = " ".join(failures)
    if not text and result.returncode:
        text = (result.stderr or "").lower()
    if any(word in text for word in ("server_overloaded", "model is at capacity", "model is overloaded")):
        return "model_capacity_unavailable"
    if ("service_tier" in text or "service tier" in text) and any(
            word in text for word in ("invalid", "unsupported", "unavailable", "not supported")):
        return "requested_service_tier_unavailable"
    return None


class SpeedProvider:
    def __init__(self, *, speed="normal", model=GPT_MODEL, reasoning_effort="low",
                 timeout=600, runner=None, environment=None, executable="codex"):
        if model != GPT_MODEL or reasoning_effort != "low":
            raise ValueError("This experiment requires exact GPT-6.1 Sol / low")
        if speed not in SPEEDS:
            raise ValueError("Unknown speed; no automatic tier fallback")
        self.speed, self.service_tier = speed, SPEEDS[speed][0]
        self.fast_mode = SPEEDS[speed][1]
        self.model, self.reasoning_effort = model, reasoning_effort
        self.timeout, self.runner, self.environment = timeout, runner, environment
        self.executable, self.readiness = executable, None
        self.transport = "codex_cli_chatgpt_subscription"

    def _adapter(self, module, runner=None):
        return module.SubscriptionCodexAdapter(executable=self.executable,
            timeout=self.timeout, runner=runner or self.runner, environment=self.environment)

    def probe(self):
        """Local CLI/auth probe only; no inference, no credential file reads."""
        module = base.codex_module()
        self.readiness = self._adapter(module).probe()
        return dict(self.readiness)

    def request(self, image_path, system_prompt, user_payload, output_schema):
        payload = base.validate_request(system_prompt, user_payload, output_schema)
        original_system_prompt = system_prompt
        system_prompt = base.schema_system_prompt(system_prompt, output_schema)
        image, mime = base.image_input(image_path)
        module = base.codex_module()
        module.PROMPT, module.LABEL_SCHEMA = system_prompt, output_schema
        module.query_prompt = lambda value: value
        final_text, telemetry = {}, {}
        module.parse_judgment = lambda value: final_text.update(text=value)

        def runner(argv, **kwargs):
            result = (self.runner or module.bounded_run)(speed_request_args(argv, self.speed), **kwargs)
            if argv[1:3] == ["--no-daemon", "exec"]:
                telemetry["served_service_tier"] = served_tier_metadata(result.stdout)
                telemetry["failure_kind"] = safe_transport_failure(result)
            return result

        adapter = self._adapter(module, runner)
        if self.readiness is None:
            self.readiness = adapter.probe()
        adapter._readiness = dict(self.readiness)
        common = {"requested_speed": self.speed, "requested_service_tier": self.service_tier,
                  "fast_mode_enabled": self.fast_mode, "reasoning_effort": "low",
                  "requested_model": self.model, "automatic_retries": 0, "fallback_used": False}
        started = time.monotonic()
        try:
            result = adapter.evaluate(payload, image, mime, self.model, "low")
        except module.SubscriptionCodexError as exc:
            raise ProviderError(telemetry.get("failure_kind") or exc.kind,
                {**exc.safe_details, **common, "wall_seconds": time.monotonic() - started,
                 "served_service_tier": telemetry.get("served_service_tier")}) from None
        wall = time.monotonic() - started
        reported, served = result.get("reported_model"), telemetry.get("served_service_tier")
        context = {**common, "reported_model": reported, "served_service_tier": served,
                   "wall_seconds": wall, "usage": result.get("usage")}
        if reported is not None and reported != self.model:
            raise ProviderError("reported_model_mismatch", context)
        if served is not None and served != self.service_tier:
            raise ProviderError("served_service_tier_mismatch", context)
        value = base.parse_model_content(final_text.get("text"), output_schema, context)
        return {"response": value, "provider": "openai", "transport": self.transport,
                **context, "cli_version": result.get("cli_version"), "tool_calls": 0,
                "model_metadata_source": result.get("model_metadata_source"),
                "served_service_tier_confirmed": served is not None,
                "tier_evidence": "Explicit per-request flags; served tier is unknown unless completion metadata reports it.",
                "timing_scope": "CLI inference invocation including startup and network; local readiness probe excluded",
                "cli_internal_retries_observable": False,
                "schema_in_system_prompt": True,
                "original_system_prompt_sha256": base.sha(original_system_prompt.encode()),
                **base.metadata(image, system_prompt, payload, output_schema)}


CodexProvider = SpeedProvider
GPTProvider = SpeedProvider


def make_provider(name="gpt", *, speed="normal", reasoning_effort="low", transport=None, **kwargs):
    if name not in {"gpt", "openai", GPT_MODEL}:
        raise ValueError("Only the requested GPT model is allowed")
    if transport not in {None, "codex_cli_chatgpt_subscription"}:
        raise ValueError("Only the existing ChatGPT subscription transport is allowed")
    return SpeedProvider(speed=speed, reasoning_effort=reasoning_effort, **kwargs)


def local_capabilities():
    provider = SpeedProvider()
    readiness = provider.probe()
    cache = Path.home() / ".codex/models_cache.json"
    raw = cache.read_bytes()
    data = json.loads(raw)
    model = next((row for row in data.get("models", []) if row.get("slug") == GPT_MODEL), None)
    return {"readiness": readiness, "model": GPT_MODEL, "reasoning_effort": "low",
            "catalog_sha256": base.sha(raw), "catalog_fetched_at": data.get("fetched_at"),
            "catalog_model_present": model is not None,
            "catalog_low_supported": bool(model and any(r.get("effort") == "low" for r in model.get("supported_reasoning_levels", []))),
            "catalog_tiers": [r.get("id") for r in model.get("service_tiers", [])] if model else [],
            "catalog_image_supported": bool(model and "image" in model.get("input_modalities", [])),
            "inference_calls": 0, "model_access_verified": False,
            "served_tier_verified": False,
            "disclosure": "Cached catalog and CLI readiness do not attest live model access or served tier."}
