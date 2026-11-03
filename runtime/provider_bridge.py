"""Explicit external-command transport for the unchanged research controller.

The command receives one JSON packet on stdin and prints an action with explicit
model/effort metadata on stdout. This is an integration boundary, not an implementation of an
API client or a claim to reproduce the evaluated native Codex session transport.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

import jsonschema


TRANSPORT = "external_command"
MODEL = "gpt-6.1-sol"
EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max", "ultra"})
DEFAULT_TRANSPORTS = {"gpt": TRANSPORT}
MODEL_OUTPUT_ERRORS = frozenset({"invalid_model_json", "invalid_model_schema",
                                "model_output_truncated", "model_refusal",
                                "model_unexpected_tool_call"})
MAX_CAPTURE_BYTES = 4 * 1024 * 1024


class ProviderError(RuntimeError):
    def __init__(self, kind, safe_details=None):
        self.kind = kind
        self.safe_details = safe_details or {}
        super().__init__(kind)


def validate_effort(name, effort):
    if name != "gpt" or effort not in EFFORTS:
        raise ValueError("Expected gpt and an explicitly supported reasoning effort")
    return effort


def require_speed_config(config):
    if (config.get("speed") != "normal"
            or config.get("service_tier", {}).get("gpt") != "default"
            or config.get("fast_mode") is not False):
        raise ValueError("The public command bridge uses explicit normal/default requests")
    return "default"


def make_provider(*args, **kwargs):
    raise ProviderError("external_command_provider_must_be_injected")


def _strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = item
        return result

    def bad_constant(value):
        raise ValueError("Nonfinite JSON number")

    result = json.loads(value, object_pairs_hook=pairs, parse_constant=bad_constant)
    json.dumps(result, allow_nan=False)
    return result


def _stop(process):
    # A parent can exit while a descendant still holds an output pipe open.
    # Kill this dedicated group even when the immediate child already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def bounded_command(argv, packet, timeout):
    """Bound time and both output pipes; never expose command stderr in errors."""
    if os.name != "posix":
        raise ProviderError("external_command_requires_posix")
    started = time.monotonic()
    try:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True,
                                   shell=False)
    except OSError:
        raise ProviderError("external_command_start_failed") from None
    capture = {"stdout": bytearray(), "stderr": bytearray()}
    pending = memoryview(packet)
    try:
        with selectors.DefaultSelector() as selector:
            for stream in (process.stdin, process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise ProviderError("external_command_timeout")
                for key, _ in selector.select(min(remaining, 0.2)):
                    if key.data == "stdin":
                        try:
                            count = os.write(key.fd, pending[:65536])
                            pending = pending[count:]
                        except BrokenPipeError:
                            pending = pending[len(pending):]
                        if not pending:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    else:
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        capture[key.data].extend(chunk)
                        if len(capture[key.data]) > MAX_CAPTURE_BYTES:
                            raise ProviderError("external_command_output_limit")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise ProviderError("external_command_timeout")
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                raise ProviderError("external_command_timeout") from None
        if code != 0:
            raise ProviderError("external_command_failed", {"returncode": code})
        return bytes(capture["stdout"])
    finally:
        _stop(process)
        for stream in (process.stdin, process.stdout, process.stderr):
            if not stream.closed:
                stream.close()


class CommandProvider:
    transport = TRANSPORT
    service_tier = "default"

    def __init__(self, argv, *, reasoning_effort="low", timeout=600):
        if (not isinstance(argv, (list, tuple)) or not argv
                or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv)):
            raise ValueError("Provider command must be a nonempty argument list, not a shell string")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ValueError("Command timeout must be greater than zero and at most 600 seconds")
        self.argv = list(argv)
        self.reasoning_effort = validate_effort("gpt", reasoning_effort)
        self.timeout = timeout

    def request(self, image_path, system_prompt, user_payload, output_schema):
        image_path = Path(image_path).resolve(strict=True)
        image_bytes = image_path.read_bytes()
        if not 0 < len(image_bytes) <= 20 * 1024 * 1024:
            raise ValueError("Expected an image smaller than 20 MiB")
        if not image_bytes.startswith((b"\x89PNG", b"\xff\xd8\xff")):
            raise ValueError("Expected PNG or JPEG input")
        jsonschema.Draft202012Validator.check_schema(output_schema)
        packet = {"schema": "qgvi_external_provider_request_v1",
                  "requested_model": MODEL, "reasoning_effort": self.reasoning_effort,
                  "requested_service_tier": "default", "fast_mode_enabled": False,
                  "image_path": str(image_path),
                  "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                  "system_prompt": system_prompt, "user_payload": user_payload,
                  "output_schema": output_schema,
                  "response_contract": {"response": "Action object matching output_schema",
                                        "reported_model": MODEL,
                                        "reasoning_effort": self.reasoning_effort}}
        payload = (json.dumps(packet, ensure_ascii=False, allow_nan=False) + "\n").encode()
        started = time.monotonic()
        raw = bounded_command(self.argv, payload, self.timeout)
        try:
            response = _strict_json(raw)
        except (ValueError, UnicodeError, RecursionError):
            raise ProviderError("invalid_model_json") from None
        if (not isinstance(response, dict)
                or set(response) != {"response", "reported_model", "reasoning_effort"}):
            raise ProviderError("missing_external_model_metadata")
        if response["reported_model"] != MODEL:
            raise ProviderError("reported_model_mismatch")
        if response["reasoning_effort"] != self.reasoning_effort:
            raise ProviderError("reported_reasoning_effort_mismatch")
        action = response["response"]
        try:
            jsonschema.validate(action, output_schema)
        except jsonschema.ValidationError:
            raise ProviderError("invalid_model_schema") from None
        return {"response": action, "provider": "user_supplied_command",
                "transport": TRANSPORT, "requested_model": MODEL,
                "reported_model": response["reported_model"], "reasoning_effort": self.reasoning_effort,
                "requested_speed": "normal", "requested_service_tier": "default",
                "served_service_tier": None, "fast_mode_enabled": False,
                "served_service_tier_confirmed": False,
                "wall_seconds": time.monotonic() - started,
                "image_sha256": packet["image_sha256"],
                "timing_scope": "external command startup, execution and response parsing",
                "model_identity_verified": False,
                "model_identity_source": "external_command_report_not_independent_attestation",
                "native_codex_transport_reproduction": False,
                "automatic_retries": 0}
