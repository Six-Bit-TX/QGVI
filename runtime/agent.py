"""Additive adapter that calls the original QGVI-A controller unchanged."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading

from . import provider_bridge


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "research/original"
FULL = ORIGINAL / "experiments/qgvi_a_full_eval_20261029_v1"
_LOCK = threading.RLock()
_MODULES = ("providers", "io_utils", "protocol", "state", "early_exit",
            "tool_host", "baselines", "navigation_backend", "controller")


@contextmanager
def original_controller():
    """Isolate the research modules' legacy absolute imports during one run."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("The original controller uses SIGALRM; run it in the main process thread")
    with _LOCK:
        previous = {name: sys.modules.get(name) for name in _MODULES}
        previous_path = list(sys.path)
        try:
            sys.path.insert(0, str(FULL))
            sys.path.insert(0, str(ORIGINAL / "qgvi-rag"))
            sys.modules["providers"] = provider_bridge
            for name in _MODULES[1:]:
                spec = importlib.util.spec_from_file_location(name, FULL / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            yield sys.modules["controller"], sys.modules["tool_host"]
        finally:
            sys.path[:] = previous_path
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


def run_agent(query, folder, retrieval, navigation, provider_argv, effort="low"):
    """Run the evaluated state machine with explicit public tool/command adapters.

The original output remains untouched in ``result.json``. A separate transport
receipt describes the changed integration; it is not a new scientific score.
An external command is trusted local code and is responsible for its own model
credentials. No command is invoked if the geometric early exit resolves a task.
"""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    provider = provider_bridge.CommandProvider(provider_argv, reasoning_effort=effort)
    command_digest = hashlib.sha256(json.dumps(list(provider_argv),
                                              ensure_ascii=False).encode()).hexdigest()
    binding = {"schema": "qgvi_public_command_transport_v1",
               "transport": provider_bridge.TRANSPORT,
               "command_sha256": command_digest,
               "requested_model": provider_bridge.MODEL,
               "reasoning_effort": effort,
               "algorithm_source_modified": False,
               "requested_model_identity_verified": False,
               "native_codex_transport_reproduction": False,
               "note": "The external command must honor requested_model and reasoning_effort. The controller's model/tier fields record requests, not verified service identity. No credentials or command arguments are stored in this receipt."}
    receipt = folder / "public_transport.json"
    if receipt.exists() and json.loads(receipt.read_text()) != binding:
        raise ValueError("Provider command or effort changed; use a new output directory")
    if not receipt.exists():
        receipt.write_text(json.dumps(binding, indent=2) + "\n")
    with original_controller() as (controller, tools):
        config = controller.default_config()
        config.update(early_exit_enabled=True, early_exit_route="auto")
        config["reasoning_effort"] = {"gpt": effort}
        config["provider_transport"] = {"gpt": provider_bridge.TRANSPORT}
        host = tools.ToolHost(query, folder, retrieval=retrieval, navigation=navigation)
        result = controller.run_one(query, "agent", "gpt", folder, config,
                                    provider=provider, host=host)
        return {"result": result, "public_transport": binding}
