"""Blinded one-image reviews through the existing ChatGPT Codex CLI login.

No API credential, auth file, conversation history, or model reasoning is read
or persisted by this adapter. Only ``evaluate`` can start inference. ``probe``
uses local CLI help/version/login status and never starts a model turn.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import tempfile
import time

try:
    from .model_review import LABEL_SCHEMA, PROMPT, canonical, normalize_usage, parse_judgment
except ImportError:
    from model_review import LABEL_SCHEMA, PROMPT, canonical, normalize_usage, parse_judgment


MAX_CAPTURE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
REQUIRED_FLAGS = ("--image", "--output-schema", "--json", "--ephemeral",
                  "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
                  "--strict-config")
DISABLED_FEATURES = (
    "shell_tool", "shell_snapshot", "apps", "plugins", "remote_plugin", "hooks",
    "memories", "multi_agent", "multi_agent_v2", "browser_use", "browser_use_external",
    "computer_use", "in_app_browser", "image_generation", "view_image", "code_mode",
    "code_mode_host", "skill_search", "skill_mcp_dependency_install", "tool_suggest",
    "sleep_tool", "goals", "request_permissions_tool", "unbounded_connection_retries",
)
KNOWN_EVENTS = {"thread.started", "turn.started", "turn.completed", "turn.failed",
                "item.started", "item.updated", "item.completed", "error"}
ENV_ALLOWLIST = {
    "PATH", "HOME", "CODEX_HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
}


class SubscriptionCodexError(Exception):
    """A fixed error category and allowlisted metadata, never provider output."""

    def __init__(self, kind, safe_details=None):
        self.kind = kind
        self.safe_details = safe_details or {}
        super().__init__(kind)


def sanitized_environment(source=None):
    """Preserve the installed login location; drop keys/provider/session overrides."""
    source = os.environ if source is None else source
    env = {key: value for key, value in source.items() if key in ENV_ALLOWLIST}
    env["RUST_LOG"] = "off"
    env["NO_COLOR"] = "1"
    return env


def query_prompt(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 16000 or "\x00" in query:
        raise ValueError("Expected a nonempty query of at most 16000 characters")
    return "Query to evaluate:\n" + query


def config_overrides(workdir, model, reasoning_effort):
    """Explicit CLI configuration, also reusable by an offline prompt audit.

    The caller writes the unchanged PROMPT to ``workdir/rubric.txt``. These
    overrides accompany --ignore-user-config in inference; they are not a
    substitute for that flag when auditing another CLI subcommand.
    """
    if not isinstance(model, str) or not MODEL_NAME.fullmatch(model):
        raise ValueError("An explicit model identifier is required")
    if reasoning_effort not in EFFORTS:
        raise ValueError("Unsupported reasoning effort")
    values = {
        "model": model, "model_provider": "openai", "forced_login_method": "chatgpt",
        "model_reasoning_effort": reasoning_effort, "model_reasoning_summary": "none",
        "model_instructions_file": str(Path(workdir) / "rubric.txt"),
        "developer_instructions": "", "approval_policy": "never", "sandbox_mode": "read-only",
        "web_search": "disabled", "mcp_servers": {}, "agents.enabled": False,
        "project_doc_max_bytes": 0, "project_doc_fallback_filenames": [],
        "skills.bundled.enabled": False, "skills.include_instructions": False,
        "memories.use_memories": False, "memories.generate_memories": False,
        "history.persistence": "none", "hide_agent_reasoning": True,
        "show_raw_agent_reasoning": False, "analytics.enabled": False,
        "feedback.enabled": False, "check_for_update_on_startup": False,
        "suppress_unstable_features_warning": True,
        "shell_environment_policy.inherit": "none",
        "features.skip_host_skill_discovery": True,
    }
    values.update({"features." + feature: False for feature in DISABLED_FEATURES})
    # JSON scalars/arrays are also TOML literals; an empty table needs TOML syntax.
    return [key + "=" + ("{}" if value == {} else json.dumps(value, ensure_ascii=False))
            for key, value in values.items()]


def _stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def bounded_run(argv, *, input="", cwd, env, timeout):
    """Private in-memory output capture; kill the process group on time/size limits."""
    try:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd=cwd, env=env,
                                   start_new_session=True)
    except (OSError, ValueError):
        raise SubscriptionCodexError("cli_launch_failed") from None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    pending = memoryview(input.encode("utf-8"))
    try:
        with selectors.DefaultSelector() as selector:
            for name in buffers:
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            os.set_blocking(process.stdin.fileno(), False)
            if pending:
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                process.stdin.close()
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SubscriptionCodexError("cli_timeout")
                for key, _ in selector.select(min(remaining, 0.2)):
                    if key.data == "stdin":
                        try:
                            pending = pending[os.write(key.fd, pending[:65536]):]
                        except BrokenPipeError:
                            pending = pending[:0]
                        if not pending:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    block = os.read(key.fd, 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    buffers[key.data].extend(block)
                    if sum(map(len, buffers.values())) > MAX_CAPTURE_BYTES:
                        raise SubscriptionCodexError("cli_output_too_large")
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        _stop_process(process)
        raise SubscriptionCodexError("cli_timeout") from None
    except BaseException:
        _stop_process(process)
        raise
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
    return subprocess.CompletedProcess(argv, process.returncode,
                                       buffers["stdout"].decode("utf-8", errors="replace"),
                                       buffers["stderr"].decode("utf-8", errors="replace"))


def _failure_kind(text):
    value = text.lower()
    if any(token in value for token in ("quota", "usage limit", "usage_limit", "insufficient_quota",
                                        "credits exhausted", "out of credits")):
        return "subscription_quota_exhausted"
    if any(token in value for token in ("rate limit", "rate_limit", "too many requests", "429")):
        return "subscription_rate_limited"
    if any(token in value for token in ("unauthorized", "authentication", "not logged in", "401",
                                        "token expired", "refresh token", "sign in", "login required")):
        return "subscription_auth_unavailable"
    if any(token in value for token in ("model is not supported", "unsupported model", "model_not_found",
                                        "model does not exist", "not supported when using codex")):
        return "subscription_model_unavailable"
    if any(token in value for token in ("unknown field", "unknown configuration field",
                                        "error loading config", "unexpected argument", "unknown variant",
                                        "error parsing", "invalid config", "unrecognized")):
        return "unsupported_cli_configuration"
    return "cli_failed"


def parse_cli_result(result):
    """Extract a final label and numeric metadata; discard every other payload."""
    stdout, stderr = result.stdout or "", result.stderr or ""
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise SubscriptionCodexError("invalid_cli_output")
    if len(stdout.encode()) + len(stderr.encode()) > MAX_CAPTURE_BYTES:
        raise SubscriptionCodexError("cli_output_too_large")
    counts, item_counts, startup_warnings = Counter(), Counter(), Counter()
    usage, final_text, reported_model, reasoning_output_tokens = None, None, None, None
    completed, failure, tool_seen = 0, None, False
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            raise SubscriptionCodexError(_failure_kind(stderr) if result.returncode else
                                         "invalid_cli_output") from None
        if not isinstance(event, dict) or event.get("type") not in KNOWN_EVENTS:
            raise SubscriptionCodexError("unsupported_cli_events")
        kind = event["type"]
        counts[kind] += 1
        if kind in {"thread.started", "turn.started", "turn.completed"} and "model" in event:
            model = event["model"]
            if not isinstance(model, str) or not MODEL_NAME.fullmatch(model):
                raise SubscriptionCodexError("invalid_model_metadata")
            if reported_model is not None and model != reported_model:
                raise SubscriptionCodexError("conflicting_model_metadata")
            reported_model = model
        if kind.startswith("item."):
            item = event.get("item")
            if not isinstance(item, dict):
                raise SubscriptionCodexError("invalid_cli_output")
            item_kind = item.get("type")
            safe_kind = item_kind if item_kind in {"agent_message", "reasoning", "error"} else "other"
            item_counts[safe_kind] += 1
            if safe_kind == "other":
                tool_seen = True
            elif item_kind == "error":
                # Codex 0.159.2 uses error items for these nonfatal startup
                # warnings. Save only fixed codes, never their local paths/text.
                message = item.get("message", "")
                warning = None
                if isinstance(message, str) and counts["turn.started"] == 0:
                    if message.startswith("Under-development features enabled: skip_host_skill_discovery."):
                        warning = "host_skill_discovery_disabled"
                    elif message.startswith("Code Mode is unavailable because code-mode host is disabled."):
                        warning = "code_mode_host_disabled"
                if warning:
                    startup_warnings[warning] += 1
                else:
                    failure = _failure_kind(message if isinstance(message, str) else "")
            elif kind == "item.completed" and item_kind == "agent_message":
                final_text = item.get("text")
        elif kind == "turn.completed":
            completed += 1
            try:
                raw_usage = event.get("usage")
                if not isinstance(raw_usage, dict):
                    raise ValueError("Missing usage")
                usage = normalize_usage({key: raw_usage.get(key) for key in
                                         ("input_tokens", "cached_input_tokens", "output_tokens")})
                # Reasoning is part of output_tokens, never another billable total.
                reasoning_output_tokens = raw_usage.get("reasoning_output_tokens")
                if reasoning_output_tokens is not None and (
                        type(reasoning_output_tokens) is not int or
                        not 0 <= reasoning_output_tokens <= usage["output_tokens"]):
                    raise ValueError("Invalid reasoning token count")
            except (ValueError, TypeError):
                raise SubscriptionCodexError("invalid_usage_metadata") from None
        elif kind in {"error", "turn.failed"}:
            failure = _failure_kind(json.dumps(event))
    details = {"event_counts": dict(sorted(counts.items())),
               "item_counts": dict(sorted(item_counts.items())), "usage": usage,
               "startup_warning_counts": dict(sorted(startup_warnings.items())),
               "reasoning_output_tokens": reasoning_output_tokens,
               "reported_model": reported_model, "effective_model": reported_model,
               "model_metadata_source": "cli_json_event" if reported_model else None}
    if tool_seen:
        raise SubscriptionCodexError("unexpected_tool_activity", details)
    if failure or result.returncode:
        raise SubscriptionCodexError(failure or _failure_kind(stderr), details)
    if completed != 1 or counts["turn.started"] != 1 or counts["thread.started"] != 1:
        raise SubscriptionCodexError("incomplete_or_multiple_turns", details)
    try:
        judgment = parse_judgment(final_text)
    except (ValueError, TypeError):
        raise SubscriptionCodexError("invalid_final_label", details) from None
    return {"judgment": judgment, "finish_reason": "completed", **details}


class SubscriptionCodexAdapter:
    """One fresh ephemeral CLI invocation per image, with no retry or fallback.

    ``runner`` follows ``bounded_run``'s signature and is injectable for tests.
    Failures raise SubscriptionCodexError; its kind/safe_details are safe to save.
    A successful probe establishes local CLI/auth readiness only, not remaining
    subscription quota or model access. Those are established by a live pilot.
    """

    def __init__(self, *, executable="codex", timeout=120, runner=None, environment=None):
        if type(timeout) not in (int, float) or not 1 <= timeout <= 600:
            raise ValueError("Timeout must be between 1 and 600 seconds")
        self.executable = str(executable)
        self.timeout = timeout
        self.runner = runner or bounded_run
        self.environment = sanitized_environment(environment)
        self._readiness = None

    def _run(self, args, *, cwd, input="", timeout=None):
        try:
            return self.runner([self.executable, *args], input=input, cwd=str(cwd),
                               env=dict(self.environment), timeout=timeout or self.timeout)
        except SubscriptionCodexError:
            raise
        except subprocess.TimeoutExpired:
            raise SubscriptionCodexError("cli_timeout") from None
        except Exception:
            raise SubscriptionCodexError("cli_launch_failed") from None

    def probe(self):
        """Local-only readiness; neither read auth files nor contact a model."""
        results = {}
        with tempfile.TemporaryDirectory(prefix="review-", dir="/tmp") as folder:
            for key, args in (("version", ["--version"]), ("help", ["--help"]),
                              ("exec_help", ["exec", "--help"]),
                              ("features", ["features", "list"]),
                              ("auth", ["-c", 'forced_login_method="chatgpt"', "login", "status"])):
                results[key] = self._run(args, cwd=folder, timeout=min(15, self.timeout))
        version = (results["version"].stdout or "").strip()
        if not re.fullmatch(r"codex-cli [A-Za-z0-9.+_-]{1,80}", version):
            version = None
        auth = (results["auth"].stdout or "") + (results["auth"].stderr or "")
        chatgpt = results["auth"].returncode == 0 and "Logged in using ChatGPT" in auth
        flags_ok = all(flag in results["exec_help"].stdout for flag in REQUIRED_FLAGS)
        flags_ok = flags_ok and "--no-daemon" in results["help"].stdout
        features = {line.split()[0] for line in results["features"].stdout.splitlines() if line.split()}
        missing = sorted(set(DISABLED_FEATURES + ("skip_host_skill_discovery",)) - features)
        commands_ok = all(result.returncode == 0 for result in results.values())
        ready = bool(version and chatgpt and flags_ok and not missing and commands_ok)
        self._readiness = {"ready": ready, "cli_version": version,
                           "login_method": "chatgpt" if chatgpt else "unavailable",
                           "auth_mode": "chatgpt" if chatgpt else "unavailable",
                           "required_flags_supported": bool(flags_ok), "missing_features": missing,
                           "model_access_verified": False, "quota_verified": False,
                           "reason": None if ready else "local_subscription_readiness_failed"}
        return dict(self._readiness)

    def evaluate(self, query, image_bytes, mime, model, reasoning_effort):
        prompt = query_prompt(query)
        config_overrides("/tmp", model, reasoning_effort)  # Validate before any subprocess.
        if not isinstance(image_bytes, bytes) or not 0 < len(image_bytes) <= MAX_IMAGE_BYTES:
            raise ValueError("Expected image bytes up to 20 MiB")
        if mime not in {"image/png", "image/jpeg"}:
            raise ValueError("Only PNG and JPEG images are supported")
        magic = b"\x89PNG\r\n\x1a\n" if mime == "image/png" else b"\xff\xd8\xff"
        if not image_bytes.startswith(magic):
            raise ValueError("Image signature and MIME type differ")
        readiness = self._readiness or self.probe()
        if not readiness["ready"]:
            raise SubscriptionCodexError("local_subscription_readiness_failed", readiness)
        with tempfile.TemporaryDirectory(prefix="review-", dir="/tmp") as folder:
            workdir = Path(folder)
            image_name = "image.png" if mime == "image/png" else "image.jpg"
            for name, data in ((image_name, image_bytes), ("rubric.txt", PROMPT.encode()),
                               ("schema.json", canonical(LABEL_SCHEMA))):
                path = workdir / name
                path.write_bytes(data)
                path.chmod(0o600)
            args = ["--no-daemon", "exec", "--strict-config", "--ignore-user-config",
                    "--ignore-rules", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                    "--json", "--color", "never", "--image", image_name,
                    "--output-schema", "schema.json", "--model", model]
            for override in config_overrides(workdir, model, reasoning_effort):
                args.extend(("-c", override))
            args.append("-")
            response = self._run(args, cwd=workdir, input=prompt)
            result = parse_cli_result(response)
        return {**result, "provider": "openai", "transport": "codex_cli_chatgpt_subscription",
                "requested_model": model, "requested_reasoning_effort": reasoning_effort,
                "cli_version": readiness["cli_version"],
                "rubric_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "query_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "label_schema_sha256": hashlib.sha256(canonical(LABEL_SCHEMA)).hexdigest()}


# A short alias for callers that keep provider-specific imports symmetric.
CodexSubscriptionAdapter = SubscriptionCodexAdapter


def probe():
    return SubscriptionCodexAdapter().probe()


def evaluate(query, image_bytes, mime, model, reasoning_effort):
    return SubscriptionCodexAdapter().evaluate(query, image_bytes, mime, model, reasoning_effort)
