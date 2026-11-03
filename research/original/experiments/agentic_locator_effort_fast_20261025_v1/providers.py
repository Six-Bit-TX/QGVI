"""Exact-model, zero-fallback JSON adapters for the versioned v2 controller.

Discovery never starts a model turn. Each request performs one transport call;
the experiment controller owns durable budgets, provider locks and resumption.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import jsonschema

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODEX_ADAPTER = ROOT / "experiments/dual_model_review_20261020_v1/subscription_codex.py"
GPT_MODEL = "gpt-6.1-sol"
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_RESPONSES_ENDPOINT = "https://api.deepseek.com/responses"
GPT_EFFORTS = frozenset({'low', 'medium', 'high', 'xhigh', 'max'})
DEEPSEEK_EFFORTS = frozenset({'none', 'low', 'high', 'max'})
DEFAULT_TRANSPORTS = {'gpt': 'codex_cli_chatgpt_subscription',
                      'deepseek': 'official_responses_api'}
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_USER_PAYLOAD_CHARS = 64_000
MODEL_OUTPUT_ERRORS = frozenset({'invalid_model_json', 'invalid_model_schema',
    'model_output_truncated', 'model_refusal', 'model_unexpected_tool_call'})
FAST_OVERRIDES = ('service_tier="priority"', 'features.fast_mode=true')
LOCAL_READINESS_COMMANDS = (
    ('--version',), ('--help',), ('exec', '--help'), ('features', 'list'),
    ('-c', 'forced_login_method="chatgpt"', 'login', 'status'),
)


def fast_request_args(argv):
    """Pass local probes untouched; require explicit Fast on isolated inference."""
    if not argv: raise ValueError('Expected Codex executable')
    if tuple(argv[1:]) in LOCAL_READINESS_COMMANDS: return list(argv)
    if argv[1:3] != ['--no-daemon', 'exec'] or argv[-1] != '-':
        raise ValueError('Unsupported Fast subprocess route')
    if '--strict-config' not in argv or '--ignore-user-config' not in argv:
        raise ValueError('Fast inference requires strict isolated configuration')
    if any(arg.startswith(('service_tier=', 'features.fast_mode=')) for arg in argv):
        raise ValueError('Ambiguous preexisting Fast configuration')
    return list(argv[:-1]) + [arg for item in FAST_OVERRIDES for arg in ('-c', item)] + ['-']


def require_fast_config(config):
    if config.get('service_tier', {}).get('gpt') != 'priority' or config.get('fast_mode') is not True:
        raise ValueError('This fork requires explicit GPT priority service_tier and fast_mode=true')
    return 'priority'


class ProviderError(RuntimeError):
    def __init__(self, kind, safe_details=None):
        self.kind = kind
        self.safe_details = safe_details or {}
        super().__init__(kind)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = item
        return result
    def invalid(_):
        raise ValueError("nonfinite number")
    try:
        result = json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)
        canonical(result)  # Also rejects overflowing exponent literals.
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ProviderError("invalid_json_response") from None


def content_details(value):
    raw = value.encode('utf-8', errors='replace') if isinstance(value, str) else b''
    return {'parse_stage': 'model_content', 'response_bytes': len(raw),
            'response_sha256': sha(raw), 'response_type': type(value).__name__}


def parse_model_content(value, schema, context=None):
    details = {**(context or {}), **content_details(value)}
    try:
        parsed = strict_json(value)
    except ProviderError:
        raise ProviderError('invalid_model_json', details) from None
    try:
        return validate_response(parsed, schema)
    except ProviderError as exc:
        raise ProviderError('invalid_model_schema', {**details, **exc.safe_details}) from None


def codex_module():
    # Separate module globals per provider invocation: no patch to the frozen
    # historical adapter or another concurrently running experiment.
    parent = str(CODEX_ADAPTER.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("v2_private_codex_adapter", CODEX_ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_input(path):
    data = Path(path).read_bytes()
    if not 0 < len(data) <= 20 * 1024 * 1024:
        raise ValueError("Image must contain at most 20 MiB")
    if data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    else:
        raise ValueError("Only PNG/JPEG inputs are accepted")
    return data, mime


def validate_request(system_prompt, user_payload, output_schema):
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("Explicit system prompt required")
    payload = user_payload if isinstance(user_payload, str) else canonical(user_payload)
    if not payload.strip() or len(payload) > MAX_USER_PAYLOAD_CHARS or "\x00" in payload:
        raise ValueError("Invalid user payload")
    jsonschema.Draft202012Validator.check_schema(output_schema)
    return payload


def validate_response(value, schema):
    try:
        jsonschema.Draft202012Validator(schema).validate(value)
    except jsonschema.ValidationError as exc:
        # Record only host-schema paths and categories, never response prose,
        # unknown property names, refusal text, credentials, or model reasoning.
        details = {"schema_validator": exc.validator,
                   "schema_path": list(exc.absolute_schema_path),
                   "response_type": type(value).__name__,
                   "response_sha256": sha(canonical(value).encode())}
        if exc.validator == "required" and isinstance(exc.instance, dict):
            details["missing_required_fields"] = [key for key in exc.validator_value if key not in exc.instance]
        raise ProviderError("invalid_response_schema", details) from None
    return value


def schema_system_prompt(system_prompt, output_schema):
    # JSON-object mode alone does not tell a model the fields to return.
    # Deliver the exact same explicit contract to both transports, in addition
    # to Codex's native output-schema constraint and strict local validation.
    return (system_prompt + "\n\nAuthoritative output JSON Schema follows. Return exactly one JSON object that validates against this schema; do not include Markdown fences or extra fields.\n"
            + canonical(output_schema))


def metadata(image, system_prompt, payload, schema):
    return {"input_image_sha256": sha(image), "system_prompt_sha256": sha(system_prompt.encode()),
            "user_payload_sha256": sha(payload.encode()), "output_schema_sha256": sha(canonical(schema).encode()),
            "prompt_sha256": sha(canonical({"system": system_prompt, "user": payload}).encode()),
            "automatic_retries": 0, "fallback_used": False}


def discover_codex(*, executable="codex", timeout=35, reasoning_effort='high'):
    validate_effort('gpt', reasoning_effort)
    module = codex_module()
    readiness = module.SubscriptionCodexAdapter(executable=executable).probe()
    if not readiness["ready"]:
        raise ProviderError("local_subscription_readiness_failed", readiness)
    with tempfile.TemporaryDirectory(prefix="comparison-model-discovery-", dir="/tmp") as folder:
        process = subprocess.Popen([executable, "--no-daemon", "app-server", "--stdio", "-c",
                                    "analytics.enabled=false", "-c", 'forced_login_method="chatgpt"'],
                                   cwd=folder, env=module.sanitized_environment(), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                   start_new_session=True)
        def send(value):
            process.stdin.write(json.dumps(value) + "\n"); process.stdin.flush()
        def receive(ident):
            deadline = time.monotonic() + timeout
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while time.monotonic() < deadline:
                    if not selector.select(max(0, deadline - time.monotonic())):
                        break
                    line = process.stdout.readline(MAX_RESPONSE_BYTES + 1)
                    if not line or len(line) > MAX_RESPONSE_BYTES:
                        break
                    value = strict_json(line)
                    if value.get("id") == ident:
                        if "error" in value:
                            raise ProviderError("model_discovery_rpc_failed")
                        return value["result"]
            raise ProviderError("model_discovery_timeout")
        try:
            send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "clientInfo": {"name": "comparison-preflight", "version": "1.0"},
                "capabilities": {"experimentalApi": True}}})
            receive(1)
            send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
            send({"jsonrpc": "2.0", "id": 2, "method": "model/list", "params": {"includeHidden": True, "limit": 100}})
            result = receive(2)
            if result.get("nextCursor"):
                raise ProviderError("incomplete_model_catalog")
            rows = [{k: r[k] for k in ("id", "model", "displayName", "hidden", "inputModalities",
                                      "supportedReasoningEfforts", "defaultReasoningEffort") if k in r}
                    for r in result.get("data", [])]
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            process.stdin.close(); process.stdout.close()
    selected = next((r for r in rows if r.get("model") == GPT_MODEL), None)
    if selected is None or "image" not in selected.get("inputModalities", []):
        raise ProviderError("requested_model_unavailable")
    efforts = selected.get("supportedReasoningEfforts", [])
    supported = {item.get("reasoningEffort") if isinstance(item, dict) else item for item in efforts}
    if reasoning_effort not in supported:
        raise ProviderError("requested_reasoning_effort_unavailable")
    return {"provider": "openai", "transport": "codex_cli_chatgpt_subscription", "requested_model": GPT_MODEL,
            "requested_reasoning_effort": reasoning_effort, "image_input_catalog_verified": True,
            "available_models": rows, "selected_model": selected, "readiness": readiness,
            "inference_calls": 0, "model_catalog_only_not_inference_validation": True}


def deepseek_key():
    package = str(ROOT / "qgvi-rag")
    if package not in sys.path:
        sys.path.insert(0, package)
    from caption_rag.credentials import load_deepseek_key
    return load_deepseek_key(Path("/home/xiongti1/others/.env"))


def http_json(request, opener, timeout):
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderError("response_too_large")
        try:
            return strict_json(raw)
        except ProviderError:
            raise ProviderError('invalid_transport_json', {'parse_stage': 'http_envelope',
                'response_bytes': len(raw), 'response_sha256': sha(raw)}) from None
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise ProviderError("http_" + str(status)) from None
    except (urllib.error.URLError, OSError, TimeoutError):
        raise ProviderError("network_or_timeout") from None


def discover_deepseek(*, key_loader=None, opener=None, reasoning_effort='high', transport='official_responses_api'):
    validate_effort('deepseek', reasoning_effort)
    if transport not in {'official_responses_api', 'official_chat_completions_api'}:
        raise ValueError('Unsupported DeepSeek transport')
    key = (key_loader or deepseek_key)()
    request = urllib.request.Request("https://api.deepseek.com/models", headers={"Authorization": "Bearer " + key}, method="GET")
    data = http_json(request, opener or urllib.request.build_opener(NoRedirect()), 30)
    ids = [r.get("id") for r in data.get("data", []) if isinstance(r.get("id"), str)]
    if DEEPSEEK_MODEL not in ids:
        raise ProviderError("requested_model_unavailable")
    return {"provider": "deepseek", "transport": transport,
            "requested_model": DEEPSEEK_MODEL, "available_model_ids": ids, "inference_calls": 0,
            "immutable_v4_1_backend_verified": False,
            "documented_backend_version": "DeepSeek-V4.1-Flash-1017",
            "documentation_checked_at": "2026-10-25",
            "documentation_url": "https://api-docs.deepseek.com/quick_start/pricing/",
            "documented_image_input": True, "requested_reasoning_effort": reasoning_effort,
            "documented_reasoning_efforts": sorted(DEEPSEEK_EFFORTS),
            "disclosure": "Official documentation maps deepseek-flash to DeepSeek-V4.1-Flash-1017 as of 2026-10-25; the request alias is mutable and this discovery performs no inference."}


class CodexProvider:
    def __init__(self, *, model=GPT_MODEL, reasoning_effort="high", timeout=600, runner=None, environment=None,
                 service_tier='priority', fast_mode=True):
        if model != GPT_MODEL:
            raise ValueError("Only the explicitly requested gpt-6.1-sol is permitted")
        validate_effort('gpt', reasoning_effort)
        if service_tier != 'priority' or fast_mode is not True:
            raise ValueError('Fast fork permits only explicit priority with fast_mode=true; no fallback')
        self.model, self.reasoning_effort, self.timeout = model, reasoning_effort, timeout
        self.service_tier, self.fast_mode = service_tier, fast_mode
        self.transport = DEFAULT_TRANSPORTS['gpt']
        self.runner, self.environment, self.readiness = runner, environment, None

    def request(self, image_path, system_prompt, user_payload, output_schema):
        payload = validate_request(system_prompt, user_payload, output_schema)
        original_system_prompt = system_prompt
        system_prompt = schema_system_prompt(system_prompt, output_schema)
        image, mime = image_input(image_path)
        module = codex_module()
        module.PROMPT, module.LABEL_SCHEMA = system_prompt, output_schema
        module.query_prompt = lambda query: query  # Identical text sent to both routes.
        final_text = {}
        # Parse content after the historical adapter has validated the CLI
        # envelope, usage, tool isolation, and reported model metadata.
        module.parse_judgment = lambda value: final_text.update(text=value)
        # The historical adapter probes local version/help/login commands through
        # its runner before inference. Only its strict exec command receives Fast
        # overrides; readiness commands retain their exact working arguments.
        def fast_runner(argv, **kwargs):
            return (self.runner or module.bounded_run)(fast_request_args(argv), **kwargs)
        adapter = module.SubscriptionCodexAdapter(timeout=self.timeout, runner=fast_runner, environment=self.environment)
        if self.readiness is not None:
            adapter._readiness = dict(self.readiness)
        start = time.monotonic()
        try:
            result = adapter.evaluate(payload, image, mime, self.model, self.reasoning_effort)
        except module.SubscriptionCodexError as exc:
            raise ProviderError(exc.kind, {**exc.safe_details, 'requested_service_tier': self.service_tier,
                'served_service_tier': None, 'fast_mode_enabled': True}) from None
        self.readiness = adapter._readiness
        reported = result.get("reported_model")
        if reported is not None and reported != self.model:
            raise ProviderError("reported_model_mismatch")
        context = {'requested_model': self.model, 'reported_model': reported,
                   'usage': result.get('usage'), 'reasoning_effort': self.reasoning_effort,
                   'requested_service_tier': self.service_tier, 'served_service_tier': None,
                   'fast_mode_enabled': True}
        value = parse_model_content(final_text.get('text'), output_schema, context)
        return {"response": value, "provider": "openai", "transport": result["transport"],
                "requested_model": self.model, "reported_model": reported, "reasoning_effort": self.reasoning_effort,
                "wall_seconds": time.monotonic() - start, "usage": result.get("usage"),
                "cli_version": result.get("cli_version"), "tool_calls": 0,
                "model_metadata_source": result.get("model_metadata_source"),
                "requested_service_tier": self.service_tier, "served_service_tier": None,
                "fast_mode_enabled": True, "service_tier_display_name": "Fast",
                "tier_evidence": "Explicit per-request CLI flags; served tier is not exposed by this adapter.",
                "schema_in_system_prompt": True, "original_system_prompt_sha256": sha(original_system_prompt.encode()),
                **metadata(image, system_prompt, payload, output_schema)}


class DeepSeekProvider:
    """Historical JSON-object route, selectable explicitly; never a fallback."""
    def __init__(self, *, model=DEEPSEEK_MODEL, reasoning_effort='high', timeout=600, key_loader=None, opener=None):
        if model != DEEPSEEK_MODEL:
            raise ValueError("Only the discovered deepseek-flash route is permitted")
        self.model, self.timeout = model, timeout
        validate_effort('deepseek', reasoning_effort)
        self.reasoning_effort = reasoning_effort
        self.transport = 'official_chat_completions_api'
        self.key_loader, self.opener = key_loader or deepseek_key, opener or urllib.request.build_opener(NoRedirect())

    def request(self, image_path, system_prompt, user_payload, output_schema):
        payload = validate_request(system_prompt, user_payload, output_schema)
        original_system_prompt = system_prompt
        system_prompt = schema_system_prompt(system_prompt, output_schema)
        image, mime = image_input(image_path)
        body = {"model": self.model, "messages": [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": [{"type": "text", "text": payload},
                  {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(image).decode(), "detail": "original"}}]}],
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled" if self.reasoning_effort == 'none' else "enabled"},
                "max_tokens": 65536, "stream": False}
        if self.reasoning_effort != 'none':
            body['reasoning_effort'] = self.reasoning_effort
        encoded = canonical(body).encode()
        if len(encoded) > 48 * 1024 * 1024:
            raise ValueError("Request too large")
        request = urllib.request.Request(DEEPSEEK_ENDPOINT, data=encoded, method="POST",
            headers={"Authorization": "Bearer " + self.key_loader(), "Content-Type": "application/json", "Accept": "application/json"})
        start = time.monotonic()
        envelope = http_json(request, self.opener, self.timeout)
        try:
            choices = envelope["choices"]
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ValueError()
            message = choices[0]["message"]
            finish = choices[0]['finish_reason']
            if not isinstance(message, dict) or not isinstance(finish, str):
                raise ValueError()
        except (KeyError, TypeError, ValueError, IndexError):
            raise ProviderError("invalid_response_envelope") from None
        reported = envelope.get("model")
        if reported is not None and (not isinstance(reported, str) or len(reported) > 200):
            raise ProviderError("invalid_model_metadata")
        if reported is not None and reported != self.model:
            raise ProviderError('reported_model_mismatch')
        usage = envelope.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise ProviderError("invalid_usage_metadata")
            allowed = ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
            usage = {k: usage[k] for k in allowed if k in usage}
            if any(type(v) is not int or v < 0 for v in usage.values()):
                raise ProviderError("invalid_usage_metadata")
        context = {'requested_model': self.model, 'reported_model': reported, 'usage': usage,
                   'reasoning_effort': self.reasoning_effort, 'finish_reason': finish}
        details = {**context, **content_details(message.get('content'))}
        if finish == 'length':
            raise ProviderError('model_output_truncated', details)
        if finish == 'content_filter' or message.get('refusal'):
            raise ProviderError('model_refusal', details)
        if message.get('tool_calls') or message.get('function_call') or finish in {'tool_calls', 'function_call'}:
            raise ProviderError('model_unexpected_tool_call', details)
        if finish != 'stop' or 'content' not in message:
            raise ProviderError('invalid_response_envelope')
        value = parse_model_content(message['content'], output_schema, context)
        return {"response": value, "provider": "deepseek", "transport": "official_chat_completions_api",
                "requested_model": self.model, "reported_model": reported, "reasoning_effort": self.reasoning_effort,
                "thinking": "disabled" if self.reasoning_effort == 'none' else "enabled", "max_tokens": 65536,
                "wall_seconds": time.monotonic() - start, "usage": usage, "tool_calls": 0,
                "immutable_v4_1_backend_verified": False, "schema_in_system_prompt": True, "original_system_prompt_sha256": sha(original_system_prompt.encode()),
                **metadata(image, system_prompt, payload, output_schema)}


def responses_usage(value):
    if not isinstance(value, dict):
        raise ProviderError('invalid_usage_metadata')
    result = {}
    for key in ('input_tokens', 'output_tokens', 'total_tokens'):
        number = value.get(key)
        if type(number) is not int or number < 0:
            raise ProviderError('invalid_usage_metadata')
        result[key] = number
    for group, key, maximum in [('input_tokens_details', 'cached_tokens', result['input_tokens']),
                               ('output_tokens_details', 'reasoning_tokens', result['output_tokens'])]:
        details = value.get(group)
        if details is None:
            continue
        if not isinstance(details, dict):
            raise ProviderError('invalid_usage_metadata')
        number = details.get(key)
        if number is not None:
            if type(number) is not int or not 0 <= number <= maximum:
                raise ProviderError('invalid_usage_metadata')
            result[group] = {key: number}
    return result


class DeepSeekResponsesProvider:
    """Documented image + JSON-schema route. Qualification is recorded outside.

    An HTTP request is made exactly once. No repair calls, transport switching,
    schema relaxation, effort aliases, or hidden model fallbacks are allowed.
    """
    def __init__(self, *, model=DEEPSEEK_MODEL, reasoning_effort='high', timeout=600,
                 key_loader=None, opener=None):
        if model != DEEPSEEK_MODEL:
            raise ValueError('Only the discovered deepseek-flash route is permitted')
        validate_effort('deepseek', reasoning_effort)
        self.model, self.reasoning_effort, self.timeout = model, reasoning_effort, timeout
        self.transport = 'official_responses_api'
        self.key_loader = key_loader or deepseek_key
        self.opener = opener or urllib.request.build_opener(NoRedirect())

    def request(self, image_path, system_prompt, user_payload, output_schema):
        payload = validate_request(system_prompt, user_payload, output_schema)
        original_system_prompt = system_prompt
        system_prompt = schema_system_prompt(system_prompt, output_schema)
        image, mime = image_input(image_path)
        body = {'model': self.model, 'instructions': system_prompt,
            'input': [{'role': 'user', 'content': [
                {'type': 'input_text', 'text': payload},
                {'type': 'input_image', 'image_url': 'data:'+mime+';base64,'+base64.b64encode(image).decode(),
                 'detail': 'original'}]}],
            'reasoning': {'effort': self.reasoning_effort}, 'max_output_tokens': 65536, 'stream': False,
            'text': {'format': {'type': 'json_schema', 'name': 'qgvi_locator_action', 'schema': output_schema}}}
        encoded = canonical(body).encode()
        if len(encoded) > 48 * 1024 * 1024:
            raise ValueError('Request too large')
        request = urllib.request.Request(DEEPSEEK_RESPONSES_ENDPOINT, data=encoded, method='POST',
            headers={'Authorization': 'Bearer '+self.key_loader(), 'Content-Type': 'application/json', 'Accept': 'application/json'})
        started = time.monotonic()
        envelope = http_json(request, self.opener, self.timeout)
        answer, usage = self.parse_response(envelope, output_schema)
        return {'response': answer, 'provider': 'deepseek', 'transport': self.transport,
            'requested_model': self.model, 'reported_model': envelope['model'],
            'reasoning_effort': self.reasoning_effort,
            'thinking': 'disabled' if self.reasoning_effort == 'none' else 'enabled',
            'wall_seconds': time.monotonic()-started, 'usage': usage, 'tool_calls': 0,
            'max_output_tokens': 65536, 'server_schema_requested': True,
            'immutable_v4_1_backend_verified': False, 'schema_in_system_prompt': True,
            'original_system_prompt_sha256': sha(original_system_prompt.encode()),
            'documentation_url': 'https://api-docs.deepseek.com/api/create-response/',
            **metadata(image, system_prompt, payload, output_schema)}

    def parse_response(self, envelope, schema):
        if not isinstance(envelope, dict) or envelope.get('object') != 'response':
            raise ProviderError('invalid_response_envelope')
        if envelope.get('model') != self.model:
            raise ProviderError('reported_model_mismatch')
        usage = responses_usage(envelope.get('usage'))
        context = {'requested_model': self.model, 'reported_model': envelope['model'],
            'reasoning_effort': self.reasoning_effort, 'usage': usage,
            'response_status': envelope.get('status'), 'transport': self.transport}
        # DeepSeek may include its normalized request settings in the envelope.
        # When present, reject a different effort rather than mislabel the run.
        reported_reasoning = envelope.get('reasoning')
        if reported_reasoning is not None:
            if not isinstance(reported_reasoning, dict):
                raise ProviderError('invalid_reasoning_metadata')
            if reported_reasoning.get('effort') not in {None, self.reasoning_effort}:
                raise ProviderError('reported_reasoning_effort_mismatch', context)
        if envelope.get('status') == 'incomplete':
            details = envelope.get('incomplete_details')
            reason = details.get('reason') if isinstance(details, dict) else None
            if reason == 'max_output_tokens':
                raise ProviderError('model_output_truncated', context)
            if reason == 'content_filter':
                raise ProviderError('model_refusal', context)
            raise ProviderError('invalid_response_envelope')
        if envelope.get('status') == 'failed':
            raise ProviderError('responses_provider_failed')
        if envelope.get('status') != 'completed' or envelope.get('error') is not None:
            raise ProviderError('invalid_response_envelope')
        outputs = envelope.get('output')
        if not isinstance(outputs, list):
            raise ProviderError('invalid_response_envelope')
        messages = []
        for item in outputs:
            if not isinstance(item, dict):
                raise ProviderError('invalid_response_envelope')
            if item.get('type') == 'reasoning':
                continue  # Never parse or persist hidden reasoning as an answer.
            if item.get('type') != 'message':
                raise ProviderError('model_unexpected_tool_call', context)
            messages.append(item)
        if len(messages) != 1:
            raise ProviderError('invalid_response_envelope')
        message = messages[0]
        if message.get('role') != 'assistant' or message.get('status') != 'completed':
            raise ProviderError('invalid_response_envelope')
        parts = message.get('content')
        if not isinstance(parts, list):
            raise ProviderError('invalid_response_envelope')
        chunks = []
        for part in parts:
            if not isinstance(part, dict):
                raise ProviderError('invalid_response_envelope')
            if part.get('type') == 'refusal':
                raise ProviderError('model_refusal', context)
            if part.get('type') != 'output_text' or not isinstance(part.get('text'), str):
                raise ProviderError('invalid_response_envelope')
            chunks.append(part['text'])
        return parse_model_content(''.join(chunks), schema, context), usage


GPTProvider = CodexProvider


def validate_effort(name, reasoning_effort):
    allowed = GPT_EFFORTS if name == 'gpt' else DEEPSEEK_EFFORTS if name == 'deepseek' else frozenset()
    if reasoning_effort not in allowed:
        raise ValueError('Unsupported explicit reasoning effort; effort aliases are not silently remapped')
    return reasoning_effort


def make_provider(name, *, reasoning_effort='high', transport=None, **kwargs):
    if name in {"gpt", "openai", GPT_MODEL}:
        if transport not in {None, DEFAULT_TRANSPORTS['gpt']}:
            raise ValueError('Unsupported GPT transport')
        return CodexProvider(reasoning_effort=reasoning_effort, **kwargs)
    if name in {"deepseek", DEEPSEEK_MODEL}:
        if transport in {None, DEFAULT_TRANSPORTS['deepseek']}:
            return DeepSeekResponsesProvider(reasoning_effort=reasoning_effort, **kwargs)
        if transport == 'official_chat_completions_api':
            return DeepSeekProvider(reasoning_effort=reasoning_effort, **kwargs)
        raise ValueError('Unsupported DeepSeek transport')
    raise ValueError("Unknown comparison provider; no automatic fallback")
