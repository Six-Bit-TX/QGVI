"""Read an explicitly identified DeepSeek key without executing an env file.

Only explicit DeepSeek fields/provider labels are considered. No values are
expanded, no shell is invoked, and the process environment is never changed.
Errors intentionally omit file contents, values, and parser exception details.
"""

from __future__ import annotations

import json
from pathlib import Path
import re


class CredentialError(ValueError):
    """A credential file could not provide one unambiguous DeepSeek key."""


_KEY = re.compile(r"sk-[A-Za-z0-9_-]{16,}\Z")
_DOTENV = re.compile(r"^(?:export[ \t]+)?DEEPSEEK_API_KEY[ \t]*=(.*)$")
_LABEL = re.compile(
    r"^deepseek(?:[ \t_-]+(?:api(?:[ \t_-]*key)?|key|v4(?:[ \t_-]+flash)?|flash))?"
    r"(?=$|[ \t:=：])[ \t]*(?:[:=：][ \t]*)?(.*)$",
    flags=re.IGNORECASE,
)
_EXPLICIT_FIELDS = ("DEEPSEEK_API_KEY", "deepseek_api_key")
_NESTED_FIELDS = (*_EXPLICIT_FIELDS, "api_key")


def _valid_key(value: object) -> str:
    if not isinstance(value, str) or _KEY.fullmatch(value) is None:
        raise CredentialError("The explicit DeepSeek credential is malformed")
    return value


def _text_value(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith(("'", '"')):
        quote = raw[0]
        end = raw.find(quote, 1)
        if end == -1:
            raise CredentialError("The explicit DeepSeek credential is malformed")
        value, remainder = raw[1:end], raw[end + 1 :].strip()
        if remainder and not remainder.startswith("#"):
            raise CredentialError("The explicit DeepSeek credential is malformed")
    else:
        # An unquoted comment must follow whitespace. This also rejects shell
        # substitutions, command tails, interpolation, and escaped values.
        value = re.split(r"[ \t]+#", raw, maxsplit=1)[0].strip()
    return _valid_key(value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CredentialError("Credential JSON contains duplicate field names")
        result[key] = value
    return result


def _json_keys(text: str) -> list[str]:
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError):
        raise CredentialError("Credential JSON is malformed or has duplicate fields") from None
    if not isinstance(data, dict):
        raise CredentialError("Credential JSON must be an object with an explicit DeepSeek field")
    values = [_valid_key(data[name]) for name in _EXPLICIT_FIELDS if name in data]
    if "deepseek" in data:
        provider = data["deepseek"]
        if not isinstance(provider, dict):
            raise CredentialError("The DeepSeek credential object is malformed")
        values.extend(_valid_key(provider[name]) for name in _NESTED_FIELDS if name in provider)
    return values


def _label_or_dotenv_keys(text: str) -> list[str]:
    values = []
    awaiting_key = False
    awaiting_comment_heading = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if awaiting_comment_heading:
            # The next nonempty line must be the key itself. In particular,
            # never skip another provider heading and select that vendor's key.
            values.append(_text_value(line))
            awaiting_key = awaiting_comment_heading = False
            continue
        if line.startswith("#"):
            heading = line[1:].strip()
            label = _LABEL.fullmatch(heading)
            if label and not label.group(1).strip() and "=" not in heading:
                if awaiting_key:
                    raise CredentialError("The DeepSeek provider label has no following credential")
                awaiting_key = awaiting_comment_heading = True
            continue
        if awaiting_key:
            values.append(_text_value(line))
            awaiting_key = False
            continue
        assignment = _DOTENV.fullmatch(line)
        if assignment:
            values.append(_text_value(assignment.group(1)))
            continue
        label = _LABEL.fullmatch(line)
        if label:
            tail = label.group(1).strip()
            if not tail or tail.startswith("#"):
                awaiting_key = True
            else:
                values.append(_text_value(tail))
    if awaiting_key:
        raise CredentialError("The DeepSeek provider label has no following credential")
    return values


def load_deepseek_key(path: str | Path) -> str:
    """Return exactly one explicit DeepSeek key from a local credential file.

    Supported formats are dotenv ``DEEPSEEK_API_KEY=...``, an explicit JSON
    field (including ``deepseek.api_key``), or a DeepSeek provider label with
    its key on the same/next nonempty line. Exact ``# DeepSeek`` comment headings
    are supported when the next nonempty line is a key; commented assignments
    and comments merely mentioning DeepSeek are ignored. Duplicates are rejected
    even when their values agree; unrelated providers are never selected.
    """
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError, ValueError, TypeError):
        raise CredentialError("Cannot read the credential file as UTF-8 text") from None
    values = _json_keys(text) if text.lstrip().startswith(("{", "[")) else _label_or_dotenv_keys(text)
    if not values:
        raise CredentialError("No explicit DeepSeek credential was found in the credential file")
    if len(values) != 1:
        raise CredentialError("Multiple explicit DeepSeek credentials were found; keep exactly one")
    return values[0]
