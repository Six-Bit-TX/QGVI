"""Synthetic credential fixtures only; never read user credentials or call APIs."""

import json
import os

import pytest

from caption_rag.credentials import CredentialError, load_deepseek_key


KEY = "sk-syntheticDeepSeekKey_1234567890"
OTHER_KEY = "sk-syntheticOtherProvider_1234567890"


def read_fixture(tmp_path, text):
    path = tmp_path / "synthetic.env"
    path.write_text(text, encoding="utf-8")
    return load_deepseek_key(path)


@pytest.mark.parametrize(
    "text",
    [
        f"DEEPSEEK_API_KEY={KEY}\n",
        f" export DEEPSEEK_API_KEY = '{KEY}'\n",
        f'DEEPSEEK_API_KEY="{KEY}" # example\n',
        f"DEEPSEEK_API_KEY={KEY} # example\n",
        f"DeepSeek: {KEY}\n",
        f"deepseek = {KEY}\n",
        f"DeepSeek {KEY}\n",
        f"DeepSeek\n{KEY}\n",
        f"DeepSeek API Key:\n\n# example\n{KEY}\n",
        f"DeepSeek V4 Flash： {KEY}\n",
        f"DeepSeek:\n'{KEY}'\n",
        f"\ufeffDEEPSEEK_API_KEY={KEY}\n",
    ],
)
def test_supported_text_formats(tmp_path, text):
    assert read_fixture(tmp_path, text) == KEY


@pytest.mark.parametrize(
    "data",
    [
        {"DEEPSEEK_API_KEY": KEY},
        {"deepseek_api_key": KEY},
        {"deepseek": {"api_key": KEY}},
        {"deepseek": {"deepseek_api_key": KEY}},
        {"deepseek": {"DEEPSEEK_API_KEY": KEY}},
        {"openai": {"api_key": OTHER_KEY}, "deepseek": {"api_key": KEY}},
    ],
)
def test_supported_explicit_json_formats(tmp_path, data):
    assert read_fixture(tmp_path, json.dumps(data)) == KEY


def test_labeled_multi_provider_file_selects_only_deepseek(tmp_path):
    text = f"OpenAI\n{OTHER_KEY}\nDeepSeek\n{KEY}\nAnother Provider\n{OTHER_KEY}\n"
    assert read_fixture(tmp_path, text) == KEY


@pytest.mark.parametrize("heading", ["# DeepSeek", "# DeepSeek API Key", "# Deepseek API", "# deepseek:", "# DeepSeek V4 Flash:"])
def test_comment_style_provider_heading(tmp_path, heading):
    text = f"# OpenAI\n{OTHER_KEY}\n{heading}\n\n{KEY}\n# Another Provider\n{OTHER_KEY}\n"
    assert read_fixture(tmp_path, text) == KEY


@pytest.mark.parametrize(
    "comment",
    [
        f"# DEEPSEEK_API_KEY={OTHER_KEY}",
        f'# export DEEPSEEK_API_KEY="{OTHER_KEY}"',
        "# DEEPSEEK_API_KEY=",
        "# Use DeepSeek for caption generation",
        "# DeepSeek usage instructions",
        "# DeepSeek API key should be set elsewhere",
    ],
)
def test_comment_mentions_and_dotenv_assignments_remain_ignored(tmp_path, comment):
    with pytest.raises(CredentialError, match="No explicit DeepSeek"):
        read_fixture(tmp_path, f"{comment}\n{OTHER_KEY}\n")
    assert read_fixture(tmp_path, f"{comment}\nDEEPSEEK_API_KEY={KEY}\n") == KEY


@pytest.mark.parametrize("text", [f"# DeepSeek\n# OpenAI\n{OTHER_KEY}\n", "# DeepSeek\n", f"# DeepSeek\n{KEY}\n# DeepSeek\n{OTHER_KEY}\n"])
def test_comment_heading_requires_its_own_unambiguous_key(tmp_path, text):
    with pytest.raises(CredentialError):
        read_fixture(tmp_path, text)


def test_parser_does_not_populate_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    before = dict(os.environ)
    assert read_fixture(tmp_path, f"OPENAI_API_KEY={OTHER_KEY}\nDEEPSEEK_API_KEY={KEY}\n") == KEY
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "text",
    [
        "",
        f"{OTHER_KEY}\n",
        f"OPENAI_API_KEY={OTHER_KEY}\n",
        f"OpenAI\n{OTHER_KEY}\n",
        f"# DEEPSEEK_API_KEY={KEY}\n",
        f"MY_DEEPSEEK_API_KEY={KEY}\n",
        f"deepseekish: {OTHER_KEY}\n",
        json.dumps({"api_key": OTHER_KEY}),
        json.dumps({"openai": {"api_key": OTHER_KEY}}),
        json.dumps({"other": {"DEEPSEEK_API_KEY": OTHER_KEY}}),
    ],
)
def test_does_not_fall_back_to_unscoped_or_non_deepseek_keys(tmp_path, text):
    with pytest.raises(CredentialError, match="No explicit DeepSeek"):
        read_fixture(tmp_path, text)


@pytest.mark.parametrize(
    "text",
    [
        f"DEEPSEEK_API_KEY={KEY}\nDEEPSEEK_API_KEY={OTHER_KEY}\n",
        f"DEEPSEEK_API_KEY={KEY}\nDeepSeek: {KEY}\n",
        json.dumps({"DEEPSEEK_API_KEY": KEY, "deepseek": {"api_key": OTHER_KEY}}),
        json.dumps({"deepseek": {"api_key": KEY, "deepseek_api_key": KEY}}),
        f'{{"DEEPSEEK_API_KEY":"{KEY}","DEEPSEEK_API_KEY":"{OTHER_KEY}"}}',
    ],
)
def test_ambiguous_and_duplicate_declarations_are_rejected(tmp_path, text):
    with pytest.raises(CredentialError):
        read_fixture(tmp_path, text)


@pytest.mark.parametrize(
    "text",
    [
        "DEEPSEEK_API_KEY=sk-short",
        "DEEPSEEK_API_KEY=",
        f"DEEPSEEK_API_KEY='{KEY}",
        f"DEEPSEEK_API_KEY={KEY}; echo unsafe",
        f'DEEPSEEK_API_KEY="{KEY}" trailing-text',
        f"DEEPSEEK_API_KEY={KEY}#not-a-separated-comment",
        "DeepSeek:\n",
        f"DeepSeek:\nOpenAI:\n{OTHER_KEY}",
        '{"DEEPSEEK_API_KEY": ',
        json.dumps({"DEEPSEEK_API_KEY": None}),
        json.dumps({"deepseek": KEY}),
        json.dumps({"deepseek": {"api_key": [KEY]}}),
        json.dumps([{"DEEPSEEK_API_KEY": KEY}]),
    ],
)
def test_malformed_explicit_values_are_rejected_without_contents(tmp_path, text):
    with pytest.raises(CredentialError) as captured:
        read_fixture(tmp_path, text)
    assert KEY not in str(captured.value)
    assert OTHER_KEY not in str(captured.value)
    assert "sk-short" not in str(captured.value)


@pytest.mark.parametrize("template", ["$(touch {path})", "`touch {path}`", "${{DEEPSEEK_API_KEY}}"])
def test_shell_substitution_and_interpolation_are_rejected(tmp_path, template):
    sentinel = tmp_path / "must-not-exist"
    value = template.format(path=sentinel)
    with pytest.raises(CredentialError):
        read_fixture(tmp_path, f'DEEPSEEK_API_KEY="{value}"\n')
    assert not sentinel.exists()


def test_file_read_error_does_not_leak_filename(tmp_path):
    with pytest.raises(CredentialError) as captured:
        load_deepseek_key(tmp_path / KEY)
    assert KEY not in str(captured.value)


def test_non_utf8_error_does_not_leak_contents(tmp_path):
    path = tmp_path / "invalid.env"
    path.write_bytes(KEY.encode() + b"\xff")
    with pytest.raises(CredentialError, match="UTF-8") as captured:
        load_deepseek_key(path)
    assert KEY not in str(captured.value)
