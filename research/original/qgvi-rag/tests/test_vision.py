"""Offline contract tests: no network, model downloads, or API charges."""

import base64
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from caption_rag.vision import (
    CAPTION_PROMPT, VERIFY_PROMPT, DeepSeekVision, VisionError, VisionResponseError,
    _json_loads, _model_json_loads, validate_caption, validate_verification,
)


def caption_result():
    return {"summary": "A red brick building beside a road.",
            "objects": ["building", "road"], "attributes": ["red brick facade"],
            "relations": ["building beside road"], "visible_text": [], "uncertainty": []}


def verification_result(status="supported", verdict="match", score=0.9):
    return {"verdict": verdict, "score": score, "evidence": "The facade is red brick.",
            "requirements": [{"requirement": "red brick building", "status": status,
                              "evidence": "Red brick facade is visible."}]}


def response(result=None, *, content=None, finish="stop"):
    result = caption_result() if result is None else result
    return io.BytesIO(json.dumps({"id": "request-1", "model": "deepseek-flash",
        "usage": {"prompt_tokens": 50, "completion_tokens": 30,
                  "prompt_tokens_details": {"cached_tokens": 0}},
        "choices": [{"finish_reason": finish,
                     "message": {"content": content if content is not None else json.dumps(result)}}]
    }).encode())


class DeepSeekVisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = Path(self.directory.name) / "private-camera-name.png"
        Image.new("RGB", (200, 100), color="red").save(self.image)
        self.client = DeepSeekVision(api_key="fake-unit-test-secret", max_side=128,
                                     retries=2, response_retries=0)
        self.opener = Mock()
        self.client._opener = self.opener
        self.opener.open.return_value = response()

    def sent_payload(self):
        return json.loads(self.opener.open.call_args.args[0].data)

    def test_caption_contract_and_only_pixels_are_uploaded(self):
        result = self.client.caption(self.image)
        self.assertEqual(result["summary"], caption_result()["summary"])
        self.assertEqual(result["_api"]["response_id"], "request-1")
        self.assertEqual(result["_api"]["usage"]["prompt_tokens"], 50)
        payload = self.sent_payload()
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["messages"][0]["content"], CAPTION_PROMPT)
        self.assertNotIn(str(self.image), json.dumps(payload))
        data_url = payload["messages"][1]["content"][1]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as uploaded:
            self.assertEqual(uploaded.size, (128, 64))
            self.assertEqual(uploaded.mode, "RGB")
            self.assertFalse(uploaded.getexif())

    def test_verify_uses_query_and_image_without_caption(self):
        self.opener.open.return_value = response(verification_result())
        query = "red brick building beside a road"
        result = self.client.verify(self.image, query)
        self.assertEqual(result["verdict"], "match")
        payload = self.sent_payload()
        self.assertEqual(payload["messages"][0]["content"], VERIFY_PROMPT)
        self.assertIn(query, payload["messages"][1]["content"][0]["text"])
        self.assertNotIn(CAPTION_PROMPT, json.dumps(payload))
        self.assertEqual(len(payload["messages"]), 2)

    def test_verification_fails_closed_when_model_overrules_requirements(self):
        for status, expected in [("contradicted", "no_match"), ("unclear", "uncertain")]:
            with self.subTest(status=status):
                self.opener.open.return_value = response(verification_result(status))
                result = self.client.verify(self.image, "red brick building")
                self.assertEqual(result["verdict"], expected)
                self.assertEqual(result["score"], 0.0)

    def test_nonmatching_verdict_has_zero_score(self):
        self.opener.open.return_value = response(verification_result(verdict="uncertain"))
        self.assertEqual(self.client.verify(self.image, "building")["score"], 0.0)

    def test_invalid_caption_schema_is_rejected(self):
        variants = [[], {**caption_result(), "summary": ""},
                    {**caption_result(), "objects": "building"},
                    {**caption_result(), "uncertainty": [42]}]
        for value in variants:
            with self.subTest(value=value):
                self.opener.open.return_value = response(value)
                with self.assertRaisesRegex(VisionError, "schema"):
                    self.client.caption(self.image)

    def test_live_caption_extra_observations_are_dropped_without_retry(self):
        self.client.response_retries = 2
        self.opener.open.return_value = response({**caption_result(),
            "extra_observation": "A harmless additional observation.",
            "_api": {"response_id": "untrusted model metadata"}})
        result = self.client.caption(self.image)
        self.assertEqual({key: value for key, value in result.items() if key != "_api"}, caption_result())
        self.assertEqual(set(result), set(caption_result()) | {"_api"})
        self.assertEqual(result["_api"]["response_id"], "request-1")
        self.assertEqual(self.opener.open.call_count, 1)

    def test_live_caption_missing_or_malformed_fields_still_exhaust_retry_budget(self):
        self.client.response_retries = 2
        missing = {**caption_result(), "extra_observation": "Cannot fill missing fields."}
        del missing["relations"]
        wrong_type = {**caption_result(), "objects": {"building": 1}, "extra_observation": "ignored"}
        for invalid in [missing, wrong_type]:
            with self.subTest(invalid=invalid):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = [response(invalid) for _ in range(3)]
                with self.assertRaisesRegex(VisionResponseError, "after 3 response attempts"):
                    self.client.caption(self.image)
                self.assertEqual(self.opener.open.call_count, 3)

    def test_live_verification_extra_fields_are_dropped_without_retry(self):
        self.client.response_retries = 2
        value = {**verification_result(), "extra_observation": "Additional explanation.",
                 "_api": {"response_id": "untrusted model metadata"}}
        value["requirements"][0]["confidence"] = 0.9
        self.opener.open.return_value = response(value)
        result = self.client.verify(self.image, "building")
        self.assertEqual({key: item for key, item in result.items() if key != "_api"}, verification_result())
        self.assertEqual(set(result), set(verification_result()) | {"_api"})
        self.assertEqual(set(result["requirements"][0]), {"requirement", "status", "evidence"})
        self.assertEqual(result["_api"]["response_id"], "request-1")
        self.assertEqual(self.opener.open.call_count, 1)

    def test_live_verification_required_fields_remain_strict_after_normalization(self):
        self.client.response_retries = 2
        missing_top = {**verification_result(), "extra_observation": "Cannot replace evidence."}
        del missing_top["evidence"]
        missing_nested = verification_result()
        del missing_nested["requirements"][0]["status"]
        missing_nested["requirements"][0]["confidence"] = 0.9
        bad_nested = verification_result()
        bad_nested["requirements"][0].update({"status": ["supported"], "confidence": 0.9})
        bad_score = {**verification_result(), "score": "0.9", "extra_observation": "Cannot coerce score."}
        for invalid in [missing_top, missing_nested, bad_nested, bad_score]:
            with self.subTest(invalid=invalid):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = [response(invalid) for _ in range(3)]
                with self.assertRaisesRegex(VisionResponseError, "after 3 response attempts"):
                    self.client.verify(self.image, "building")
                self.assertEqual(self.opener.open.call_count, 3)

    def test_live_verification_normalization_preserves_fail_closed_rules(self):
        for status, expected in [("contradicted", "no_match"), ("unclear", "uncertain")]:
            with self.subTest(status=status):
                value = {**verification_result(status=status), "extra_observation": "Ignored."}
                value["requirements"][0]["confidence"] = 0.9
                self.opener.open.return_value = response(value)
                result = self.client.verify(self.image, "building")
                self.assertEqual(result["verdict"], expected)
                self.assertEqual(result["score"], 0.0)

    def test_invalid_verification_schema_and_scores_are_rejected(self):
        good = verification_result()
        variants = [{**good, "score": score} for score in [-0.1, 1.1, True, "0.8", None, 10 ** 400]]
        variants += [{**good, "requirements": []}, {**good, "verdict": "probably"},
                     {**good, "requirements": [{"status": "supported"}]},
                     {**good, "requirements": [{**good["requirements"][0], "status": []}]}]
        for value in variants:
            with self.subTest(value=value):
                self.opener.open.return_value = response(value)
                with self.assertRaisesRegex(VisionError, "schema"):
                    self.client.verify(self.image, "building")

    def test_nan_and_duplicate_keys_are_rejected(self):
        for content in ['{"score": NaN}', '{"summary": "one", "summary": "two"}']:
            with self.subTest(content=content):
                self.opener.open.return_value = response(content=content)
                with self.assertRaisesRegex(VisionError, "malformed JSON"):
                    self.client.caption(self.image)

    def test_live_identical_score_duplicates_succeed_once_and_preserve_fail_closed_rules(self):
        content = json.dumps(verification_result(status="contradicted", verdict="match", score=0.0))
        content = content.replace('"score": 0.0', '"score": 0.0, "score": 0.0, "score": 0.0')
        self.opener.open.return_value = response(content=content)
        result = self.client.verify(self.image, "red brick building")
        self.assertEqual(result["verdict"], "no_match")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["_api"]["normalized_duplicate_keys"], 2)
        self.assertEqual(validate_verification(result)["_api"]["normalized_duplicate_keys"], 2)
        self.opener.open.assert_called_once()

    def test_live_identical_caption_lists_and_strings_are_canonicalized(self):
        content = json.dumps(caption_result())
        summary = json.dumps(caption_result()["summary"])
        content = content.replace('"summary": ' + summary, '"summary": ' + summary + ', "summary": ' + summary)
        content = content.replace('"objects": ["building", "road"]',
                                  '"objects": ["building", "road"], "objects": ["building", "road"]')
        self.opener.open.return_value = response(content=content)
        result = self.client.caption(self.image)
        self.assertEqual({key: value for key, value in result.items() if key != "_api"}, caption_result())
        self.assertEqual(result["_api"]["normalized_duplicate_keys"], 2)
        self.opener.open.assert_called_once()

    def test_live_nested_identical_status_is_accepted_but_conflict_is_rejected(self):
        original = json.dumps(verification_result())
        for repeated, accepted in [("supported", True), ("contradicted", False)]:
            content = original.replace('"status": "supported"',
                                       '"status": "supported", "status": ' + json.dumps(repeated))
            self.opener.open.return_value = response(content=content)
            if accepted:
                result = self.client.verify(self.image, "building")
                self.assertEqual(result["verdict"], "match")
                self.assertEqual(result["_api"]["normalized_duplicate_keys"], 1)
            else:
                with self.assertRaisesRegex(VisionResponseError, "malformed JSON"):
                    self.client.verify(self.image, "building")

    def test_live_duplicate_normalization_does_not_coerce_required_field_types(self):
        content = json.dumps({**caption_result(), "objects": "building"})
        content = content.replace('"objects": "building"', '"objects": "building", "objects": "building"')
        self.opener.open.return_value = response(content=content)
        with self.assertRaisesRegex(VisionResponseError, "objects must be an array"):
            self.client.caption(self.image)

    def test_duplicate_http_envelope_keys_remain_strict_even_when_identical(self):
        envelope = response().getvalue().replace(b'"id": "request-1"',
                                                 b'"id": "request-1", "id": "request-1"')
        self.opener.open.return_value = io.BytesIO(envelope)
        with self.assertRaisesRegex(VisionResponseError, "malformed JSON"):
            self.client.caption(self.image)

    def test_truncation_is_not_retried(self):
        self.opener.open.return_value = response(finish="length")
        with self.assertRaisesRegex(VisionError, "truncated"):
            self.client.caption(self.image)
        self.assertEqual(self.opener.open.call_count, 1)

    def test_malformed_json_is_not_retried_or_echoed(self):
        self.opener.open.return_value = response(content="fake-unit-test-secret invalid JSON")
        with self.assertRaises(VisionError) as context:
            self.client.caption(self.image)
        self.assertNotIn("fake-unit-test-secret", str(context.exception))
        self.assertEqual(self.opener.open.call_count, 1)

    def test_429_retries_with_capped_retry_after(self):
        error = urllib.error.HTTPError("https://irrelevant", 429, "secret diagnostics",
                                      {"Retry-After": "100000"}, io.BytesIO(b"secret body"))
        self.opener.open.side_effect = [error, response()]
        with patch("caption_rag.vision.time.sleep") as sleep:
            self.client.caption(self.image)
        sleep.assert_called_once_with(30.0)
        self.assertEqual(self.opener.open.call_count, 2)

    def test_5xx_retries_then_succeeds(self):
        self.opener.open.side_effect = [urllib.error.HTTPError("https://irrelevant", 503,
            "secret diagnostics", {}, io.BytesIO(b"secret body")), response()]
        with patch("caption_rag.vision.time.sleep") as sleep:
            self.client.caption(self.image)
        sleep.assert_called_once_with(1.0)

    def test_http_authentication_errors_are_not_retried_or_echoed(self):
        self.opener.open.side_effect = urllib.error.HTTPError("https://irrelevant", 401,
            "fake-unit-test-secret", {}, io.BytesIO(b"fake-unit-test-secret"))
        with self.assertRaisesRegex(VisionError, "HTTP 401") as context:
            self.client.caption(self.image)
        self.assertNotIn("fake-unit-test-secret", str(context.exception))
        self.assertIsNone(context.exception.__cause__)
        self.assertTrue(context.exception.__suppress_context__)
        self.assertEqual(self.opener.open.call_count, 1)

    def test_network_errors_are_bounded_and_do_not_echo_secrets(self):
        self.opener.open.side_effect = urllib.error.URLError("fake-unit-test-secret")
        with patch("caption_rag.vision.time.sleep") as sleep:
            with self.assertRaisesRegex(VisionError, "network or timeout") as context:
                self.client.caption(self.image)
        self.assertEqual(self.opener.open.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertNotIn("fake-unit-test-secret", str(context.exception))

    def test_timeout_is_retried(self):
        self.opener.open.side_effect = [TimeoutError("fake-unit-test-secret"), response()]
        with patch("caption_rag.vision.time.sleep"):
            self.client.caption(self.image)
        self.assertEqual(self.opener.open.call_count, 2)

    def test_url_validation_and_optional_v1(self):
        for url in ["http://api.deepseek.com", "https://user:secret@example.com",
                    "https://example.com?key=secret", "https://example.com/#fragment",
                    "http://localhost.evil.example", "https://exa mple.com"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                DeepSeekVision(api_key="test", base_url=url)
        for url in ["https://api.deepseek.com", "https://api.deepseek.com/v1",
                    "http://127.0.0.1:8000", "http://[::1]:8000", "http://localhost:8000"]:
            with self.subTest(url=url):
                client = DeepSeekVision(api_key="test", base_url=url + "/")
                self.assertEqual(client.endpoint, url + "/chat/completions")

    def test_redirects_are_disabled(self):
        from caption_rag.vision import _NoRedirect
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "redirect", {},
                                                        "https://other-host.example"))

    def test_key_is_taken_only_from_explicit_argument_or_environment(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "environment-key"}):
            self.assertEqual(DeepSeekVision()._api_key, "environment-key")
            self.assertEqual(DeepSeekVision(api_key="explicit-key")._api_key, "explicit-key")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(VisionError, "DEEPSEEK_API_KEY"):
                DeepSeekVision()
        with self.assertRaises(VisionError):
            DeepSeekVision(api_key="secret\nheader-injection")

    def test_empty_query_is_rejected_without_http(self):
        with self.assertRaises(ValueError):
            self.client.verify(self.image, " ")
        self.opener.open.assert_not_called()

    def test_unreadable_image_is_rejected_without_http(self):
        with self.assertRaisesRegex(VisionError, "decode reference image"):
            self.client.caption(Path(self.directory.name) / "missing.png")
        self.opener.open.assert_not_called()

    def test_response_retry_default_is_two_and_cannot_exceed_two(self):
        self.assertEqual(DeepSeekVision(api_key="test").response_retries, 2)
        for count in [-1, 3, True, 1.0]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                DeepSeekVision(api_key="test", response_retries=count)

    def test_caption_schema_error_retries_then_returns_only_valid_result(self):
        self.client.response_retries = 2
        self.opener.open.side_effect = [response({**caption_result(), "objects": "invalid"}),
                                       response()]
        result = self.client.caption(self.image)
        self.assertEqual(result["objects"], ["building", "road"])
        self.assertEqual(self.opener.open.call_count, 2)

    def test_truncation_and_malformed_json_retry_within_same_budget(self):
        self.client.response_retries = 2
        self.opener.open.side_effect = [response(finish="length"),
                                       response(content="fake-unit-test-secret malformed"), response()]
        self.assertEqual(self.client.caption(self.image)["summary"], caption_result()["summary"])
        self.assertEqual(self.opener.open.call_count, 3)

    def test_verification_schema_error_is_retried(self):
        self.client.response_retries = 2
        self.opener.open.side_effect = [response({**verification_result(), "score": "invalid"}),
                                       response(verification_result())]
        self.assertEqual(self.client.verify(self.image, "building")["verdict"], "match")
        self.assertEqual(self.opener.open.call_count, 2)

    def test_response_schema_retry_budget_is_exhausted_after_exactly_three_requests(self):
        self.client.response_retries = 2
        for operation in [lambda: self.client.caption(self.image),
                          lambda: self.client.verify(self.image, "building")]:
            self.opener.open.reset_mock()
            self.opener.open.side_effect = [response([]), response([]), response([])]
            with self.assertRaisesRegex(VisionResponseError, "after 3 response attempts"):
                operation()
            self.assertEqual(self.opener.open.call_count, 3)

    def test_auth_errors_bypass_response_retries(self):
        self.client.response_retries = 2
        for status in [401, 403]:
            self.opener.open.reset_mock()
            self.opener.open.side_effect = urllib.error.HTTPError("https://irrelevant", status,
                "fake-unit-test-secret", {}, io.BytesIO(b"fake-unit-test-secret"))
            with self.assertRaisesRegex(VisionError, f"HTTP {status}") as context:
                self.client.caption(self.image)
            self.assertNotIsInstance(context.exception, VisionResponseError)
            self.assertNotIn("fake-unit-test-secret", str(context.exception))
            self.assertEqual(self.opener.open.call_count, 1)

    def test_exhausted_transport_errors_bypass_response_retries(self):
        self.client.response_retries = 2
        self.opener.open.side_effect = urllib.error.URLError("fake-unit-test-secret")
        with patch("caption_rag.vision.time.sleep"):
            with self.assertRaises(VisionError) as context:
                self.client.caption(self.image)
        self.assertNotIsInstance(context.exception, VisionResponseError)
        self.assertEqual(self.opener.open.call_count, 3)

    def test_transport_and_response_retry_budgets_are_both_bounded(self):
        self.client.response_retries = 2
        attempts = []
        for _ in range(3):
            attempts.extend([urllib.error.URLError("network"), TimeoutError("timeout"), response([])])
        self.opener.open.side_effect = attempts
        with patch("caption_rag.vision.time.sleep") as sleep:
            with self.assertRaisesRegex(VisionResponseError, "after 3 response attempts"):
                self.client.caption(self.image)
        self.assertEqual(self.opener.open.call_count, 9)
        self.assertEqual(sleep.call_count, 6)

    def test_invalid_success_envelope_retries(self):
        self.client.response_retries = 2
        self.opener.open.side_effect = [io.BytesIO(b'{"choices": []}'), response()]
        self.client.caption(self.image)
        self.assertEqual(self.opener.open.call_count, 2)


class CachedResultValidationTests(unittest.TestCase):
    def test_raw_cached_json_duplicate_keys_still_rejected(self):
        content = json.dumps(verification_result()).replace('"score": 0.9', '"score": 0.9, "score": 0.9')
        with self.assertRaisesRegex(VisionResponseError, "malformed JSON"):
            _json_loads(content)

    def test_normalization_count_is_optional_strict_and_preserved_in_both_validators(self):
        for validator, original in [(validate_caption, caption_result()),
                                    (validate_verification, verification_result())]:
            for count in (0, 2):
                cached = {**original, "_api": {"normalized_duplicate_keys": count}}
                self.assertEqual(validator(cached)["_api"]["normalized_duplicate_keys"], count)
            for invalid in (True, 1.0, -1, "1", None):
                with self.subTest(validator=validator.__name__, invalid=invalid):
                    with self.assertRaisesRegex(VisionResponseError, "nonnegative integer"):
                        validator({**original, "_api": {"normalized_duplicate_keys": invalid}})

    def test_schema_diagnostics_describe_fields_and_types_without_echoing_contents(self):
        missing = caption_result()
        del missing["relations"]
        missing["fake-unit-test-secret"] = "fake-unit-test-secret"
        with self.assertRaises(VisionResponseError) as context:
            validate_caption(missing)
        diagnostic = str(context.exception)
        self.assertIn("missing fields: relations", diagnostic)
        self.assertIn("extra fields: 1", diagnostic)
        self.assertNotIn("fake-unit-test-secret", diagnostic)
        with self.assertRaises(VisionResponseError) as context:
            validate_caption({**caption_result(), "objects": "fake-unit-test-secret"})
        self.assertIn("objects must be an array; received string", str(context.exception))
        self.assertNotIn("fake-unit-test-secret", str(context.exception))

    def test_malformed_cached_caption_and_unknown_fields_are_rejected(self):
        for cached in [{**caption_result(), "objects": "building"},
                       {**caption_result(), "camera_id": 42},
                       {**caption_result(), "_api": "malformed"}]:
            with self.subTest(cached=cached), self.assertRaises(VisionError):
                validate_caption(cached)

    def test_malformed_cached_verification_is_rejected(self):
        for cached in [{**verification_result(), "score": float("nan")},
                       {**verification_result(), "requirements": []},
                       {**verification_result(), "unexpected": "value"}]:
            with self.subTest(cached=cached), self.assertRaises(VisionError):
                validate_verification(cached)

    def test_cached_verification_extra_requirement_fields_remain_errors(self):
        cached = verification_result()
        cached["requirements"][0]["confidence"] = 0.9
        with self.assertRaisesRegex(VisionResponseError, "extra fields: 1"):
            validate_verification(cached)

    def test_cached_contradicted_match_is_downgraded_without_mutating_input(self):
        cached = verification_result(status="contradicted")
        result = validate_verification(cached)
        self.assertEqual(result["verdict"], "no_match")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(cached["verdict"], "match")
        self.assertEqual(cached["score"], 0.9)
        result["requirements"][0]["evidence"] = "edited copy"
        self.assertNotEqual(cached["requirements"][0]["evidence"], "edited copy")

    def test_metadata_is_optional_and_sanitized_when_present(self):
        for validator, original in [(validate_caption, caption_result()),
                                    (validate_verification, verification_result())]:
            self.assertNotIn("_api", validator(original))
            metadata = {"model": "deepseek-flash", "response_id": "request-1",
                        "usage": {"prompt_tokens": 100, "diagnostics": "omit",
                                  "details": {"cached_tokens": 80}},
                        "diagnostics": "omit"}
            result = validator({**original, "_api": metadata})
            self.assertEqual(result["_api"], {
                "model": "deepseek-flash", "response_id": "request-1",
                "usage": {"prompt_tokens": 100, "details": {"cached_tokens": 80}}})
            result["_api"]["usage"]["details"]["cached_tokens"] = 0
            self.assertEqual(metadata["usage"]["details"]["cached_tokens"], 80)

    def test_caption_copy_has_independent_lists(self):
        cached = caption_result()
        result = validate_caption(cached)
        result["objects"].append("edited copy")
        self.assertNotIn("edited copy", cached["objects"])


class ModelJsonNormalizationTests(unittest.TestCase):
    def test_nested_objects_compare_canonically_and_preserve_every_json_type(self):
        result, count = _model_json_loads(
            '{"x":{"a":[1,1.0,true,null,"s"],"b":2},"x":{"b":2,"a":[1,1.0,true,null,"s"]}}')
        self.assertEqual(count, 1)
        self.assertEqual(result["x"]["b"], 2)
        for left, right in [("true", "1"), ("false", "0"), ("1", "1.0"), ("0.0", "-0.0"),
                            ('[1,true]', '[true,1]'), ('{"n":1}', '{"n":1.0}'),
                            ('"1"', '1'), ('null', '"null"')]:
            with self.subTest(left=left, right=right), self.assertRaises(VisionResponseError):
                _model_json_loads('{"x":' + left + ',"x":' + right + '}')

    def test_nested_nonfinite_numbers_and_malformed_syntax_are_never_repaired(self):
        for content in ('{"extra":[NaN]}', '{"extra":{"n":Infinity}}', '{"n":1e400}',
                        '{"n":-1e400}', '{"x":1,}', '```json\n{"x":1}\n```'):
            with self.subTest(content=content), self.assertRaisesRegex(VisionResponseError, "malformed JSON"):
                _model_json_loads(content)


if __name__ == "__main__":
    unittest.main()
