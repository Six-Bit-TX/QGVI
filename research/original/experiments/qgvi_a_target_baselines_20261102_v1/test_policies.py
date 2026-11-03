"""Acceptance and isolation checks for fixed visual target-view baselines."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

# Avoid importing another experiment's identically named policies module.
SPEC = importlib.util.spec_from_file_location("target_baseline_policies", Path(__file__).with_name("policies.py"))
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


@pytest.fixture
def task():
    return {"id": "task_1", "output_type": "target_view", "instruction": "Show the same bay completely.",
            "source_image": "/nonexistent/source.png", "predicates": [
                {"id": "P1", "text": "The same physical building is visible.",
                 "origin": "frozen_development_task_constraint"},
                {"id": "P2", "text": "The complete bay is visible."}]}


@pytest.fixture
def candidates():
    return [{"candidate_id": f"fixed_{i}", "image_path": f"/nonexistent/{i}.png",
             "frame": {"c2w": [[1, 0, 0, i], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}}
            for i in range(6)]


def response(task, candidate="fixed_0", observation="C1"):
    return {"schema_version": policy.VERSION, "classification": "answer", "candidate_id": candidate,
            "assessment": {"answer_id": candidate, "anchor_supported": True,
                           "geometric_evidence_ids": [], "prefer_over_saved_answer": False,
                           "predicate_results": [{"predicate_id": p["id"], "status": "satisfied",
                                                  "evidence_ids": ["SOURCE", observation],
                                                  "reason": "Visible matching anchor and requested framing."}
                                                 for p in task["predicates"]]},
            "reason": "The supplied view visibly supports all frozen predicates."}


def test_single_selection_accepts_only_the_selected_candidate(task, candidates):
    r = response(task, "fixed_5", "C6")
    answer = policy.validate_response(r, task, candidates)
    assert answer == {"accepted": True, "candidate_id": "fixed_5", "assessment": r["assessment"],
                      "reason": "all_frozen_predicates_visually_supported"}
    answer["assessment"]["predicate_results"][0]["status"] = "unknown"
    assert r["assessment"]["predicate_results"][0]["status"] == "satisfied"


def test_fixed_call_has_only_current_image_and_no_hidden_state(task, candidates):
    request = policy.build_request(task, [candidates[4]], "fixed_search_verify")
    p = request["user_payload"]
    assert p["early_exit_candidate_id"] == "fixed_4"
    assert p["candidates"] == [{"candidate_id": "fixed_4", "render_observation_id": "C1"}]
    assert [o["id"] for o in p["observations"]] == ["SOURCE", "C1"]
    assert policy.validate_response(response(task, "fixed_4", "C1"), task, [candidates[4]])["accepted"]
    assert "fixed_3" not in json.dumps(request) and "fixed_5" not in json.dumps(request)
    with pytest.raises(ValueError, match="requires_one_current_candidate"):
        policy.build_request(task, candidates, "fixed_search_verify")


def test_request_is_pure_and_does_not_leak_labels_frames_paths_or_scores(task, candidates, monkeypatch):
    task.update(ground_truth_pose="SECRET_GT", candidate_judgments="SECRET_LABEL", history="SECRET_HISTORY")
    task["predicates"][0]["reviewer_label"] = "SECRET_REVIEW"
    candidates[0].update(score=999, reviewer_label="SECRET_CANDIDATE_LABEL", future_views="SECRET_FUTURE")
    before = copy.deepcopy((task, candidates))
    def forbid(*args, **kwargs):
        raise AssertionError("build_request must not read images or other files")
    monkeypatch.setattr("builtins.open", forbid)
    monkeypatch.setattr(Path, "open", forbid)
    request = policy.build_request(task, candidates, "retrieve_select")
    assert (task, candidates) == before
    public = json.dumps(request["user_payload"])
    assert "SECRET_" not in public and "/nonexistent" not in public and "c2w" not in public
    assert request["user_payload"]["required_predicates"][0]["text"] == task["predicates"][0]["text"]
    assert policy.MODEL == "gpt-6.1-sol" and policy.REASONING_EFFORT == "low"


@pytest.mark.parametrize("mode", policy.MODES)
def test_transport_schema_omits_unsupported_unique_items_keyword(task, candidates, mode):
    views = candidates if mode == "retrieve_select" else candidates[:1]
    schema = policy.build_request(task, views, mode)["output_schema"]
    policy.jsonschema.Draft202012Validator.check_schema(schema)
    def inspect(value):
        if isinstance(value, dict):
            assert "uniqueItems" not in value
            for child in value.values(): inspect(child)
        elif isinstance(value, list):
            for child in value: inspect(child)
    inspect(schema)
    # The request schema intentionally describes the supported transport subset;
    # the host checker below retains the original full uniqueness requirement.
    duplicate = response(task)
    duplicate["assessment"]["predicate_results"][0]["evidence_ids"].append("SOURCE")
    policy.jsonschema.Draft202012Validator(schema).validate(duplicate)
    assert policy.validate_response(duplicate, task, views)["reason"] == "invalid_response:duplicate_evidence_ids"


@pytest.mark.parametrize("status", ["satisfied", "unknown", "violated"])
@pytest.mark.parametrize("duplicate", ["SOURCE", "C1"])
def test_duplicate_evidence_is_rejected_locally_for_every_predicate_status(task, candidates, status, duplicate):
    r = response(task)
    check = r["assessment"]["predicate_results"][0]
    check["status"] = status
    check["evidence_ids"].append(duplicate)
    checked = policy.validate_response(r, task, candidates)
    assert checked == {"accepted": False, "candidate_id": None, "assessment": None,
                       "reason": "invalid_response:duplicate_evidence_ids"}


@pytest.mark.parametrize("status,reason", [("unknown", "predicate_unknown"), ("violated", "predicate_violated")])
def test_semantic_failure_is_abstention_not_schema_error_or_ground_truth(task, candidates, status, reason):
    r = response(task)
    r["assessment"]["predicate_results"][1]["status"] = status
    r["assessment"]["predicate_results"][1]["evidence_ids"] = []
    checked = policy.validate_response(r, task, candidates)
    assert checked == {"accepted": False, "candidate_id": None,
                       "assessment": r["assessment"], "reason": reason}
    assert not any(k in checked for k in ("correct", "wrong", "evaluation_correctness"))


def test_unknown_source_anchor_never_accepts(task, candidates):
    r = response(task)
    r["assessment"]["anchor_supported"] = False
    checked = policy.validate_response(r, task, candidates)
    assert not checked["accepted"] and checked["reason"] == "source_anchor_unsupported"


def test_explicit_abstention_can_omit_assessment(task, candidates):
    r = response(task)
    r.update(classification="insufficient_evidence", candidate_id=None, assessment=None)
    assert policy.validate_response(r, task, candidates) == {
        "accepted": False, "candidate_id": None, "assessment": None, "reason": "insufficient_evidence"}


def test_explicit_abstention_is_not_overridden_by_supported_assessment(task, candidates):
    r = response(task)
    r.update(classification="insufficient_evidence", candidate_id=None)
    checked = policy.validate_response(r, task, candidates)
    assert not checked["accepted"] and checked["reason"] == "insufficient_evidence"


@pytest.mark.parametrize("mutation", [
    lambda r: r["assessment"]["predicate_results"].pop(),
    lambda r: r["assessment"]["predicate_results"].append(copy.deepcopy(r["assessment"]["predicate_results"][0])),
    lambda r: r["assessment"]["predicate_results"][1].update(predicate_id="P1"),
    lambda r: r["assessment"]["predicate_results"][1].update(predicate_id="P999"),
    lambda r: r["assessment"]["predicate_results"][0].update(status="pass"),
    lambda r: r["assessment"]["predicate_results"][0].update(evidence_ids=["C1"]),
    lambda r: r["assessment"]["predicate_results"][0].update(evidence_ids=["SOURCE"]),
    lambda r: r["assessment"]["predicate_results"][0].update(evidence_ids=["SOURCE", "C2"]),
    lambda r: r["assessment"]["predicate_results"][0].update(evidence_ids=["SOURCE", "C1", "C2"]),
    lambda r: r["assessment"]["predicate_results"][0].update(evidence_ids=["SOURCE", "C7"]),
    lambda r: r["assessment"].update(answer_id="fixed_1"),
    lambda r: r.update(candidate_id="not_supplied"),
    lambda r: r.update(assessment=None),
    lambda r: r["assessment"].update(anchor_supported=1),
    lambda r: r["assessment"].update(geometric_evidence_ids=["invented_pnp"]),
    lambda r: r["assessment"].update(prefer_over_saved_answer=True),
    lambda r: r.update(classification="out_of_map"),
    lambda r: r.update(classification="insufficient_evidence"),
    lambda r: r.update(movement="yaw_right"),
])
def test_malformed_or_unbound_evidence_fails_closed(task, candidates, mutation):
    r = response(task)
    mutation(r)
    checked = policy.validate_response(r, task, candidates)
    assert checked["accepted"] is False and checked["candidate_id"] is None
    assert checked["assessment"] is None and checked["reason"].startswith("invalid_response:")


@pytest.mark.parametrize("malformed", [None, [], {}, "not json", '{"schema_version":"x","schema_version":"y"}',
                                     '{"x":NaN}', {"reason": object()}])
def test_malformed_json_is_separate_from_semantic_abstention(task, candidates, malformed):
    checked = policy.validate_response(malformed, task, candidates)
    assert checked["reason"].startswith("invalid_response:") and not checked["accepted"]


def test_valid_json_string_is_supported(task, candidates):
    assert policy.validate_response(json.dumps(response(task)), task, candidates)["accepted"]


@pytest.mark.parametrize("status", ["satisfied", "unknown", "violated"])
def test_assessment_matches_existing_native_schema_and_early_exit(task, candidates, monkeypatch, status):
    native = Path(__file__).resolve().parents[1] / "qgvi_a_full_eval_20261029_v1"
    def load(name):
        spec = importlib.util.spec_from_file_location("baseline_test_native_" + name, native / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    protocol = load("protocol")
    monkeypatch.setitem(sys.modules, "protocol", protocol)
    monkeypatch.setitem(sys.modules, "io_utils", load("io_utils"))
    early_exit = load("early_exit")
    r = response(task)
    r["assessment"]["predicate_results"][1]["status"] = status
    policy.jsonschema.validate(r["assessment"], protocol.ASSESSMENT)
    state = {"source_observation_id": "SOURCE",
             "answers": {"fixed_0": {"kind": "target_view", "render_observation_id": "C1"}},
             "observations": {"SOURCE": {"origin": "source"},
                              "C1": {"origin": "gaussian_render", "answer_id": "fixed_0"}}}
    native_accepted = early_exit.check_assessment(state, task, r["assessment"], "fixed_0", ["SOURCE", "C1"])
    assert policy.validate_response(r, task, candidates)["accepted"] == native_accepted


@pytest.mark.parametrize("fields", [{"valid": False}, {"valid": 1}, {"status": "unusable_render"},
                                    {"status": "insufficient_coverage"}, {"origin": "retrieved_reference"},
                                    {"image_path": ""}, {"frame": None}])
def test_invalid_render_cannot_become_a_model_request_or_answer(task, candidates, fields):
    candidates[0].update(fields)
    with pytest.raises(ValueError):
        policy.build_request(task, candidates, "retrieve_select")
    with pytest.raises(ValueError):
        policy.validate_response(response(task), task, candidates)


def test_input_predicate_and_candidate_ids_must_be_unique(task, candidates):
    task["predicates"][1]["id"] = "P1"
    with pytest.raises(ValueError, match="duplicate_task_predicate"):
        policy.build_request(task, candidates, "retrieve_select")
    task["predicates"][1]["id"] = "P2"
    candidates[1]["candidate_id"] = candidates[0]["candidate_id"]
    with pytest.raises(ValueError, match="duplicate_candidate"):
        policy.build_request(task, candidates, "retrieve_select")


def test_source_and_nonempty_bounded_candidate_set_required(task, candidates):
    for views in ([], candidates + [dict(candidates[0], candidate_id="seventh")]):
        with pytest.raises(ValueError, match="need_one_to_six"):
            policy.build_request(task, views, "retrieve_select")
    task["source_image"] = None
    with pytest.raises(ValueError, match="source_image_required"):
        policy.build_request(task, candidates, "retrieve_select")
