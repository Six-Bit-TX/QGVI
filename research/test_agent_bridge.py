"""CPU-only integration tests against the unmodified controller source."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.agent import FULL, original_controller, run_agent
from runtime.provider_bridge import CommandProvider, ProviderError, bounded_command


def task(tmp_path, output_type="target_view"):
    image = tmp_path / "source.png"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(image)
    return {"id": "query", "scene": "fixture", "output_type": output_type,
            "instruction": "Show the same doorway.",
            "predicates": [{"id": "same", "text": "The same doorway is visible"}],
            "host_bundle_id": "host_source", "source_image": str(image),
            "source_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "K": [[20, 0, 16], [0, 20, 12], [0, 0, 1]],
            "width": 32, "height": 24, "domain": "street"}


class Retrieval:
    def __init__(self, query):
        self.query = query

    def bind_source(self, value):
        self.source = value

    def search_text(self, text, limit=6, bundle_id=None):
        q = self.query
        return {"candidates": [{"view_id": "reference", "image_path": q["source_image"],
                                "score": 1., "caption": "A doorway",
                                "frame": {"c2w": np.eye(4).tolist(),
                                          **{k: q[k] for k in ("K", "width", "height", "domain")}}}]}

    def search_image(self, bundle, limit=6):
        return self.search_text("", limit, bundle)

    def estimate_pose(self, bundle, region=None):
        return {"pose_c2w": np.eye(4).tolist(), "evidence": {"unique_inlier_queries": 40}}


class Navigation:
    def __init__(self, root):
        self.root = root

    def propose_view(self, frame):
        path = self.root / "render.png"
        Image.new("RGB", (frame["width"], frame["height"]), (50, 60, 70)).save(path)
        return {"valid": True, "frame": copy.deepcopy(frame), "image_path": str(path),
                "coverage": .99, "cache_hit": False}


def provider(tmp_path, *, first_unknown=False):
    script = tmp_path / "provider.py"
    script.write_text('''import json, pathlib, sys
p = json.load(sys.stdin)
assert p['requested_model'] == 'gpt-6.1-sol'
assert pathlib.Path(p['image_path']).is_file()
u = p['user_payload']
target = [a for a in u['camera_answers'] if a['kind'] == 'target_view'][-1]
source = next(o['id'] for o in u['observations'] if o['origin'] == 'source')
counter = pathlib.Path(__file__).with_suffix('.count')
n = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(n + 1))
status = 'unknown' if FIRST_UNKNOWN and n == 0 else 'satisfied'
args = {key: None for key in p['output_schema']['properties']['arguments']['properties']}
args.update(classification='answer', answer_id=target['id'])
a = {'schema_version': u['schema_version'], 'action': 'finish', 'arguments': args,
     'assessment': {'answer_id': target['id'], 'anchor_supported': True,
       'geometric_evidence_ids': [], 'prefer_over_saved_answer': True,
       'predicate_results': [{'predicate_id': v['id'], 'status': status,
          'evidence_ids': [source, target['render_observation_id']], 'reason': 'Fixture pixels'}
          for v in u['required_predicates']]},
     'next_stage': None, 'unresolved_requirements': [], 'reason': 'Fixture decision'}
print(json.dumps({'response': a, 'reported_model': p['requested_model'],
                  'reasoning_effort': p['reasoning_effort']}))
'''.replace("FIRST_UNKNOWN", repr(first_unknown)))
    return [sys.executable, str(script)]


def test_target_early_exit_calls_original_controller_once(tmp_path):
    q = task(tmp_path)
    before = hashlib.sha256((FULL / "controller.py").read_bytes()).hexdigest()
    out = run_agent(q, tmp_path / "run", Retrieval(q), Navigation(tmp_path), provider(tmp_path))
    result = out["result"]
    assert result["answer"]["classification"] == "answer"
    assert result["model_calls"] == 1
    assert result["early_exit"]["early_exit"] is True
    assert result["transport"] == "external_command"
    assert result["reported_models"] == ["gpt-6.1-sol"]
    assert out["public_transport"]["native_codex_transport_reproduction"] is False
    with original_controller() as (controller, _):
        assert controller.replay(tmp_path / "run")["status"] == "passed"
    assert before == hashlib.sha256((FULL / "controller.py").read_bytes()).hexdigest()


def test_source_geometric_early_exit_never_starts_provider(tmp_path):
    q = task(tmp_path, "source_camera_pose")
    out = run_agent(q, tmp_path / "run", Retrieval(q), Navigation(tmp_path),
                    ["this-command-must-not-be-started"])
    assert out["result"]["model_calls"] == 0
    assert out["result"]["answer"]["classification"] == "answer"
    assert out["result"]["tool_counts"] == {"search_image": 1, "estimate_pose": 1}


def test_unknown_verification_continues_without_repeating_tools(tmp_path):
    q = task(tmp_path)
    out = run_agent(q, tmp_path / "run", Retrieval(q), Navigation(tmp_path),
                    provider(tmp_path, first_unknown=True))
    result = out["result"]
    assert result["model_calls"] == 2
    assert result["early_exit"]["fallback_to_agent"] is True
    assert result["tool_counts"] == {"search_text": 1, "propose_view": 1}
    assert result["answer"]["classification"] == "answer"


def test_model_metadata_mismatch_is_not_labeled_as_requested_model(tmp_path):
    q = task(tmp_path)
    command = [sys.executable, "-c", "import json; print(json.dumps({'response':{},'reported_model':'other','reasoning_effort':'low'}))"]
    with pytest.raises(ProviderError, match="reported_model_mismatch"):
        CommandProvider(command).request(q["source_image"], "test", {}, {"type": "object"})


def test_command_argument_change_cannot_resume_same_run(tmp_path):
    q = task(tmp_path, "source_camera_pose")
    run_agent(q, tmp_path / "run", Retrieval(q), Navigation(tmp_path), ["first"])
    with pytest.raises(ValueError, match="command or effort changed"):
        run_agent(q, tmp_path / "run", Retrieval(q), Navigation(tmp_path), ["second"])


def test_provider_uses_no_shell_and_does_not_expose_stderr():
    with pytest.raises(ProviderError) as error:
        bounded_command([sys.executable, "-c", "import sys;sys.stderr.write('secret-content');sys.exit(7)"], b"{}", 2)
    assert error.value.kind == "external_command_failed"
    assert error.value.safe_details == {"returncode": 7}
    assert "secret-content" not in str(error.value)


def test_timeout_is_bounded():
    with pytest.raises(ProviderError, match="external_command_timeout"):
        bounded_command([sys.executable, "-c", "import time;time.sleep(10)"], b"{}", .05)


def test_output_limit_is_bounded(monkeypatch):
    from runtime import provider_bridge
    monkeypatch.setattr(provider_bridge, "MAX_CAPTURE_BYTES", 100)
    with pytest.raises(ProviderError, match="external_command_output_limit"):
        bounded_command([sys.executable, "-c", "print('x'*200)"], b"{}", 2)


def test_imports_are_restored_after_controller_run():
    previous = sys.modules.get("providers")
    with original_controller() as (controller, tools):
        assert controller.__file__ == str(FULL / "controller.py")
        assert tools.__file__ == str(FULL / "tool_host.py")
        assert sys.modules["providers"].TRANSPORT == "external_command"
    assert sys.modules.get("providers") is previous
