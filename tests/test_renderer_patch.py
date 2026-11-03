"""Source and bounded PLY parser checks; these do not import a CUDA renderer."""
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "third_party_patches"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _added_ply_reader(tmp_path):
    """Extract the new-file hunk; other hunks require the upstream checkout."""
    patch = (PATCHES / "HorizonGS-runtime.patch").read_text()
    part = patch.split("diff --git a/utils/streaming_ply.py b/utils/streaming_ply.py\n", 1)[1]
    source = "".join(line[1:] for line in part.splitlines(keepends=True)
                     if line.startswith("+") and not line.startswith("+++"))
    file = tmp_path / "streaming_ply.py"
    file.write_text(source)
    return _load(file, "qgvi_patch_ply_test")


def test_renderer_patch_hash_and_license():
    receipt = json.loads((PATCHES / "HorizonGS-runtime-provenance.json").read_text())
    assert hashlib.sha256((PATCHES / receipt["patch"]).read_bytes()).hexdigest() == receipt["patch_sha256"]
    assert receipt["verification"]["selected_definition_parity"]
    assert receipt["verification"]["applies_to_pinned_clean_source"]
    assert not receipt["verification"]["gpu_execution"]
    assert len(receipt["files"]) == 4
    assert "Gaussian-Splatting License" in (PATCHES / "HorizonGS-LICENSE.md").read_text()


def test_gsplat_archive_provenance():
    receipt = json.loads((PATCHES / "gsplat-source-provenance.json").read_text())
    assert receipt["commit"] == "2323de5905d5e90e035f792fe65bad0fedd413e7"
    assert len(receipt["source_files"]) == 154
    assert receipt["archive_files_match_installed_editable_source"]
    assert receipt["local_source_changes"] == []


def test_strict_ply_layout_accepts_exact_bytes(tmp_path):
    reader = _added_ply_reader(tmp_path)
    file = tmp_path / "valid.ply"
    header = b"ply\nformat binary_little_endian 1.0\nobj_info standard_dist 1\nelement vertex 2\nproperty float x\nproperty float y\nend_header\n"
    file.write_bytes(header + struct.pack("<ffff", 1, 2, 3, 4))
    result = reader.read_binary_float_ply(file)
    assert result.vertex_count == 2
    assert result.properties == ("x", "y")
    assert result.expected_size == len(header) + 16


@pytest.mark.parametrize("damage", ["truncated", "trailing", "duplicate", "ascii"])
def test_strict_ply_layout_rejects_invalid_data(tmp_path, damage):
    reader = _added_ply_reader(tmp_path)
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n"
    data = header + struct.pack("<f", 1)
    if damage == "truncated":
        data = data[:-1]
    elif damage == "trailing":
        data += b"x"
    elif damage == "duplicate":
        data = data.replace(b"end_header", b"property float x\nend_header")
    else:
        data = data.replace(b"binary_little_endian", b"ascii")
    file = tmp_path / "invalid.ply"
    file.write_bytes(data)
    with pytest.raises(ValueError):
        reader.read_binary_float_ply(file)
