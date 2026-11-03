#!/usr/bin/env python3
"""Exact adapted LoGS geometric initialization, without Gaussian/photo loading.

Reference features/SfM/NetVLAD are reused read-only. This produces RAW poses:
acceptance is null until a separately frozen, qualified policy is supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from catalog import VALIDATION_SCENES, _h5_names, read, sha, shared_query_rows, verify_plan, write_new


def require_runtime_version(actual, qualified):
    if actual != "3.13.0" or actual != qualified:
        raise ValueError("PnP solver must match the qualified original pycolmap3.13.0")


def require_smoke_qualification(plan):
    if plan.get("purpose") == "smoke":
        return
    gates = plan.get("required_smoke_summaries", [])
    if {g["scene"] for g in gates} != set(VALIDATION_SCENES) or len(gates) != 2:
        raise ValueError("Validation requires both Block A and SF two-query implementation-parity smokes")
    for gate in gates:
        old_plan = read(gate["plan"])
        summary = read(gate["summary"])
        if (summary.get("status") != "complete" or summary.get("plan_sha256") != sha(gate["plan"])
                or old_plan.get("purpose") != "smoke" or old_plan["scene"] != gate["scene"]
                or len(old_plan["queries"]) != 2 or old_plan["source_sha256"] != plan["source_sha256"]
                or summary.get("smoke_qualification", {}).get("passed") is not True):
            raise ValueError("Smoke implementation parity has not passed for the frozen sources")


def qualify_descriptor(actual, cached, tolerances):
    import numpy as np
    actual, cached = np.asarray(actual, np.float64), np.asarray(cached, np.float64)
    if actual.shape != (4096,) or cached.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(cached).all():
        raise ValueError("Invalid reference descriptor parity vectors")
    denom = np.linalg.norm(actual) * np.linalg.norm(cached)
    cosine = float(actual @ cached / denom) if denom else -1.
    delta = float(np.max(np.abs(actual - cached)))
    passed = bool(np.allclose(actual, cached, atol=tolerances["descriptor_atol"],
                              rtol=tolerances["descriptor_rtol"]) and cosine >= tolerances["descriptor_min_cosine"])
    return dict(passed=passed, max_abs_difference=delta, cosine_similarity=cosine,
                atol=tolerances["descriptor_atol"], rtol=tolerances["descriptor_rtol"],
                minimum_cosine=tolerances["descriptor_min_cosine"],
                tolerance_reason="Fixed before smoke; permits float16 rounding plus floating-point execution variation, not changed descriptors")


def qualify_pose_parity(predictions, original, tolerances):
    import numpy as np
    by_id = {q["query_id"]: q for q in original}
    records = []
    for new in predictions:
        old = by_id[new["query_id"]]
        a, b = new["raw_c2w"], old["c2w"]
        pose_ok = a is None and b is None
        delta = None
        if a is not None and b is not None:
            delta = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            pose_ok = bool(np.isfinite(delta) and delta <= tolerances["pose_max_abs_native"])
        exact_fields = ("status", "retrieved_training_names", "num_inliers", "num_matches")
        field_parity = {key: new[key] == old[key] for key in exact_fields}
        records.append(dict(query_id=new["query_id"], passed=pose_ok and all(field_parity.values()),
                            max_abs_pose_difference_native=delta, exact_field_parity=field_parity))
    return dict(passed=all(row["passed"] for row in records), queries=records,
                max_abs_tolerance_native=tolerances["pose_max_abs_native"],
                tolerance_reason="Fixed implementation-repeatability tolerance on native rigid-transform entries; never an accuracy threshold",
                no_ground_truth_used=True)


def prediction_record(query, pose, *, pnp, names, num_inliers, num_matches, seconds):
    import numpy as np
    c2w = None
    if pose is not None:
        # Preserve the legacy adapter's arithmetic exactly; no projection or
        # acceptance-dependent pose modification is performed.
        w2c = np.eye(4)
        w2c[:3] = pose.matrix()
        c2w = np.linalg.inv(w2c).tolist()
        if not np.isfinite(c2w).all():
            raise ValueError("Nonfinite geometric initializer pose")
    return dict(query_id=query["id"], completed=True, raw_pose_available=c2w is not None,
                accepted=None, decision_status="uncalibrated", raw_c2w=c2w, c2w=c2w,
                status="pnp" if pnp else "retrieval_top1_fallback" if names else "no_retrieved_reference",
                retrieved_training_names=names, num_inliers=int(num_inliers),
                num_matches=int(num_matches), pnp_seconds=seconds)


def copy_reused_query_features(plan, output_dir):
    """Copy qualified image-only rows into new H5 files; never modify producer."""
    reuse = plan.get("query_feature_reuse")
    if reuse is None:
        return dict(reused_query_count=0, copied_feature_files=[], source=None)
    if plan.get("purpose") != "validation" or plan.get("scene") != "sf":
        raise ValueError("Feature reuse is restricted to SF validation, never smoke")
    source_path = Path(reuse["source_plan"])
    if sha(source_path) != reuse["source_plan_sha256"]:
        raise ValueError("Query feature producer plan changed")
    producer = read(source_path)
    if (producer.get("scene") != "block_a" or producer.get("purpose") != "validation"
            or producer.get("split") != "validation"
            or producer["source_sha256"] != plan["source_sha256"]
            or any(producer[key] != plan[key] for key in ("recipe", "feature_config", "matcher_config"))
            or any(producer["reference_assets"][key] != plan["reference_assets"][key]
                   for key in ("seed", "netvlad_weights_sha256"))
            or producer["reference_assets"].get("qualified") is not True):
        raise ValueError("Feature producer does not match qualified recipe and sources")
    expected_rows = shared_query_rows(producer["queries"], plan["queries"])
    if not reuse["rows"] or reuse["rows"] != expected_rows:
        raise ValueError("Feature reuse differs from matching pixels and dimensions")
    source_dir = Path(producer["output_dir"])
    summary_path = source_dir / "summary.json"
    summary_sha256 = sha(summary_path)
    summary = read(summary_path)
    if (summary.get("status") != "complete" or summary.get("plan_sha256") != sha(source_path)
            or summary.get("query_count") != len(producer["queries"])
            or summary.get("method") != producer["recipe"]["method"]
            or summary.get("source_bindings_sha256") != producer["source_sha256"]
            or summary.get("no_query_ground_truth_access") is not True
            or summary.get("photometric_refinement") is not False
            or summary.get("reference_assets") != producer["reference_assets"]["bindings_sha256"]):
        raise ValueError("Feature producer has no complete qualified initializer receipt")
    source_queries = {q["id"]: q for q in producer["queries"]}
    target_queries = {q["id"]: q for q in plan["queries"]}
    for row in reuse["rows"]:
        for query in (source_queries[row["source_query_id"]], target_queries[row["target_query_id"]]):
            if sha(query["image_path"]) != row["image_sha256"]:
                raise ValueError("Feature reuse public query pixels changed")
    import h5py
    import numpy as np
    copied, bindings = [], {}
    filenames = ("query_netvlad.h5", "query_superpoint.h5")
    output_dir = Path(output_dir)
    for filename in filenames:
        source = source_dir / filename
        expected = summary.get("artifacts_sha256", {}).get(str(source))
        if expected is None or sha(source) != expected:
            raise ValueError("Feature producer artifact changed or lacks digest")
        bindings[str(source)] = expected
        if (output_dir / filename).exists():
            raise FileExistsError("Feature reuse never overwrites destination caches")
        with h5py.File(source, "r") as handle:
            if _h5_names(handle) != {q["id"] + ".png" for q in producer["queries"]}:
                raise ValueError("Feature producer cache membership is incomplete")
            for row in reuse["rows"]:
                group = handle[row["source_query_id"] + ".png"]
                if group["image_size"][()].tolist() != [row["width"], row["height"]]:
                    raise ValueError("Feature cache image dimensions differ from public pixels")
                if filename == "query_netvlad.h5":
                    descriptor = group["global_descriptor"]
                    if descriptor.shape != (4096,) or descriptor.dtype != np.float16:
                        raise ValueError("Feature cache NetVLAD precision/shape differs")
                else:
                    n = len(group["keypoints"])
                    if (group["keypoints"].shape != (n, 2) or group["descriptors"].shape != (256, n)
                            or group["scores"].shape != (n,)
                            or any(group[k].dtype != np.float16 for k in ("keypoints", "descriptors", "scores"))):
                        raise ValueError("Feature cache SuperPoint precision/shape differs")
    # Every source is verified before creating either destination. HDF5 copy
    # preserves feature values and keypoint uncertainty attributes exactly.
    for filename in filenames:
        target = output_dir / filename
        with h5py.File(source_dir / filename, "r") as source, h5py.File(target, "x") as dest:
            for row in reuse["rows"]:
                source.copy(row["source_query_id"] + ".png", dest, name=row["target_query_id"] + ".png")
        copied.append(str(target))
    if (sha(source_path) != reuse["source_plan_sha256"] or sha(summary_path) != summary_sha256
            or any(sha(path) != digest for path, digest in bindings.items())):
        raise ValueError("Feature producer changed during read-only reuse")
    return dict(reused_query_count=len(reuse["rows"]), copied_feature_files=copied,
                source=dict(plan=str(source_path), plan_sha256=sha(source_path),
                            summary=str(summary_path), summary_sha256=summary_sha256,
                            feature_artifacts_sha256=bindings),
                rows=reuse["rows"], source_open_mode="read_only",
                reused_stages=["query_netvlad", "query_superpoint"],
                fresh_stages=["retrieval", "query_superglue", "pnp"])


def run(plan_path):
    plan_path = Path(plan_path)
    p = read(plan_path)
    out = Path(p["output_dir"])
    if out.exists():
        raise FileExistsError("Output already exists; completed/partial results are immutable")
    verify_plan(p)  # no GPU imported before public inputs/source checks pass
    require_smoke_qualification(p)
    a = p["reference_assets"]
    sys.path.insert(0, a["upstream"])
    import numpy as np
    import pycolmap
    import torch
    from hloc import extract_features, localize_sfm, match_features, pairs_from_retrieval
    from hloc.utils.parsers import parse_retrieval
    require_runtime_version(pycolmap.__version__, a["qualification"]["pycolmap"])
    if not torch.cuda.is_available():
        raise RuntimeError("Qualified GPU admission is required; no silent CPU substitution")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "STARTED.json", dict(plan_sha256=sha(plan_path), started_unix=time.time(),
                                       private_labels_accessed=False))
    torch.set_num_threads(2)
    torch.manual_seed(a["seed"])
    np.random.seed(a["seed"])
    torch.hub.set_dir(a["torch_cache"])
    torch.hub.download_url_to_file = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Unprepared download prohibited"))
    refs = a["reference_names"]
    ref_features = Path(a["reference_features"])
    ref_global = Path(a["reference_netvlad"])
    query_root = Path(p["queries"][0]["image_path"]).parent
    names = [q["id"] + ".png" for q in p["queries"]]
    timing, started = {}, time.perf_counter()
    smoke_qualification = None
    reuse_started = time.perf_counter()
    feature_reuse = copy_reused_query_features(p, out)
    timing["query_feature_reuse_cpu_s"] = time.perf_counter() - reuse_started
    write_new(out / "QUERY_FEATURE_REUSE.json", feature_reuse)

    def stage(name, fn):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        timing[name] = time.perf_counter() - begin
        return value

    if p.get("purpose") == "smoke":
        import h5py
        reference_check = stage("smoke_reference_netvlad_s", lambda: extract_features.main(
            extract_features.confs["netvlad"], Path(a["reference_image_root"]), out, as_half=True,
            image_list=[a["smoke_reference_name"]], feature_path=out / "smoke_reference_netvlad.h5"))
        with h5py.File(reference_check, "r") as fresh, h5py.File(ref_global, "r") as cached:
            key = a["smoke_reference_name"] + "/global_descriptor"
            parity = qualify_descriptor(fresh[key][()], cached[key][()], p["smoke_tolerances"])
        smoke_qualification = dict(reference_netvlad=parity, passed=False)
        write_new(out / "REFERENCE_DESCRIPTOR_PARITY.json", parity)
        if not parity["passed"]:
            raise ValueError("Fresh pinned reference descriptor differs from the historical cache")

    # as_half=True is explicit: the controlled ablation's float32/960 features
    # are intentionally not substituted for this exact historical LoGS route.
    query_global = stage("query_netvlad_s", lambda: extract_features.main(
        extract_features.confs["netvlad"], query_root, out, as_half=True,
        image_list=names, feature_path=out / "query_netvlad.h5"))
    features = stage("query_superpoint_s", lambda: extract_features.main(
        p["feature_config"], query_root, out, as_half=True,
        image_list=names, feature_path=out / "query_superpoint.h5"))
    pairs = out / "retrieval_top10.txt"
    stage("retrieval_s", lambda: pairs_from_retrieval.main(
        query_global, pairs, num_matched=10, query_list=names, db_list=refs, db_descriptors=ref_global))
    matches = stage("query_superglue_s", lambda: match_features.main(
        p["matcher_config"], pairs, features, matches=out / "matches.h5", features_ref=ref_features))
    rec = pycolmap.Reconstruction(a["sfm_dir"])
    # pycolmap materializes Python-facing maps on access. Reuse the same
    # immutable objects instead of reconstructing the landmark map for every
    # correspondence; XYZ values and correspondence order remain identical.
    images, points3D = rec.images, rec.points3D
    name_to_id = {im.name: i for i, im in images.items()}
    if set(name_to_id) != set(refs):
        raise ValueError("SfM image membership changed")

    class CompatibleLocalizer(localize_sfm.QueryLocalizer):
        def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera):
            if len(points3D_id) < 4:
                return None
            return pycolmap.estimate_and_refine_absolute_pose(
                points2D_all[points2D_idxs], np.asarray([points3D[i].xyz for i in points3D_id]), query_camera,
                estimation_options={"ransac": {"max_error": 12.0, "random_seed": a["seed"]}},
                refinement_options={})

    localizer = CompatibleLocalizer(rec)
    retrieval = parse_retrieval(pairs)
    preds = []
    per_query = out / "predictions"
    per_query.mkdir()
    for q in p["queries"]:
        begin = time.perf_counter()
        name = q["id"] + ".png"
        db_names = retrieval.get(name, [])
        if not set(db_names) <= set(refs):
            raise ValueError("Retrieval contains a nontraining reference")
        K = np.asarray(q["K"])
        cam = pycolmap.Camera(model="PINHOLE", width=q["width"], height=q["height"],
                             params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        ret, log = localize_sfm.pose_from_cluster(
            localizer, name, cam, [name_to_id[n] for n in db_names], features, matches) if db_names else (None, {})
        pose = (ret["cam_from_world"] if ret is not None else
                images[name_to_id[db_names[0]]].cam_from_world() if db_names else None)
        pred = prediction_record(q, pose, pnp=ret is not None, names=db_names,
                                 num_inliers=ret["num_inliers"] if ret is not None else 0,
                                 num_matches=log.get("num_matches", 0), seconds=time.perf_counter() - begin)
        write_new(per_query / (q["id"] + ".json"), pred)
        preds.append(pred)
        print(json.dumps(dict(event="initializer_progress", scene=p["scene"], split=p["split"],
                              completed=len(preds), total=len(p["queries"]), query_id=q["id"])), flush=True)
    # Query images remain frozen through extraction and inference.
    for q in p["queries"]:
        if sha(q["image_path"]) != q["image_sha256"]:
            raise ValueError("Query RGB changed during inference")
    if smoke_qualification is not None:
        original = read(p["smoke_original_predictions"]["path"])
        parity = qualify_pose_parity(preds, original, p["smoke_tolerances"])
        smoke_qualification.update(pose_parity=parity, passed=parity["passed"])
        write_new(out / "SMOKE_QUALIFICATION.json", smoke_qualification)
        if not parity["passed"]:
            raise ValueError("Fresh public query poses differ from the original geometric initializer")
    write_new(out / "predictions.json", preds)
    outputs = [out / "predictions.json", features, query_global, pairs, matches]
    summary = dict(status="complete", method=p["recipe"]["method"], query_count=len(preds),
                   plan_sha256=sha(plan_path), artifacts_sha256={str(f): sha(f) for f in outputs},
                   source_bindings_sha256=p["source_sha256"], reference_assets=a["bindings_sha256"],
                   photometric_refinement=False, gaussian_map_loaded=False,
                   acceptance_status="uncalibrated", no_query_ground_truth_access=True,
                   pnp_count=sum(x["status"] == "pnp" for x in preds),
                   fallback_count=sum(x["status"] == "retrieval_top1_fallback" for x in preds),
                   timings=timing, total_seconds=time.perf_counter() - started,
                   query_feature_reuse=feature_reuse,
                   fresh_feature_query_count=len(preds)-feature_reuse["reused_query_count"],
                   pycolmap=pycolmap.__version__, torch=torch.__version__,
                   smoke_qualification=smoke_qualification,
                   timing_boundary="Batch query computation; reference descriptors reused. Not online p50/p95.")
    write_new(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan), indent=2))


if __name__ == "__main__":
    main()
