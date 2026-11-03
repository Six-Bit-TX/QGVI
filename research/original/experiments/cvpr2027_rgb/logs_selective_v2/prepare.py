#!/usr/bin/env python3
"""Freeze public-only validation/smoke plans; CPU preparation does not launch GPU."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from catalog import (EXPECTED_VALIDATION_COUNT, EXPECTED_VALIDATION_COUNTS,
                     FEATURE_CONFIG, MATCHER_CONFIG, RECIPE, ROOT, SCENES, SMOKE_TOLERANCES,
                     VALIDATION_SCENES,
                     canonical_protocol, public_queries, reference_assets, sha, shared_query_rows,
                     source_bindings, write_new)


def prepare(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Preparation destination exists; use a new run version")
    destination.mkdir(parents=True, exist_ok=False)
    sources = source_bindings()
    jobs, smoke_jobs, counts, prepared_plans = [], [], {}, {}
    reused_query_count = 0
    started = time.time()
    for scene in VALIDATION_SCENES:
        source = ROOT / "qgvi/.staging/horizongs_eval_v1" / scene / "queries.json"
        qs = public_queries(source, "validation")
        if len(qs) != EXPECTED_VALIDATION_COUNTS[scene]:
            raise ValueError("Frozen validation membership differs for " + scene)
        print(json.dumps(dict(event="qualifying_reference_assets", scene=scene, queries=len(qs))), flush=True)
        assets = reference_assets(scene)
        plan = dict(schema_version=1, status="prepared_not_launched", scene=scene, split="validation",
                    purpose="validation",
                    validation_scenes=list(VALIDATION_SCENES), test_scenes=list(SCENES),
                    recipe=copy.deepcopy(RECIPE), feature_config=copy.deepcopy(FEATURE_CONFIG),
                    matcher_config=copy.deepcopy(MATCHER_CONFIG), reference_assets=assets,
                    public_queries=dict(path=str(source), sha256=sha(source)), queries=qs,
                    output_dir=str(destination / "validation" / scene / "initializer"), source_sha256=sources,
                    private_labels_read=False, requires_gpu_admission=True,
                    reference_cache_policy="Reuse original training SfM, SuperPoint features and NetVLAD descriptors read-only; never rebuild the reference database",
                    query_cache_policy="Compute missing exact-LoGS validation query stages; incompatible ablation features are not substituted",
                    query_feature_reuse=None,
                    required_smoke_summaries=[dict(scene=s, plan=str(destination / "plans" / (s + "_smoke.json")),
                                                  summary=str(destination / "smoke" / s / "initializer/summary.json"))
                                              for s in VALIDATION_SCENES])
        if scene == "sf":
            producer = prepared_plans["block_a"]
            shared = shared_query_rows(producer["queries"], qs)
            if shared:
                producer_path = destination / "plans/block_a_validation.json"
                plan["query_feature_reuse"] = dict(source_plan=str(producer_path),
                                                   source_plan_sha256=sha(producer_path), rows=shared,
                                                   scope="Copy image-only NetVLAD and SuperPoint rows from completed Block A; recompute SF retrieval, matching and PnP")
                reused_query_count += len(shared)
        plan_path = destination / "plans" / (scene + "_validation.json")
        write_new(plan_path, plan)
        jobs.append(dict(scene=scene, plan=str(plan_path), sha256=sha(plan_path), query_count=len(qs)))
        counts[scene] = len(qs)
        prepared_plans[scene] = plan
        if scene in VALIDATION_SCENES:
            smoke = copy.deepcopy(plan)
            original_predictions = canonical_protocol(scene).parent / "initializer/predictions.json"
            smoke.update(queries=public_queries(source, "test")[:2], split="test", purpose="smoke",
                         output_dir=str(destination / "smoke" / scene / "initializer"),
                         required_smoke_summaries=[], smoke_tolerances=SMOKE_TOLERANCES,
                         query_feature_reuse=None,
                         smoke_original_predictions=dict(path=str(original_predictions), sha256=sha(original_predictions)),
                         scope="Two public test IDs selected in sorted order, for raw implementation parity only; no private labels or policy calibration")
            smoke_path = destination / "plans" / (scene + "_smoke.json")
            write_new(smoke_path, smoke)
            smoke_jobs.append(dict(scene=scene, plan=str(smoke_path), sha256=sha(smoke_path), query_count=2))
        print(json.dumps(dict(event="reference_assets_qualified", scene=scene)), flush=True)
    if sum(counts.values()) != EXPECTED_VALIDATION_COUNT:
        raise ValueError("Block A and SF validation membership must total490")
    result = dict(schema_version=1, status="prepared_not_launched", method=RECIPE["method"],
                  validation_queries=sum(counts.values()), per_scene_queries=counts,
                  validation_scenes=list(VALIDATION_SCENES), test_scenes=list(SCENES),
                  scope="Policy calibration on the complete Block A and SF validation splits; subsequent frozen evaluation on all nine test scenes",
                  smoke_scope="Four public test queries (two per validation scene), for implementation parity only; never calibration",
                  cache_reuse=dict(reference_databases="Reuse existing SfM and all reference descriptors",
                                   test_initializers="Reuse existing nine-scene geometric initializer predictions and query/match caches",
                                   new_validation_query_count=EXPECTED_VALIDATION_COUNT,
                                   fresh_validation_feature_queries=EXPECTED_VALIDATION_COUNT-reused_query_count,
                                   cross_scene_query_feature_rows_reused=reused_query_count,
                                   implementation_parity_smoke_query_count=4,
                                   reference_database_rebuilds=0),
                  validation_jobs=jobs, smoke_jobs=smoke_jobs, source_sha256=sources,
                  private_labels_read=False, gpu_jobs_launched=0,
                  qualification="All reference hashes, exact feature membership/indices, public pixels checked on CPU",
                  created_unix=time.time(), cpu_preparation_seconds=time.time() - started,
                  next_gate="GPU queue admission, two-scene reference descriptor parity plus four-query raw pose parity, independent review before validation")
    write_new(destination / "PREPARED.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
