# Original implementation and public adapters

This release keeps the research implementation in `research/original/` **byte for
byte**. It does not rewrite the algorithms or change their device rules. The
small `runtime/` layer supplies public paths, assets and an explicit external
model-command interface. The previous research workspace remains intact.

`research/source_manifest.json` lists each original file, its repository-relative
source path, byte size and SHA-256. The files are a snapshot of the working tree,
including the latest uncommitted research changes; the recorded Git commit alone
is not a complete description of their contents. `scripts/export_original.py`
uses an explicit allowlist and checks that the source did not change during
copying. It never discovers or copies credentials, datasets, logs or checkpoints
implicitly.

```bash
python scripts/export_original.py --verify
python research/check_sources.py --tests
```

The fixture tests are CPU-only checks of the original code. They do not claim
that full feature extraction or Gaussian rendering runs without a GPU. They do
not launch a model, touch an existing experiment, or obtain new accuracy results.
Without a PyTorch installation, the checker explicitly skips the optional
tensor-render module and the remaining original renderer tests report their
own PyTorch-dependent cases as skipped. Install your chosen compatible PyTorch
build to run those additional tests; skipped cases are never reported as passed.

## Source map

Paths in this table are relative to `research/original/`.

| Component | Original source | Role |
| --- | --- | --- |
| QGVI-T | `experiments/qgvi_a_full_eval_20261029_v1/retrieval_backend.py` | Whole-caption BM25, source exclusions, candidate ranking and duplicate removal. `BM25` is the current caption ranker. |
| Caption preparation | `qgvi-rag/caption_rag/{pipeline,vision,credentials}.py` | Frozen caption structure, image requests, validation and chunk assembly. No credential values are included. |
| Caption tokenization | `qgvi-rag/caption_rag/retrieval.py` | English/CJK tokenizer used by current BM25. This older module also contains historical MiniLM/A-RAG fusion code; current QGVI-T does not call those embedding paths. |
| QGVI-I initialization | `experiments/agentic_locator_comparison_20261025_v1/geometry_adapter.py` | SuperPoint/SuperGlue correspondence geometry, ROI filtering and PnP. |
| Fresh image processing | `experiments/qgvi_a_full_eval_20261029_v1/source_geometry.py` | NetVLAD extraction, reference ranking, local matching and source preparation. Its original CUDA requirement is retained. |
| Selective gate | `experiments/cvpr2027_rgb/logs_selective_v2/{confidence,calibration}.py` | Reprojection at the unchanged pose, distinct-query support, validation-only gate selection and application. |
| RGB reference entrypoints | `experiments/cvpr2027_rgb/logs_selective_v2/{run_initializer,run_rgb,catalog,prepare,score_policy}.py` | Original initializer, asset qualification, validation preparation and scoring code. Historical benchmark plans must be supplied separately. |
| QGVI-A controller | `experiments/qgvi_a_full_eval_20261029_v1/{controller,state,protocol,early_exit,tool_host}.py` | Unchanged evidence state, seven tools, budgets, early exit, visual acceptance and journal replay. |
| Camera navigation | `experiments/qgvi_a_full_eval_20261029_v1/navigation_backend.py` | Native-coordinate SE(3), local camera increments, render validation and cache bindings. Its original resident worker and machine paths are retained as source, not automatically started. |
| Gaussian rendering | `qgvi/qgvi/multiscene_renderer.py` | HorizonGS checkpoint loading and calibrated RGB/depth/alpha rendering. Model weights are separate; the evaluated HorizonGS runtime delta is preserved in `third_party_patches/` with its upstream license and source provenance. |
| Matched target baselines | `experiments/qgvi_a_target_baselines_20261102_v1/policies.py` | Retrieve + Select and Fixed Search + Verify response contracts and acceptance rules. |

The files `providers.py` and `speed_provider.py` in the original controller tree
retain their original imports. Their older provider/parser dependency is
included, together with the optional CLI adapter source it refers to. The public
command entrypoint does **not** invoke those historical provider transports or
their hard-coded credential-file location. Native Codex session services,
coordinator queues, AlloMap admission logic, retry supervisors and private
evaluation labels are not part of this source snapshot.

## What the original methods do

QGVI-T ranks existing captions using BM25, with no query-time verifier or MiniLM
in its current default. Scores rank candidates; they are not probabilities.

QGVI-I retrieves ten original reference images with NetVLAD, matches SuperPoint
and SuperGlue features, and pools correspondences to triangulated SfM landmarks.
The saved recipe uses SuperPoint with at most 4,096 keypoints, maximum side 1,024,
float16 HLOC feature export, outdoor SuperGlue with five Sinkhorn iterations, and
PnP with a 12-pixel reprojection threshold. The public preparation must use the
matching upstream implementation and assets. Gate application never changes the
estimated pose. Photometric refinement is not enabled by the current geometric
method. The source solver qualification targets pycolmap 3.13.0.

QGVI-A calls those caption and geometric tools from the original staged
controller. A source-camera answer requires matching/PnP provenance and at least
40 distinct inlier query keypoints. A target-view answer requires the actual
render to have been shown, every supplied predicate to be satisfied, and a
supported source-anchor relation. Captions, retrieval scores and PnP support
alone cannot establish target-view semantic correctness. These are controller
acceptance rules, not independent evaluation ground truth.

The early-exit prefix is enabled by the public agent adapter, as in the full
evaluation. Source localization can finish after image retrieval and the
geometric gate without a model call. Target-view search first retrieves and
renders one candidate, then requests a visual check. A failed or uncertain check
hands the existing observations and remaining budget to the agent. It does not
restart retrieval. Missing evidence produces `insufficient_evidence`; no
calibrated out-of-map certificate is supplied by default.

The unchanged defaults allow 12 decisions, six renders and 48 observations, with
at most seven images shown per decision and six candidates per retrieval call.
The stages advance from discovery to verification, refinement and finalization,
with one bounded return to discovery. Saved supported answers survive ordinary
budget/deadline termination. The original controller uses POSIX `SIGALRM`, so
the public adapter must execute in a process's main thread; a web server should
launch the agent CLI in a child process.

Camera transforms are finite, proper OpenCV camera-to-world SE(3) matrices.
Local x is right, y is down, and z is forward. Intrinsics and dimensions remain
fixed for a query. Navigation is a free virtual camera, not a collision-aware
robot simulator. Scene units are not automatically metres, and a world axis does
not establish geographic north. The existing renderer guard requires at least
5% supported depth/alpha coverage.

## External model command

`runtime.agent.run_agent(query, folder, retrieval, navigation, provider_argv,
effort="low")` injects public tools and the command provider into the **unchanged**
`controller.run_one`. The integration restores temporary import bindings after
the call. It never edits the source snapshot.

The command is an argument list executed without a shell. It receives one JSON
object on standard input:

```json
{
  "schema": "qgvi_external_provider_request_v1",
  "requested_model": "gpt-6.1-sol",
  "reasoning_effort": "low",
  "requested_service_tier": "default",
  "fast_mode_enabled": false,
  "image_path": "/absolute/path/to/contact-sheet.jpg",
  "image_sha256": "...",
  "system_prompt": "...",
  "user_payload": {},
  "output_schema": {},
  "response_contract": {}
}
```

The command must inspect the supplied image and return exactly one JSON envelope
on standard output. The `response` value must satisfy the supplied action schema:

```json
{
  "response": {"schema_version": "qgvi_agent_tools_v2.0", "action": "..."},
  "reported_model": "gpt-6.1-sol",
  "reasoning_effort": "low"
}
```

This shortened example is not a valid action by itself: the full schema includes
arguments, assessment, next stage, unresolved predicates and reason. The model
identity and effort must match the request; missing or mismatched metadata is a
provider failure. Supported effort labels can be requested explicitly, but the
external service must support and honor them. Low is the evaluated default.

The command owns its credentials and network client. It is trusted local code;
do not accept command arguments from an untrusted web user. Pass credentials via
the command's own secure configuration/environment rather than command arguments.
The host bounds execution to at most 600 seconds, caps each output stream at
4 MiB, terminates the dedicated process group on failure/timeout, validates JSON
and does not copy stderr into failure messages. It does not silently retry,
change model, or replace the algorithm after a failed model request.

`result.json` remains the original controller output with its original journal
and hash. `public_transport.json` separately binds the command digest and effort
and states that the transport is `external_command`. Model metadata is reported
by that command, not independently attested. The bridge therefore does not claim
to reproduce the evaluation's native Codex session transport or hosted latency.
The function returns `{"result": original_result, "public_transport": receipt}`.

## Asset and runtime boundaries

The original evaluation `RetrievalBackend` loads frozen caption/feature plans
and historical paths. Its image ranking uses cached source NetVLAD descriptors;
its full-image pose request can return cached PnP, while an ROI request re-solves
the stored full-image correspondences without changing intrinsics. These cached
functions must not be presented as fresh arbitrary-image feature inference.
Fresh queries use the source-preparation path and matching map assets.

The public wrappers resolve supplied asset paths; the preserved source still
contains historical machine-specific constants and experiment bindings. Those
are part of the source record, not portable configuration or bundled data.
Original CUDA guards and solver/feature settings remain intact. Checkpoint
availability alone does not provide the SfM map, reference features, captions,
query intrinsics, third-party software or hosted model access required by each
route. Follow the release's asset/environment instructions for those inputs.

The historical protocols included with the snapshot describe the development
evaluation and its limitations. They are not claims of new test-set validation,
human annotations, benchmark independence, or remeasured performance for this
release.
