# Matched target-view controls, 2 November 2026

Evaluate the unchanged 250 image-and-instruction tasks from all nine scenes.
The source image identifies the anchor; frozen task predicates specify the
requested output. These are development-cohort, model-assessed results, not
human ground truth. The existing adaptive QGVI-A outputs remain unchanged.

## Controls

1. **Retrieve + Select.** Render the same initial six BM25 candidates using each
   task's intrinsics and dimensions. GPT-6.1 Sol, low effort, sees SOURCE plus
   the available candidate renders in one call and returns one supported
   existing view or insufficient evidence. There is no navigation.
2. **Fixed Search + Verify.** Start from the same first BM25 candidate. Inspect
   six precommitted cameras: anchor, yaw left/right 10 degrees, then left/right/
   backward one training-camera-derived step. All offsets are independently
   relative to the anchor. The same low-effort model checks each current render
   and stops at the first fully supported output. The model cannot choose the
   next camera, see future views, rewrite predicates, or consult prior judgments.

Each arm attempts at most six render slots; invalid or out-of-bounds slots are
retained and not replaced. The controls use at most one/six model calls,
respectively. Actual budgets differ and will be reported; we do not equalize
runtime by adding dummy calls. The fixed search is a deterministic control,
not a reproduction of a separately published method. Neither control adds
photometric optimization or a new captioning pass.

Candidates were frozen before new baseline model outcomes. Exact old renders
are reused only with matching complete camera, intrinsics, dimensions, renderer
binding and pixel hashes. Novel cameras use the unchanged Gaussian renderer.
Existing adaptive outputs are not used to choose fixed camera movements.

## Assessment and reporting

Every returned view is independently assessed with GPT-6.1 Sol high using the
same final-render rubric as the existing 250-query adaptive result. Review
sessions receive only the source and final pixels, instruction, required
predicates and schema, with no method, confidence, controller evidence, camera
ground truth, or other assessments. This is an independent *call*, not an
independent model family. Reports retain success, failure and uncertainty.
Historical reviews may be reused only after exact input/rubric/model/effort
verification; identical new assessments are deduplicated across the two arms.

Report returned fraction, confirmed success / all 250, precision bounds among
returns, and utility bounds [(C-W-U)/N, (C+U-W)/N], with C/W/U denoting confirmed
correct/incorrect/uncertain returns. Abstention contributes zero utility.
Missing or infrastructure-failed jobs are unresolved; they never count as
semantic abstentions or wrong answers. No final metric is claimed until all
executions and every returned output's assessment are complete.

Report per-family results and paired differences against frozen QGVI-A; paired
uncertainty and statistical intervals must be distinguished. A gain over these
controls would support adaptive search on this cohort, not general superiority
of agentic localization or Gaussian representations.

## Execution and recovery

Use visible persistent Codex sessions under `/home/xiongti1/gs_qry`, normal
service tier, tools disabled for visual decisions and judgments. The new
adapter transport differs from the historical coordinator/subagent transport;
therefore latency is not treated as a controlled historical comparison.
Four maximum concurrent requests use the prior measured healthy setting, with
an actual three-scene/six-execution smoke before full execution.

Each call persists its exact input and session/turn identity. Capacity and rate
limits back off exponentially with bounded jitter and resume automatically.
An ambiguous request resumes the same attempt; a new attempt requires a
verified terminal failed/interrupted predecessor. Malformed outputs have a
bounded three-attempt budget. Completed outputs are immutable across restart.

AlloMap continues uninterrupted. A separate renderer uses its own lock, one
scene at a time, a 48 GiB Torch allocator cap, 70 GiB free memory before model
load and 20 GiB between renders, and at most two renders/second. Resource
pressure defers unfinished assets and retries; no AlloMap locks or jobs are
changed. These are shared-GPU preparation timings, not isolated online latency.
The initial 24 GiB cap was raised after a repeatable Block A allocation failure;
the unchanged failing camera passed the revised resource check. Three allocator
failures without progress now require inspection; temporary physical memory
pressure continues to back off. No camera, rendering precision or checkpoint
changed during this resource recovery.

The progress service is read-only at port 8788, refreshes every five seconds,
and separates assets, baseline executions and independent reviews. Worker PID,
heartbeat, retries and unresolved jobs must remain visible.
