# QGVI-A full evaluation

Execute in order: **250 image-and-instruction target-view queries**, **3,678 public RGB test queries**, then **180 text-only descriptions**. Use GPT-6.1 Sol, low effort, requested normal/default service, with the ordinary-localizer early-exit option enabled. This requested configuration is not proof of the served model/tier. Freeze the runtime configuration and all task/code bindings before inference. Baseline predictions remain unchanged.

`prepare_full.py` creates `prepared/{target250,rgb,text}/plan.json` and per-query tasks. It hashes and checks source pixels and dimensions using CPU only. Plans enumerate exact task hashes. Repeating preparation accepts byte-equivalent data; changing a frozen task/plan fails. The 250 targets share 125 source pairs. Their explicit predicates are task requirements, not gallery labels. No accepted-gallery IDs, target witnesses, anchor boxes, reviewer explanations, source GT pose, or other correctness labels enter an inference task.

RGB tasks use only the original public test pixels, hashes, calibration and generic source-localization instruction. Validation is excluded. IDs are namespaced by scene. All 3,678 queries remain, including 1,839 negatives. Text tasks use only the original 180 descriptions and scene identifiers; the full verbatim description is one required predicate. They have no source image, intrinsic matrix or source handle. Do not supply the text author's source image or evaluate a text answer by distance to the author's source camera.

Private truth is stored under `evaluation/`, with mode 0600. `rgb_ground_truth.json` contains in/out-of-map labels, native c2w for positives, and frozen scale provenance. `target250_source_ground_truth.json` is solely for a secondary diagnostic on source estimates produced by `estimate_pose`. **Never compare the final target-view camera against this source pose.** Private files are scorer inputs, not controller inputs. This is procedural separation in a shared filesystem, not OS isolation.

The RGB scorer reads `runs/rgb/executions/{id}/result.json`. It verifies task and result content hashes before computing camera-centre Euclidean error and SO(3) angle. Translation is retained in native scene units and converted using the unchanged scene scale. Inclusive 1 m/2° and 5 m/5° success use all in-map queries. Four source-verified scales and five estimated scales are reported separately; estimated-scale metre results do not establish physically verified accuracy. Native error summaries are most meaningful within each scene.

FAR is accepted negatives divided by all negatives. FRR is resolved rejected positives divided by all positives. Incorrect accepted positive poses are pose failures, not binary rejection failures. Accepted-pose risk includes both inaccurate accepted positives and accepted negatives. Missing, malformed, interrupted and infrastructure/provider-failed jobs remain unresolved; they are never counted as ordinary abstentions. Partial reports withhold complete headline rates and retain bounds. Deadline/budget abstentions are measured outcomes. A supported answer preserved at a normal timeout can still be scored.

The existing 40-distinct-inlier gate is not newly calibrated for adaptive retries. Report acceptance, FAR and abstention; an abstention is not an out-of-map certificate. Negative examples must not be silently skipped or used to tune the test policy.

Target250 and text answers require separate blinded final-render assessment. Use the source image for target250 identity constraints and no source image for text. Report all-required-constraints success, render usability, uncertain judgments, answer coverage and failures. Same-family model assessments are not human ground truth. Every frozen query remains in its denominator. The current agent returns one continuous camera, so text comparison is Hit@1; a five-return metric requires a separately frozen multi-output protocol. Old bank-image labels cannot score new continuous renders. Group target uncertainty by source pair; stratify text uncertainty by scene.

Report model/tool work, source feature preparation, renderer/map setup, and queue wait separately. A reconstructed cached-controller-plus-preparation total is not a directly timed new-input request. Reuse immutable reference assets and exact, hash-qualified source caches. Do not import previous controller decisions, semantic labels, or mismatched PnP recipes as fresh results. The full runner must pass real tool/render smoke under the shared GPU lease before production; these preparation/scoring programs never acquire a GPU or call a model.

CPU preparation:

```bash
python -B prepare_full.py
python -B -m unittest -v test_prepare_full test_score_rgb
python -B score_rgb.py --freeze-scoring
python -B score_rgb.py --output evaluation/rgb_scores.json
```

The last command is also safe before execution: all absent outputs remain explicitly missing, with no accuracy claim.
