# Queryable Gaussian View Index (QGVI)

Text retrieval, selective RGB camera localization, and bounded agentic view search inside posed 3D Gaussian assets.

[Project page](https://six-bit-tx.github.io/qgvi-project/) · [Paper and interactive examples](https://six-bit-tx.github.io/qgvi-project/#paper) · [Checkpoint instructions](docs/CHECKPOINTS.md)

## Original implementation, preserved

`research/original/` contains **51 original source, protocol and test files**, copied byte for byte from the evaluated research implementation. The source manifest records every hash. `runtime/` adds paths, launch commands and a model-command interface around those files. It does not rewrite the algorithms, convert checkpoints, force CPU execution, or override GPU visibility.

- **QGVI-T:** original BM25 caption retrieval. No MiniLM or query-time VLM filter.
- **QGVI-I:** NetVLAD → SuperPoint/SuperGlue → pooled SfM correspondences → PnP → at least 40 distinct query inliers. The gate preserves the accepted pose. Photometric optimization is not the default.
- **QGVI-A:** original bounded controller, evidence checks, camera proposals/navigation and early stopping. The external model transport is supplied separately from the controller.

The original research workspace is not needed to run the bundled examples. See [implementation details](docs/IMPLEMENTATION.md) for source paths, preserved constraints and the distinction between the research evaluation cache and fresh-image inference.

## One-command demonstration

On Linux with Python 3.10 or 3.11 and `venv`:

```bash
./run_demo.sh
```

This creates an isolated environment and opens `http://127.0.0.1:7860`. The included 78-view sample supports live browser caption search; the RGB and agent panels are **recorded demonstrations**, with real inputs, outputs and camera poses. They do not claim to process a new uploaded image or run a new model call. Nothing changes your GPU settings. Use `--port 7861 --no-browser` on a remote machine.

Run the original Python computations after installation:

```bash
.venv/bin/python -m runtime.cli text 'black glass building red signs' --scene block_a
.venv/bin/python -m runtime.cli pnp
```

The second command genuinely recomputes PnP and the gate on the included SF correspondences; feature extraction and matching for this small example were performed previously. The example images are not a benchmark and should not be used to claim published accuracy.

## Full live environment

With scene-download access and the required CUDA/compiler environment configured, the complete SF examples have launchers:

```bash
./run_live.sh image    # Fresh NetVLAD, SuperPoint, SuperGlue, PnP and gate
./run_live.sh agent-image # Image route with geometric early stopping
./run_live.sh render   # Actual frozen Gaussian checkpoint
./run_live.sh agent    # Text-to-view agent with your Codex login
```

The initial run installs dependencies and downloads the selected assets. Live CUDA installation has not been freshly qualified on a clean machine; see the validation limits below. The individual steps are available for inspection and custom inputs:

```bash
./setup.sh
.venv/bin/python scripts/prepare_networks.py
.venv/bin/python -m runtime.cli doctor
```

This installs the pinned Python dependencies and official LoGS source. Network weights are fetched from their original providers, under their original terms. It leaves device selection to the original implementation and your environment. The evaluated fresh-image initializer and HorizonGS renderer require CUDA; this release does not silently replace them with a different CPU algorithm. Install an appropriate PyTorch build yourself if the supplied standard wheel is unsuitable for your hardware.

[Download the exact scene assets](docs/CHECKPOINTS.md), then run a **new image** through the preserved initializer:

```bash
.venv/bin/python scripts/checkpoints.py download --scene sf --group localization --destination assets
.venv/bin/python -m runtime.cli localize \
  --scene assets/scenes/sf/scene.json --query examples/sf-query.json
```

For your own image, copy `examples/sf-query.json`, supply its actual pixel dimensions and calibrated `K`, change `image_path`, and clear or replace `excluded_reference_names`. Output is native OpenCV camera-to-world pose in scene units. A failed gate yields `accepted_pose_c2w: null`; the raw hypothesis remains diagnostic only. The launcher does not claim that a custom input is held out from reconstruction.

## Live Gaussian rendering and agent

```bash
.venv/bin/python scripts/install_upstream.py --horizon
```

`./setup_render.sh` installs the pinned renderer dependencies and applies the documented runtime patch to the external HorizonGS checkout. Gaussian rasterization extensions must match your installed CUDA toolkit. See `docs/ENVIRONMENT.md` for the evaluated versions.

```bash
.venv/bin/python scripts/checkpoints.py download --scene sf --group gaussian --destination assets
.venv/bin/python -m runtime.cli render \
  --scene assets/scenes/sf/scene.json --frame examples/render-frame.json
```

QGVI-A calls a model command through JSON stdin/stdout. The bundled `scripts/codex_provider.py` uses the original strict CLI adapter and your own `codex login`. It neither reads nor copies authentication files. The controller expects GPT-6.1 Sol at low effort. This public CLI bridge is distinct from the original native-session evaluation transport; model availability depends on your account.

```bash
.venv/bin/python -m runtime.cli agent \
  --scene assets/scenes/sf/scene.json \
  --query examples/target-query.json \
  --provider-command '.venv/bin/python scripts/codex_provider.py'
```

Check local readiness with `.venv/bin/python scripts/codex_provider.py --probe` (no model call). The command contract is in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md). Model access/authentication must be configured by the user; neither a shared key nor model quota is bundled. Rendering uses the actual frozen Gaussian asset. RGB early stopping invokes the actual initializer and may finish without any model call. Source-image candidate inspection additionally requires original reference images in an index with `reference_name` entries; the small public gallery is not the complete image bank. Missing evidence produces an explicit failure instead of a fabricated localization. See the [official Codex non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode) for local login and CLI operation.

## Checkpoints

The prepared nine-scene inventory is approximately **56.07 GB**: 33.12 GB of original Gaussian files plus sparse localization maps/features. Downloads are per scene and per component, resumable and SHA-256 verified. The small SF bundle is about 1.15 GB. Uploads are going to [TimSYQQX/QGVI-checkpoints](https://huggingface.co/TimSYQQX/QGVI-checkpoints), currently **private pending confirmation of source-asset redistribution rights**. A logged-in account with access is required. Uploaded scenes and the pinned revision are recorded in `checkpoints/manifest.json`; staged files are not reported as uploaded.

Third-party SuperPoint/SuperGlue/NetVLAD weights are not mirrored. Source dataset and upstream software terms remain applicable. The checkpoint card distinguishes source-verified scales from DA3 estimates. Native LoGS Gaussian models for historical photometric-refinement ablations are separate from these HorizonGS assets.

## Validation

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q tests
.venv/bin/python research/check_sources.py
.venv/bin/python research/check_sources.py --tests
```

The complete original test suites need PyTorch for tensor fixtures; they do not launch GPU inference. See `VALIDATION.json` for the actual checks run for this release, including tests performed without the research environment. Live model-service access and a fresh CUDA extension build are separate requirements; a cached demonstration is not evidence that those installations have passed.

## Attribution and use

QGVI builds on HorizonGS, LoGS, hierarchical localization, NetVLAD, SuperPoint, SuperGlue, COLMAP, and the MatrixCity, UC-GS and HorizonGS scene collections. See [third-party notices](THIRD_PARTY.md). Original third-party licenses are retained in their upstream checkouts and the checkpoint notices. This public source snapshot does not relicense third-party code, network parameters or scene data.

Please cite the paper linked from the project page. An arXiv identifier has not yet been assigned in this release.
