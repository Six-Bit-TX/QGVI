---
language:
- en
tags:
- gaussian-splatting
- visual-localization
- qgvi
license: other
license_name: scene-specific-upstream-terms
license_link: https://huggingface.co/TimSYQQX/QGVI-checkpoints/blob/main/LICENSES.md
---

# QGVI: original scene checkpoints and localization assets

This repository preserves the original nine-scene assets used in QGVI. These
are existing HorizonGS checkpoints and computed localization maps; they are
**not newly trained QGVI neural-network weights**. The underlying files are
preserved byte for byte, with size and SHA-256 recorded in `manifest.json`.

The repository is private while source-asset redistribution terms are reviewed.
Private storage is not a grant to redistribute the data. The original source
providers and applicable terms are listed in `LICENSES.md`. Do not assume the
QGVI source-code license applies to these assets.

## Contents

Each completed scene contains:

- The original HorizonGS PLY and saved TorchScript MLPs, plus original model
  configuration; Block A uses an explicit Gaussian PLY.
- Reference SuperPoint features and NetVLAD global descriptors.
- A COLMAP sparse map with triangulated landmarks and calibrated cameras.
- Portable `scene.json` and `reference_cameras.json` files with native OpenCV
  camera-to-world coordinates and the original per-scene scale provenance.

The complete planned inventory is 123 files, approximately 56.07 GB. Uploads
are committed and independently verified scene by scene. Only scenes listed in
`REMOTE_AVAILABILITY.json` have completed remote verification. An inventory
entry alone does not mean a file has been uploaded.

Training image archives, query ground-truth labels, private API records, and
upstream SuperPoint/SuperGlue/NetVLAD network weights are not included. Obtain
those network weights from their original providers under their own terms.

## Use

Use the `scripts/checkpoints.py` downloader in the QGVI code release. Select one
scene or all nine; it requires a fixed Hub commit and verifies every file.
The portable paths in each scene configuration are relative to that
configuration's directory. Supply the local HorizonGS checkout and Torch-cache
locations from your installation. This asset release does not choose a device
or claim CPU/GPU interchangeability for every upstream component.

The localization assets support QGVI-I's geometric initializer and distinct
keypoint gate. The HorizonGS files support QGVI-A's Gaussian rendering tools.
Additional native LoGS Gaussian maps for full LoGS photometric refinement are
not included in this inventory.

## Coordinates and limitations

Scene poses use their original model units. Five scenes have DA3-estimated
scale factors; the manifest retains their estimated status. These estimates
must not be described as independently measured physical ground truth.
The original unchanged Gaussian configuration may mention historical training
paths. Use the relative paths in `scene.json` to load assets on another machine.

## Attribution

- [QGVI project](https://six-bit-tx.github.io/qgvi-project/)
- [HorizonGS](https://github.com/InternRobotics/HorizonGS), evaluated source commit
  `56f9c82a4e84796ba7c496429504dd4ab68127f5`.
- [LoGS](https://github.com/RPL-CS-UCL/gs_localization), evaluated source commit
  `163f7c82d9c8f9ddbc4e3c18a4cf73ba9f2fbced`, including HLOC.
- [MatrixCity data](https://huggingface.co/datasets/BoDai/MatrixCity).
- [HorizonGS data](https://huggingface.co/datasets/BoDai/HorizonGS).
- [UC-GS](https://github.com/SainingZhang/UC-GS).

Please cite the corresponding original methods and scene datasets when using
these assets. Their own licenses and restrictions remain in force.
