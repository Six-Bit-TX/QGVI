# Checkpoints and scene assets

The release preserves the original checkpoint and feature-map bytes. It does
not convert Gaussian files, retrain models, change the matching recipe, or
choose a CPU/GPU device. A scene's geometry, scales, and reference identities
remain explicit in its `scene.json` and `reference_cameras.json`.

`checkpoints/manifest.json` records the exact byte size and SHA-256 of every
asset. Its `huggingface.revision` is an immutable commit once publication is
complete. A missing revision means the inventory is prepared but the download
release is not yet published; the downloader fails clearly in that case.
`huggingface.available_scenes` lists scenes that have passed remote checksum and
size verification. A partial upload does not qualify the remaining scenes.
Private checkpoint repositories require an authorized Hugging Face login.

## Download and verify

```bash
python -m pip install huggingface_hub
python scripts/checkpoints.py list --scene sf
python scripts/checkpoints.py download --scene sf --destination assets
python scripts/checkpoints.py verify --scene sf --destination assets
```

Use `--group localization` for RGB localization without a Gaussian asset;
use `--group gaussian` for Gaussian files only. Omit `--scene` to select all
nine scenes, or repeat it to select several. Both group selections include the
shared scene configuration and reference-camera metadata. Download both groups
for the complete scene demo. No full dataset image archive is downloaded implicitly.

Interrupted network transfers resume through the Hugging Face cache on the next
invocation. Completed files are independently checked with SHA-256 and reused.
Files that differ from the manifest are never silently overwritten. The
`--offline` option verifies existing selected files without contacting the Hub.
`--endpoint` supports an explicitly chosen Hub endpoint; authentication uses
the standard Hugging Face configuration, never a token in the source code.

For a separately published release, pass both `--repo-id owner/name` and
`--revision <40-character-commit-sha>`. Mutable branch names such as `main` are
intentionally rejected. `DOWNLOAD_RECEIPT.json` records the verified selection
and exact repository revision. It records the most recent invocation, not a
claim that every scene has been downloaded.

## Complete inventory

Sizes below are decimal GB, rounded; the manifest has exact sizes. The current
inventory contains 123 files totalling approximately 56.07 GB.

| Scene | Original Gaussian files | Localization assets | Source data |
|---|---:|---:|---|
| SF | 0.444 | 0.707 | UC-GS |
| NYC | 0.552 | 0.630 | UC-GS |
| Colosseum | 0.719 | 0.494 | HorizonGS |
| Elvenruin | 0.995 | 0.942 | HorizonGS |
| Citysample | 1.381 | 2.239 | HorizonGS |
| Block Small | 1.793 | 0.792 | MatrixCity |
| Road | 1.893 | 3.207 | HorizonGS |
| Park | 2.977 | 6.237 | HorizonGS |
| Block A | 22.362 | 7.703 | MatrixCity |

Gaussian files include the original `config.yaml`, PLY, and the three saved
TorchScript MLPs where applicable. Block A uses its original explicit PLY.
Localization assets contain reference SuperPoint features, NetVLAD descriptors,
the complete COLMAP binary sparse map, and reference-camera metadata. Training
databases, query labels, API responses, and original scene image archives are
not part of this inventory.

All asset paths inside a `scene.json` are relative to that file's directory:

```text
scenes/sf/
  scene.json
  reference_cameras.json
  gaussian/config.yaml
  gaussian/point_cloud/iteration_40000/...
  localization/reference_superpoint.h5
  localization/reference_netvlad.h5
  localization/sfm/{cameras,images,points3D,rigs,frames}.bin
```

The `gaussian` section supplies `model_path`, `iteration`, `explicit`, and the
appearance policy. The `rgb` section supplies reference paths, names, seed, and
the native-unit conversion. The local launcher supplies the HorizonGS checkout
and Torch cache locations. Original training paths may still appear inside the
unchanged Gaussian `config.yaml`; the runtime loads only its model parameters.
Use the portable paths from `scene.json`, not those historical training paths.

Native OpenCV camera-to-world matrices are preserved. Apply
`meters_per_model_unit` only where a metric quantity is needed. Five scenes have
DA3-estimated scales; the source and status remain in `scale`. These values are
not newly measured physical ground truth.

## Upstream models and attribution

The following are fetched from their original providers, not mirrored in the
QGVI checkpoint repository:

- [SuperPoint/SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork),
  under the provider's noncommercial research license.
- [NetVLAD Pitts30K](https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat),
  through the pinned HLOC extractor.

The reference descriptors are computed scene assets, not copies of neural
network parameters. Original model code and weights retain their own terms.

Rendering uses [HorizonGS](https://github.com/InternRobotics/HorizonGS), with the
evaluated checkout `56f9c82a4e84796ba7c496429504dd4ab68127f5`.
Localization uses HLOC from [LoGS](https://github.com/RPL-CS-UCL/gs_localization),
checkout `163f7c82d9c8f9ddbc4e3c18a4cf73ba9f2fbced`.
HorizonGS/LoGS inherit the Gaussian-Splatting noncommercial research license;
HLOC has separate Apache-2.0 terms. This does not imply that all source datasets
or learned assets share one permissive license.

## Dataset terms and publication status

- [MatrixCity](https://huggingface.co/datasets/BoDai/MatrixCity) declares
  **CC BY-NC 4.0**. Attribution and noncommercial terms must be retained.
- The [HorizonGS data repository](https://huggingface.co/datasets/BoDai/HorizonGS)
  has no dataset card or explicit dataset license at the time of this audit.
- [UC-GS](https://github.com/SainingZhang/UC-GS) links its synthetic dataset via
  Google Drive; the audited release does not identify a separate dataset
  redistribution license. Its code license is not a substitute for that grant.

The local inventory is not itself a declaration of redistribution permission.
Unresolved source-asset rights must be resolved before publishing those files.
Until then, preserve the inventory locally and link to the original providers.
Do not assign MIT, Apache-2.0, or CC BY to all scene assets by inference.

The audited HorizonGS dataset repository contains source scene ZIP archives,
not the exact trained PLY/MLP checkpoints in this inventory. Its files therefore
cannot replace these checkpoints with a byte-identical direct download.
