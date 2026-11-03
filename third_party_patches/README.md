# Evaluated renderer runtime

The evaluated QGVI renderer used a locally modified HorizonGS checkout. A clean
checkout at the same commit is **not** the same renderer. This directory preserves
the relevant modifications without exporting unrelated training changes.

Apply `HorizonGS-runtime.patch` to HorizonGS commit
`56f9c82a4e84796ba7c496429504dd4ab68127f5`, then run:

```bash
git -C external/HorizonGS apply --check ../../third_party_patches/HorizonGS-runtime.patch
git -C external/HorizonGS apply --whitespace=nowarn ../../third_party_patches/HorizonGS-runtime.patch
python third_party_patches/verify.py external/HorizonGS
```

The patch retains the evaluated source text of these functions:

| File | Included runtime changes |
| --- | --- |
| `gaussian_renderer/render.py` | Exact evaluated file: bounded explicit/neural projection, gsplat 1.5 radii handling, empty-view sentinel |
| `scene/basic_model.py` | Neural sparse-offset generation; explicit feature selection and cached masks |
| `scene/lod_model.py` | Constructor, anchor and explicit-Gaussian visibility masks, bounded explicit PLY loading |
| `utils/streaming_ply.py` | Exact binary PLY layout type and strict loader only |

The unchanged QGVI adapter calls `GaussianLoDModel`, the PLY/MLP loaders, these
visibility and generation functions, and `render`. It builds its own calibrated
camera structure. It does not instantiate HorizonGS `Scene` or `Camera`, load
training observations, train a Gaussian model, save a PLY, merge chunks, or write
videos. Those local modifications are excluded. Original upstream imports needed
by retained training functions remain in place. This patch reproduces the selected
runtime functions, not the full locally modified training checkout.

`HorizonGS-runtime-provenance.json` records upstream, evaluated, and patched file
hashes, plus exact function hashes. `gsplat-source-provenance.json` records the
source archive and all archived source-file hashes. Neither record contains local
machine paths. HorizonGS derivative code remains subject to the complete
`HorizonGS-LICENSE.md`; this is not a permissive open-source relicensing.

## Rasterizer and dependencies

The evaluated environment used **nerfstudio-project/gsplat 1.5.3**, commit
`2323de5905d5e90e035f792fe65bad0fedd413e7`. The original GitHub source ZIP carries
that commit in its archive comment. All 154 archived files match the installed
editable source, and no additional source files were found. No gsplat source patch
is needed. This is not the `tongji-rkr/gsplat-2dgs` rasterizer named by the older
HorizonGS environment file.

For a source rebuild, initialize the commit-pinned GLM submodule and build against
the selected PyTorch/CUDA installation. The local archive had an empty GLM
submodule folder; the historical compiler/header provenance is not established by
the source comparison. We do not claim binary-identical CUDA extensions.

The observed environment had PyTorch 2.9.1, torchvision 0.24.1, numpy 1.26.4,
gsplat 1.5.3, torch-scatter 2.1.2, plyfile 1.1.3, einops 0.8.1, kornia 0.8.0,
OpenCV 4.10.0.84, scikit-learn 1.7.2, matplotlib 3.10.8, colorama 0.4.6,
tqdm 4.66.5, ninja 1.13.0 and jaxtyping 0.3.7. These are observed versions,
not a claim that every package needs an exact version pin. `torch-scatter` is an
import-time dependency of the unmodified HorizonGS model module. Its installed
metadata reports 2.1.2 from a local source directory that is no longer available;
local source modifications and build flags therefore could not be verified.

## Validation scope

The patch applied to clean files from the pinned upstream commit. All four patched
files parse, all selected definition hashes match the evaluated implementation,
and 29 existing CPU tensor tests passed against a temporary patched checkout
(seven unrelated tests deselected). These cover visibility masks, sparse versus
dense neural generation, explicit feature gathering, and mocked projection paths.
CUDA devices were hidden during those tests. Strict PLY parsing is tested
separately. Fresh CUDA compilation, GPU rendering and checkpoint numerical output
parity remain untested by this release validation; the tests must not be described
as a new live GPU qualification.
