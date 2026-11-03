# Environment

The additive launchers use Linux and Python 3.10. Original code includes Linux file locks and POSIX deadlines; Windows is not qualified. Use WSL2 or Linux instead of assuming a native Windows port.

`requirements.txt` pins the lightweight text/PnP/demo environment. `requirements-live.txt` additionally records the evaluated Torch 2.9.1 / torchvision 0.24.1, OpenCV and supporting libraries. `environment.yml` provides a Conda entrypoint. `setup.sh` creates a virtual environment without changing the workstation's research environments.

No launcher exports `CUDA_VISIBLE_DEVICES`, forces CPU, enables hidden fallback, or changes the original model recipe. The original source-preparation wrapper requires CUDA. PnP and text retrieval do not. Hardware and driver/toolkit availability determine which stages can run.

HorizonGS is pinned to `56f9c82a4e84796ba7c496429504dd4ab68127f5`, with the evaluated runtime changes recorded in `third_party_patches/`. Its original upstream `environment.yml` uses Python 3.8.19, PyTorch 2.1.2 and CUDA toolkit 11.8; that is not the evaluated QGVI environment. The evaluated runtime uses `nerfstudio-project/gsplat` commit `2323de5905d5e90e035f792fe65bad0fedd413e7` (1.5.3) and torch-scatter 2.1.2 with Torch 2.9.1. The installed gsplat sources were verified byte-identical to that source archive. `setup_render.sh` installs these dependencies without choosing a CUDA device. Compiled extensions must match the target CUDA/PyTorch build; this release has not freshly qualified every compiler combination.

LoGS is pinned to `163f7c82d9c8f9ddbc4e3c18a4cf73ba9f2fbced`. Only its hloc feature/matching code is needed for the geometric initializer. The release invokes the existing direct pycolmap solver, preserving compatibility with `pycolmap==3.13.0`; upstream LoGS's older absolute-pose API is not substituted.

Network weights are prepared before inference by `scripts/prepare_networks.py`. The evaluated source runtime deliberately disallows surprise downloads during timed inference. QGVI-A's model command is configured separately; no private authentication files are included.
