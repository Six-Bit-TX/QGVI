#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
./setup.sh
.venv/bin/python scripts/install_upstream.py --horizon
.venv/bin/python -m pip install ninja wheel setuptools jaxtyping==0.3.7
.venv/bin/python -m pip install --no-build-isolation torch-scatter==2.1.2
.venv/bin/python -m pip install --no-build-isolation 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7'
.venv/bin/python -m runtime.cli doctor
