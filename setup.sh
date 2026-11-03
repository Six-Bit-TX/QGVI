#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
"${PYTHON:-python3}" -m venv .venv
.venv/bin/python -m pip install -r requirements-live.txt
.venv/bin/python scripts/install_upstream.py --logs
.venv/bin/python -m runtime.cli doctor
