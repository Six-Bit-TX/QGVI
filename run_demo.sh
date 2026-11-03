#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  "${PYTHON:-python3}" -m venv .venv
fi
if [[ ! -f .venv/qgvi-core-ready ]] || ! cmp -s requirements.txt .venv/qgvi-core-ready; then
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/qgvi-core-ready
fi
exec .venv/bin/python -m runtime.cli demo "$@"
