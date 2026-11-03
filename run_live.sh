#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode="${1:-image}"
case "$mode" in
  image|agent-image)
    ./setup.sh
    .venv/bin/python scripts/prepare_networks.py
    .venv/bin/python scripts/checkpoints.py download --scene sf --group localization --destination assets
    if [[ "$mode" == image ]]; then
      exec .venv/bin/python -m runtime.cli localize --scene assets/scenes/sf/scene.json --query examples/sf-query.json
    fi
    exec .venv/bin/python -m runtime.cli agent --scene assets/scenes/sf/scene.json --query examples/source-query.json --index examples/sf-rgb-gallery.json --provider-command '.venv/bin/python scripts/codex_provider.py'
    ;;
  render|agent)
    ./setup_render.sh
    .venv/bin/python scripts/checkpoints.py download --scene sf --group gaussian --destination assets
    if [[ "$mode" == render ]]; then
      exec .venv/bin/python -m runtime.cli render --scene assets/scenes/sf/scene.json --frame examples/render-frame.json
    fi
    .venv/bin/python scripts/codex_provider.py --probe
    exec .venv/bin/python -m runtime.cli agent --scene assets/scenes/sf/scene.json --query examples/target-query.json --provider-command '.venv/bin/python scripts/codex_provider.py'
    ;;
  *) printf 'Usage: %s image|agent-image|render|agent\n' "$0" >&2; exit 2 ;;
esac
