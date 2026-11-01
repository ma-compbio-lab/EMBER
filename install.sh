#!/bin/bash
# Create the Ember virtual environment with uv (https://docs.astral.sh/uv/).
# Usage: ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Python 3.12, pinned deps from uv.lock, torch 2.9.1 (CUDA 12.6), ember in editable mode.
uv sync --extra dev --extra plot

echo
echo "Done. Activate with:  source .venv/bin/activate"
