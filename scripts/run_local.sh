#!/usr/bin/env bash
# Run the Streamlit app locally. Creates/reuses .venv and installs deps if needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON=".venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi

if ! "$PYTHON" -c "import streamlit, librosa, gspread" 2>/dev/null; then
  echo "Installing dependencies..."
  "$PYTHON" -m pip install -q -U pip
  "$PYTHON" -m pip install -q -r requirements.txt
fi

if [[ ! -f .streamlit/secrets.toml && "${DEVMODE:-false}" != "true" ]]; then
  echo "Warning: .streamlit/secrets.toml not found and DEVMODE is not set. Login and admin" \
       "panel will not work until you add [connections.gsheets]/[admin] secrets, or run" \
       "with DEVMODE=true to use the local CSV backend (scripts/test_roster.csv)." >&2
fi

echo "Starting Streamlit app at http://localhost:8501 ..."
exec .venv/bin/streamlit run app.py --server.port "${PORT:-8501}" "$@"
