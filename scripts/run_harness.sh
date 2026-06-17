#!/usr/bin/env bash
# Run regression harness before committing. Exits non-zero if any test fails.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON=""
for candidate in .venv/bin/python .venv_test/bin/python python3; do
  if [[ -x "$candidate" ]] || command -v "$candidate" &>/dev/null; then
    if "$candidate" -c "import librosa, pytest" 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "Creating .venv_test for harness..."
  python3 -m venv .venv_test
  PYTHON=".venv_test/bin/python"
  "$PYTHON" -m pip install -q -U pip
  "$PYTHON" -m pip install -q -r requirements.txt pytest
fi

echo "Running tune-matcher harness ($( "$PYTHON" --version ))..."
"$PYTHON" -m pytest tests/test_harness.py -v --tb=short "$@"
