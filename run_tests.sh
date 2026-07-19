#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python -m pytest "$@"

echo
echo "--- manual real-server check (tests/manual_llm_client_check.py) ---"
python -m tests.manual_llm_client_check
