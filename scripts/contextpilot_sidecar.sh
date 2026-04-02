#!/usr/bin/env bash

# ContextPilot sidecar launcher
#
# Usage:
#   scripts/contextpilot_sidecar.sh [--port PORT] [--backend-url URL] [--no-reorder] [--no-dedup]
#
# Options:
#   --port PORT            HTTP port for the sidecar (default: 8765)
#   --backend-url URL      MoE-Infinity backend base URL (default: http://localhost:8000)
#   --no-reorder           Disable reorder behavior in the sidecar
#   --no-dedup             Disable dedup behavior in the sidecar
#   --help                 Show this help message
#
# Notes:
#   - Uses /usr/bin/python3.10 and /tmp/ContextPilot explicitly.
#   - Validates the backend URL with curl (3s timeout) before starting.

set -euo pipefail

PORT=8765
BACKEND_URL="http://localhost:8000"
REORDER_ENABLED=1
DEDUP_ENABLED=1

usage() {
  sed -n '1,24p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      [[ $# -ge 2 ]] || { echo "Error: --port requires a value" >&2; exit 1; }
      PORT="$2"
      shift 2
      ;;
    --backend-url)
      [[ $# -ge 2 ]] || { echo "Error: --backend-url requires a value" >&2; exit 1; }
      BACKEND_URL="$2"
      shift 2
      ;;
    --no-reorder)
      REORDER_ENABLED=0
      shift
      ;;
    --no-dedup)
      DEDUP_ENABLED=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! curl -sS --max-time 3 -o /dev/null "$BACKEND_URL"; then
  echo "Error: backend URL is not reachable within 3 seconds: $BACKEND_URL" >&2
  exit 1
fi

export CONTEXTPILOT_REORDER_ENABLED="$REORDER_ENABLED"
export CONTEXTPILOT_DEDUP_ENABLED="$DEDUP_ENABLED"

exec PYTHONPATH=/tmp/ContextPilot /usr/bin/python3.10 -m contextpilot.server.http_server --port "$PORT" --infer-api-url "$BACKEND_URL"
