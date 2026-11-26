#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

curl --silent --fail "${SIM_SERVICE_URL}/health" >/dev/null
curl --silent --fail "http://127.0.0.1:${WEB_PORT}/" >/dev/null
curl --silent --fail "${OVERLAY_DIAGNOSTICS_URL}/health" >/dev/null

curl --silent --fail \
  -H 'Content-Type: application/json' \
  --data '{"action":"joint_presentation_gesture","mode":"replace","id":"smoke-presentation"}' \
  "${SIM_SERVICE_URL}/actions/trigger" >/dev/null

sleep 1
curl --silent --fail "${SIM_SERVICE_URL}/frame" --output "${RUNTIME_DIR}/smoke_frame.png"
curl --silent --fail "${SIM_SERVICE_URL}/state" >"${RUNTIME_DIR}/smoke_state.json"

"${G1_MUJOCO_PYTHON}" -c '
import json, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
action = state.get("action")
last = state.get("last_completed")
if not action and not last:
    raise SystemExit("motion was neither running nor completed")
if len(state.get("q", {})) < 29:
    raise SystemExit("expected the complete 29-DoF simulator state")
' "${RUNTIME_DIR}/smoke_state.json"

echo "Smoke test passed."
echo "Frame: ${RUNTIME_DIR}/smoke_frame.png"
echo "State: ${RUNTIME_DIR}/smoke_state.json"
