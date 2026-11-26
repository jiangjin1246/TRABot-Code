#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PID_FILE="${RUNTIME_DIR}/web.pid"
LOG_FILE="${RUNTIME_DIR}/web.log"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(tr -cd '0-9' < "${PID_FILE}")"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "FlashHead G1 realtime is already running as PID ${OLD_PID}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

for unsafe_pattern in \
  '/gemini_joint_webui/g1_bridge/build/g1_joint_motion_cli' \
  '/g1_arm_action_example' \
  'webrtc_app_gemini_http.py'; do
  if pgrep -af "${unsafe_pattern}" >/dev/null 2>&1; then
    echo "Refusing to start while an old G1 controller is present: ${unsafe_pattern}" >&2
    pgrep -af "${unsafe_pattern}" >&2 || true
    exit 1
  fi
done

nohup bash "${PROJECT_ROOT}/scripts/run_web.sh" >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" >"${PID_FILE}"

for _ in {1..120}; do
  if ! kill -0 "${PID}" 2>/dev/null; then
    echo "FlashHead G1 realtime failed; inspect ${LOG_FILE}" >&2
    tail -80 "${LOG_FILE}" >&2 || true
    rm -f "${PID_FILE}"
    exit 1
  fi
  # Diagnostics is bound immediately before the upstream Web app starts.  Do
  # not report success during that narrow window: require both the backend
  # preflight and the actual browser endpoint to answer.
  if curl --silent --fail --max-time 1 \
      "${OVERLAY_DIAGNOSTICS_URL}/health" >/dev/null \
      && curl --silent --fail --max-time 1 \
      "http://${WEB_HOST}:${WEB_PORT}/" >/dev/null; then
    echo "FlashHead G1 realtime ready at http://${WEB_HOST}:${WEB_PORT}"
    echo "No robot control is acquired until an approved semantic action is planned."
    exit 0
  fi
  sleep 0.5
done

echo "FlashHead G1 realtime did not become healthy; inspect ${LOG_FILE}" >&2
bash "${PROJECT_ROOT}/scripts/stop_all.sh" || true
exit 1
