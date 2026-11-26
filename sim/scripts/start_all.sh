#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

start_one() {
  local name="$1"
  local command="$2"
  local pid_file="${RUNTIME_DIR}/${name}.pid"
  local log_file="${RUNTIME_DIR}/${name}.log"

  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(tr -cd '0-9' < "${pid_file}")"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "${name} is already running as PID ${old_pid}" >&2
      return 0
    fi
  fi

  nohup bash "${command}" >"${log_file}" 2>&1 &
  local new_pid=$!
  echo "${new_pid}" >"${pid_file}"
  echo "Started ${name} as PID ${new_pid}; log: ${log_file}"
}

bash "${PROJECT_ROOT}/scripts/setup_links.sh"
start_one sim "${PROJECT_ROOT}/scripts/run_sim.sh"

for _ in {1..60}; do
  if curl --silent --fail "${SIM_SERVICE_URL}/health" >/dev/null; then
    break
  fi
  sleep 0.5
done
curl --silent --fail "${SIM_SERVICE_URL}/health" >/dev/null || {
  echo "Simulator did not become healthy; inspect ${RUNTIME_DIR}/sim.log" >&2
  bash "${PROJECT_ROOT}/scripts/stop_all.sh" || true
  exit 1
}

start_one web "${PROJECT_ROOT}/scripts/run_web.sh"

for _ in {1..120}; do
  if curl --silent --fail "http://127.0.0.1:${WEB_PORT}/" >/dev/null \
    && curl --silent --fail "${OVERLAY_DIAGNOSTICS_URL}/health" >/dev/null; then
    echo "Web demo is ready on http://127.0.0.1:${WEB_PORT}"
    exit 0
  fi
  sleep 0.5
done

echo "Web demo did not become healthy; inspect ${RUNTIME_DIR}/web.log" >&2
bash "${PROJECT_ROOT}/scripts/stop_all.sh" || true
exit 1
