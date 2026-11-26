#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

stop_one() {
  local name="$1"
  local pid_file="${RUNTIME_DIR}/${name}.pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no PID file"
    return
  fi
  local pid
  pid="$(tr -cd '0-9' < "${pid_file}")"
  if [[ -z "${pid}" ]] || [[ ! -r "/proc/${pid}/cmdline" ]]; then
    echo "${name}: stale PID file"
    rm -f "${pid_file}"
    return
  fi
  local command_line
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  local expected_token="app.py"
  if [[ "${command_line}" != *"${PROJECT_ROOT}"* ]] && [[ "${command_line}" != *"${expected_token}"* ]]; then
    echo "${name}: refusing to stop unrelated PID ${pid}: ${command_line}" >&2
    return 1
  fi
  kill "${pid}"
  for _ in {1..40}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${pid_file}"
      echo "Stopped ${name}"
      return
    fi
    sleep 0.25
  done
  echo "${name}: PID ${pid} did not stop after SIGTERM" >&2
  return 1
}

stop_one web
