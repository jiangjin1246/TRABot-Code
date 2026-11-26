#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime"
PID_FILE="$RUNTIME/web.pid"
LOG_FILE="$RUNTIME/web.log"
SDK_ROOT="${UNITREE_SDK_ROOT:-$ROOT/../external/unitree_sdk2}"
SDK_LIB="$SDK_ROOT/thirdparty/lib/$(uname -m)"

mkdir -p "$RUNTIME"
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "G1 Atom Player is already running as PID $OLD_PID"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

set -a
source "$ROOT/config.env"
set +a
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SDK_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nohup python3 "$ROOT/app.py" >>"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  echo "G1 Atom Player failed to start; see $LOG_FILE" >&2
  tail -80 "$LOG_FILE" >&2 || true
  exit 1
fi
echo "G1 Atom Player started: PID $PID, http://${G1_ATOM_HOST:-127.0.0.1}:${G1_ATOM_PORT:-8776}"
