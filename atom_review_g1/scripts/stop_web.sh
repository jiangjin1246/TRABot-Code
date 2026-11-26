#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/runtime/web.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "G1 Atom Player is not running"
  exit 0
fi
PID="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ ! "$PID" =~ ^[0-9]+$ ]]; then
  echo "Invalid PID file: $PID_FILE" >&2
  exit 1
fi
if kill -0 "$PID" 2>/dev/null; then
  kill -TERM "$PID"
  for _ in {1..150}; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.1
  done
fi
if kill -0 "$PID" 2>/dev/null; then
  echo "Server did not stop within 15 seconds; PID file was kept: $PID_FILE" >&2
  exit 1
fi
rm -f "$PID_FILE"
echo "G1 Atom Player stopped"
