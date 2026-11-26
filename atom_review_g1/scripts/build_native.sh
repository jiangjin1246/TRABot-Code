#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_ROOT="${UNITREE_SDK_ROOT:-$ROOT/../external/unitree_sdk2}"

cmake -S "$ROOT/g1_bridge" -B "$ROOT/build" \
  -DUNITREE_SDK_ROOT="$SDK_ROOT" \
  -DG1_ATOM_PROJECT_ROOT="$ROOT" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$ROOT/build" --target g1_atom_motion_cli -j2
"$ROOT/build/g1_atom_motion_cli" --help
