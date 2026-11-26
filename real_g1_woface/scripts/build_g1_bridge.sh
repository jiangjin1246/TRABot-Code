#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SDK_ROOT="${UNITREE_SDK_ROOT:-${PROJECT_ROOT}/../external/unitree_sdk2}"

require_dir "${SDK_ROOT}"
require_file "${G1_REALTIME_REVIEW_STORE}"
require_dir "${G1_REALTIME_PREVIEW_ROOT}"

cmake \
  -S "${PROJECT_ROOT}/g1_bridge" \
  -B "${PROJECT_ROOT}/g1_bridge/build" \
  -DUNITREE_SDK_ROOT="${SDK_ROOT}" \
  -DG1_ATOM_PROJECT_ROOT="${PROJECT_ROOT}" \
  -DG1_ATOM_REVIEW_STORE_PATH="${G1_REALTIME_REVIEW_STORE}" \
  -DG1_ATOM_PREVIEW_ROOT="${G1_REALTIME_PREVIEW_ROOT}"
cmake --build "${PROJECT_ROOT}/g1_bridge/build" --parallel

echo "Built ${PROJECT_ROOT}/g1_bridge/build/g1_atom_motion_cli"
