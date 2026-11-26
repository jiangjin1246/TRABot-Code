#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_file "${G1_MUJOCO_PYTHON}"
require_file "${MUJOCO_MODEL_PATH}"
require_dir "${PROJECT_ROOT}/motions"

cd "${PROJECT_ROOT}"
exec env \
  DISPLAY="${SIM_DISPLAY}" \
  MUJOCO_GL="${SIM_MUJOCO_GL}" \
  CUDA_VISIBLE_DEVICES="${SIM_CUDA_VISIBLE_DEVICES}" \
  "${G1_MUJOCO_PYTHON}" -m sim_service \
  --host "${SIM_HOST}" \
  --port "${SIM_PORT}" \
  --engine mujoco \
  --model "${MUJOCO_MODEL_PATH}" \
  --motion-root "${PROJECT_ROOT}/motions" \
  --width 640 \
  --height 480 \
  --render-fps "${SIM_RENDER_FPS}" \
  --queue-capacity 2
