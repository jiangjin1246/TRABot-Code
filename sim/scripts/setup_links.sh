#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_dir "${FLASHHEAD_SOURCE_DIR}"
require_dir "${G1_SOURCE_MOTION_ROOT}"

for name in models examples flash_head; do
  source_path="${FLASHHEAD_SOURCE_DIR}/${name}"
  target_path="${PROJECT_ROOT}/${name}"
  require_dir "${source_path}"
  if [[ -e "${target_path}" || -L "${target_path}" ]]; then
    echo "Keeping existing ${target_path}"
  else
    ln -s "${source_path}" "${target_path}"
    echo "Linked ${target_path} -> ${source_path}"
  fi
done

mkdir -p "${PROJECT_ROOT}/motions"

# Give the simulator-facing actions exactly the IDs used by the existing
# Gemini bridge. Only these known JSON files are linked; no real-robot binary
# or network configuration enters this project.
link_motion() {
  local source_name="$1"
  local action_id="$2"
  local source_path="${G1_SOURCE_MOTION_ROOT}/${source_name}.json"
  local target_path="${PROJECT_ROOT}/motions/${action_id}.json"
  require_file "${source_path}"
  if [[ -e "${target_path}" || -L "${target_path}" ]]; then
    echo "Keeping existing ${target_path}"
  else
    ln -s "${source_path}" "${target_path}"
    echo "Linked ${target_path} -> ${source_path}"
  fi
}

link_motion presentation_gesture joint_presentation_gesture
link_motion presentation_gesture_b joint_presentation_gesture_b
link_motion presentation_gesture_c joint_presentation_gesture_c

mkdir -p "${PROJECT_ROOT}/motions"
neutral_target="${PROJECT_ROOT}/motions/safe_neutral.json"
if [[ ! -e "${neutral_target}" && ! -L "${neutral_target}" ]]; then
  ln -s "${REPO_ROOT}/catalog/safe_neutral.json" "${neutral_target}"
  echo "Linked ${neutral_target} -> ${REPO_ROOT}/catalog/safe_neutral.json"
fi

release_target="${PROJECT_ROOT}/motions/joint_release_arm_pose.json"
if [[ ! -e "${release_target}" && ! -L "${release_target}" ]]; then
  ln -s "${PROJECT_ROOT}/motions/safe_neutral.json" "${release_target}"
  echo "Linked ${release_target} -> ${PROJECT_ROOT}/motions/safe_neutral.json"
fi

echo "Asset links are ready."
