#!/usr/bin/env bash
# Clone the Unitree G1 MuJoCo model and SDK into external/ at the commits used in the paper.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/external"

fetch() {
  local name="$1" url="$2" commit="$3" dir="${ROOT}/external/$1"
  if [[ ! -d "${dir}/.git" ]]; then
    git clone "${url}" "${dir}"
  fi
  git -C "${dir}" fetch --quiet origin
  git -C "${dir}" checkout --quiet "${commit}"
  echo "${name}: $(git -C "${dir}" rev-parse --short HEAD)"
}

fetch unitree_mujoco https://github.com/unitreerobotics/unitree_mujoco.git c598f103acb87a5fd3de7c9037f4dab6aa7f232b
fetch unitree_sdk2   https://github.com/unitreerobotics/unitree_sdk2.git   5ea10f3230e41c6ab62aa79f49f84afa685d84aa

test -f "${ROOT}/external/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
echo "G1 MuJoCo scene: external/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
