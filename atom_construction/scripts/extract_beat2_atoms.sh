#!/usr/bin/env bash
# Extract Atom Schema v1 candidates from retargeted (G1 29-DoF) BEAT2 pickles.
# Usage: BEAT2_G1_PKL_DIR=/path/to/pkl-results bash scripts/extract_beat2_atoms.sh [extract flags]
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-$(conda info --base 2>/dev/null || dirname "$(dirname "${CONDA_EXE:-${HOME}/miniconda3/bin/conda}")")/envs}"
G1_MUJOCO_PYTHON="${G1_MUJOCO_PYTHON:-${CONDA_ENVS_DIR}/g1-mujoco/bin/python}"
MUJOCO_MODEL_PATH="${MUJOCO_MODEL_PATH:-${REPO_ROOT}/external/unitree_mujoco/unitree_robots/g1/scene_29dof.xml}"
: "${BEAT2_G1_PKL_DIR:?set BEAT2_G1_PKL_DIR to your retargeted BEAT2 pkl directory}"

for f in "${G1_MUJOCO_PYTHON}" "${MUJOCO_MODEL_PATH}"; do
  [[ -e "$f" ]] || { echo "Missing: $f (see README, Setup)" >&2; exit 1; }
done
[[ -d "${BEAT2_G1_PKL_DIR}" ]] || { echo "Missing directory: ${BEAT2_G1_PKL_DIR}" >&2; exit 1; }

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ATOM_OUTPUT_DIR="${ATOM_OUTPUT_DIR:-${PROJECT_ROOT}/runs/run_${RUN_ID}}"

cd "${PROJECT_ROOT}"
# Extraction defaults to full retarget amplitude (gain=1.0).
exec env MUJOCO_GL="${MUJOCO_GL:-egl}" \
  "${G1_MUJOCO_PYTHON}" -m atom_pipeline extract \
  --source "${BEAT2_G1_PKL_DIR}" \
  --out-dir "${ATOM_OUTPUT_DIR}" \
  --neutral "${REPO_ROOT}/catalog/safe_neutral.json" \
  --scene "${MUJOCO_MODEL_PATH}" \
  "$@"
