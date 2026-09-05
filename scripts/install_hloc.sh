#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HLOC_DIR="${PROJECT_DIR}/third_party/Hierarchical-Localization"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to the Python used by your Jupyter kernel." >&2
    exit 1
fi

if [[ ! -d "${HLOC_DIR}/.git" ]]; then
    mkdir -p "${PROJECT_DIR}/third_party"
    git clone --recursive https://github.com/cvg/Hierarchical-Localization.git "${HLOC_DIR}"
else
    git -C "${HLOC_DIR}" submodule update --init --recursive
fi

"${PYTHON_BIN}" -m pip install -r "${PROJECT_DIR}/requirements-localization.txt"
"${PYTHON_BIN}" -m pip install -e "${HLOC_DIR}"

echo "HLoc installed into: ${PYTHON_BIN}"
echo "HLoc source: ${HLOC_DIR}"
