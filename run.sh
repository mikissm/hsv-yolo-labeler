#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"
INSTALL_STAMP="$VENV_DIR/.requirements-installed"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
fi

if [[ ! -f "$INSTALL_STAMP" || "$REQUIREMENTS" -nt "$INSTALL_STAMP" ]]; then
  "$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS"
  touch "$INSTALL_STAMP"
fi

exec "$VENV_DIR/bin/python" hsv_yolo_labeler.py "$@"
