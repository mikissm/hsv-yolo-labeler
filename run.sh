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

QT_PLUGIN_SOURCES=("$VENV_DIR"/lib/python*/site-packages/PyQt5/Qt5/plugins)
QT_PLUGIN_LINK="${TMPDIR:-/tmp}/hsv_yolo_labeler_qt_plugins_${UID}"
if [[ -d "${QT_PLUGIN_SOURCES[0]}" ]]; then
  ln -sfn "$(cd "${QT_PLUGIN_SOURCES[0]}" && pwd)" "$QT_PLUGIN_LINK"
  export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_LINK"
fi
unset QT_QPA_FONTDIR
exec "$VENV_DIR/bin/python" hsv_yolo_labeler.py "$@"
