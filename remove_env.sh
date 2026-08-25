#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

VENV_DIR=".venv"
PYTHON_CACHE="__pycache__"
QT_PLUGIN_LINK="${TMPDIR:-/tmp}/hsv_yolo_labeler_qt_plugins_${UID}"

if [[ "${1:-}" != "--yes" ]]; then
  printf '%s\n' "다음 실행 환경만 제거합니다:"
  printf '  - %s\n' "$PWD/$VENV_DIR"
  printf '  - %s\n' "$PWD/$PYTHON_CACHE"
  printf '  - %s\n' "$QT_PLUGIN_LINK"
  printf '%s' "데이터셋과 소스 파일은 유지됩니다. 계속할까요? [y/N] "
  read -r answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) printf '%s\n' "취소했습니다."; exit 0 ;;
  esac
fi

if [[ -d "$VENV_DIR" ]]; then
  rm -rf -- "$VENV_DIR"
fi
if [[ -d "$PYTHON_CACHE" ]]; then
  rm -rf -- "$PYTHON_CACHE"
fi
if [[ -L "$QT_PLUGIN_LINK" ]]; then
  rm -- "$QT_PLUGIN_LINK"
fi

printf '%s\n' "실행 환경을 제거했습니다. 다시 설치하려면 ./run.sh를 실행하세요."
