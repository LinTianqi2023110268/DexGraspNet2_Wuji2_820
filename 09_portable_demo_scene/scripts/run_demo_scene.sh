#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
HEADLESS_FLAG="${1:---headless}"
if [[ "${HEADLESS_FLAG}" == "--gui" ]]; then
  HEADLESS_FLAG=""
fi
python 09_portable_demo_scene/scripts/validate_portable_scene.py
python 09_portable_demo_scene/scripts/print_scene_contents.py
if [[ -n "${HEADLESS_FLAG}" ]]; then
  python 09_portable_demo_scene/scripts/load_scene_only.py --headless
else
  python 09_portable_demo_scene/scripts/load_scene_only.py
fi
