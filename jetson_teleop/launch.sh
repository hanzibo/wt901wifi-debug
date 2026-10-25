#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
# Optional Ubuntu runtime package extracted without sudo on the Jetson.
task_qt_libs="$task_root/jetson_teleop/.runtime/usr/lib/aarch64-linux-gnu"
if [[ -d "$task_qt_libs" ]]; then
    export LD_LIBRARY_PATH="$task_qt_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$task_root/.venv/bin/python" "$task_root/jetson_teleop/run.py" "$@"
