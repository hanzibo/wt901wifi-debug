#!/usr/bin/env bash
# Prefer the complete dual-backend environment; keep pure-IMU entry unchanged.
set -u

rgbd_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rgbd_repo_dir="$(dirname -- "$rgbd_project_dir")"
rgbd_conda_executable="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
if [[ ! -x "$rgbd_conda_executable" ]]; then
    rgbd_conda_executable="$(command -v conda || true)"
fi

rgbd_log_dir="${XDG_STATE_HOME:-$HOME/.local/state}/wt901"
mkdir -p -- "$rgbd_log_dir"
rgbd_log_file="$rgbd_log_dir/rgbd-capture.log"
rgbd_environment="OptUSBProj-fusion"
rgbd_yolo_python="$HOME/miniconda3/envs/OptUSBProj-yolo/bin/python"
if [[ -x "$rgbd_yolo_python" ]] && "$rgbd_yolo_python" -c \
    'import importlib.util; import sys; sys.exit(not all(importlib.util.find_spec(n) is not None for n in ("torch", "ultralytics", "rtmlib", "onnxruntime", "PySide6")))'; then
    rgbd_environment="OptUSBProj-yolo"
fi

if [[ -z "$rgbd_conda_executable" || ! -x "$rgbd_conda_executable" ]]; then
    printf 'Conda executable not found.\n' > "$rgbd_log_file"
    rgbd_status=127
else
    cd -- "$rgbd_repo_dir" || exit 1
    printf 'RGB-D GUI environment: %s\n' "$rgbd_environment" > "$rgbd_log_file"
    "$rgbd_conda_executable" run --no-capture-output -n "$rgbd_environment" \
        python -u "$rgbd_project_dir/fusion_run.py" \
        --camera-source orbbec --rgbd-fps 10 --rgbd-warmup 2 "$@" \
        >> "$rgbd_log_file" 2>&1
    rgbd_status=$?
fi

if (( rgbd_status != 0 )); then
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title="WT901 RGB-D 联合采集启动失败" \
            --text="请确认 OptUSBProj-fusion 和 orbbec336l 环境存在、Gemini 已连接，并关闭 OrbbecViewer 及其他占用 UDP 1399 的程序。
详细日志：$rgbd_log_file" --width=560 || true
    fi
fi
exit "$rgbd_status"
