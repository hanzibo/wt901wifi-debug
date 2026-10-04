#!/usr/bin/env bash
# Experimental dual-backend GUI; original launch_rgbd.sh remains unchanged.
set -u
yolo_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
yolo_conda_executable="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
yolo_log_dir="${XDG_STATE_HOME:-$HOME/.local/state}/wt901"
mkdir -p -- "$yolo_log_dir"
yolo_log_file="$yolo_log_dir/rgbd-yolo.log"
cd -- "$yolo_project_dir/.." || exit 1
"$yolo_conda_executable" run --no-capture-output -n OptUSBProj-yolo \
    python -u "$yolo_project_dir/fusion_run.py" --camera-source orbbec \
    --rgbd-fps 10 --rgbd-warmup 2 --rgbd-pose --pose-backend yolo26 "$@" \
    > "$yolo_log_file" 2>&1
yolo_status=$?
if (( yolo_status != 0 )) && command -v zenity >/dev/null 2>&1; then
    zenity --error --title="WT901 RGB-D YOLO启动失败" --width=560 \
        --text="请检查OptUSBProj-yolo环境及本地权重，并关闭占用相机或UDP1399的程序。
日志：$yolo_log_file" || true
fi
exit "$yolo_status"
