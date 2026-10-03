#!/usr/bin/env bash
# Separate RGB-D entry; leave the original pure-IMU launcher unchanged.
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

if [[ -z "$rgbd_conda_executable" || ! -x "$rgbd_conda_executable" ]]; then
    printf 'Conda executable not found.\n' > "$rgbd_log_file"
    rgbd_status=127
else
    cd -- "$rgbd_repo_dir" || exit 1
    "$rgbd_conda_executable" run --no-capture-output -n OptUSBProj-fusion \
        python -u "$rgbd_project_dir/fusion_run.py" \
        --camera-source orbbec --rgbd-fps 10 --rgbd-warmup 2 "$@" \
        > "$rgbd_log_file" 2>&1
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
