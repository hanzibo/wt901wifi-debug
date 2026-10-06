#!/usr/bin/env bash
# Launch the full-body GUI from its own Conda environment.
set -u

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$project_dir")"
conda_executable="${CONDA_EXE:-/home/hzb/miniconda3/bin/conda}"
if [[ ! -x "$conda_executable" ]]; then
    conda_executable="$(command -v conda || true)"
fi

log_dir="${XDG_STATE_HOME:-$HOME/.local/state}/wt901"
mkdir -p -- "$log_dir"
log_file="$log_dir/optusbproj-ubuntu.log"

if [[ -z "$conda_executable" || ! -x "$conda_executable" ]]; then
    printf 'Conda executable not found.\n' > "$log_file"
    status=127
else
    cd -- "$repo_dir" || exit 1
    "$conda_executable" run --no-capture-output -n OptUSBProj \
        python -u "$project_dir/run.py" > "$log_file" 2>&1
    status=$?
fi

if (( status != 0 )); then
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title="WT901 全身姿态映射启动失败" \
            --text="程序启动失败。请确认 OptUSBProj 环境存在，并关闭可能占用 UDP 1399 的手腕控制台。
详细日志：$log_file" \
            --width=540 || true
    fi
fi
exit "$status"
