#!/usr/bin/env bash
# Desktop launcher: run the GUI in its Conda environment without a shell setup.
set -euo pipefail
cd /home/hzb/Work/WT901
if /home/hzb/miniconda3/bin/conda run --no-capture-output -n wt901 \
    python -u /home/hzb/Work/WT901/wrist_demo.py; then
    exit 0
else
    status=$?
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title="WT901 启动失败" \
            --text=$'程序启动失败。若已有 WT901 窗口，请先关闭旧窗口，避免 UDP 1399 端口冲突。\n也可在终端运行 launch_wt901.sh 查看详细错误。' \
            --width=440 || true
    fi
    exit "$status"
fi
