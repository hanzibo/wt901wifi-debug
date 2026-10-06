# WT901 六轴长期测试与 VQF 对照

这是一套只读、单设备实验。录制时独占 UDP 1399；请先关闭手腕/人体 GUI。不会向设备发送归零或参数写入命令。`long_run.py` 只用当前 `wt901` 环境；离线对照需要额外安装官方 [`vqf`](https://github.com/dlaidig/vqf) 包，不需要克隆仓库：

```bash
conda activate wt901
python -m pip install vqf
```

记录 10 秒起始中立位、30 分钟持续运动、10 秒结束中立位。启动前另有 8 秒设备发现和 10 秒准备；自动选择在线设备中**电压最高**的一台并列出所有候选。要指定设备可加 `--device-id 11590`。`约 xx%` 是电压表估算，**不是续航保证**，测试前充电更可靠。

```bash
cd /home/hzb/Work/WT901
conda run --no-capture-output -n wt901 python -u experiments/imu_vqf/long_run.py
```

听到/看到“开始录制”后保持中立 10 秒；提示“开始运动”后自由完成平举、侧举、转腕等动作。提示“还剩 60 秒运动”时准备返回，提示“回到原中立位”后尽量保持**完全相同的姿势和朝向** 10 秒。无需手动点击结束；`Ctrl+C` 可提前结束并保存部分数据。终端**每秒更新**已用时间、总时长、当前阶段剩余时间、包数、电压和在线/断流状态；交互终端在原位刷新一行，重定向到文件时逐行记录。录制文件保存到 `OptUSBProj-ubuntu/recordings/imu_long_run/<日期-编号>/`。

先用两分钟试跑可以把命令末尾改为 `--minutes 2`；对照分析支持短录制，但终判仍建议完整 30 分钟。注意单靠加速度无法证明身体在结束时恢复了相同**航向**。若要严格评估航向漂移，最好把传感器固定在可重复的夹具/桌面标记上，并在起止都让它回到同一位置与朝向。

录完后离线分析；其中 `<目录>` 替换为程序打印的实际目录：

```bash
conda run --no-capture-output -n wt901 python -u experiments/imu_vqf/compare.py '<目录>'
```

对比程序也接受此前 `imu_axis_experiment.py` 生成的完整单设备录制，以及 `imu_relative_experiment.py` 的双设备录制。双设备记录默认分析手背；加 `--device-id 11381` 可单独分析前臂，结果会写入不同子目录。短时结果不能推断 30 分钟航向漂移。

分析输出位于 `<目录>/comparison/`：`summary.json` 包含起止 SO(3) 回零残差、倾斜、航向代理、VQF/现有算法偏置估计、数据间隔与电压；`comparison_1hz.csv` 是逐秒时序；`comparison.svg` 可直接打开查看运动全过程。两种算法均使用**同一份角速度和加速度**，不使用厂商 R/P/Y 或磁力计；优先按单调可信的设备片内时间插值成统一频率，片内时间不可用时才用电脑收包时间。超过 200 ms 的设备数据间隔会记录警告，超过 1 秒则标记本次回零比较不可靠。回零残差不是有外部真值支持的“绝对精度”，也包含设备未准确回到原始姿势的误差。航向代理仅在接近中立位时解释。

VQF 的 Python API、单位和六轴调用方式见[官方文档](https://vqf.readthedocs.io/en/latest/ref_python.html)：角速度 rad/s，加速度 m/s²，`getQuat6D()` 获取无磁力计姿态。
