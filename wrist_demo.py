#!/usr/bin/env python3
"""Two-IMU wrist rotation trial with fixed forearms and moving hands.

Hold one sensor rigidly in each hand and keep the forearms still. Each hand
uses its own sensor's starting pose as neutral. Space resets both hands;
1 and 2 reset the left and right hands separately.

By default, receive UDP data on all local interfaces at port 1399, so the GUI
can stay open while Wi-Fi switches between the computer hotspot and device AP.

The right-side tabs show live sensor data and readable register parameters.
They also provide guided magnetic and accelerometer calibration, plus the
device's heading/angle reference commands and per-device algorithm, bandwidth,
and output-rate settings. Only one UDP listener is used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
import sys
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from wt901_viz.orientation import rotation_matrix
from wt901_viz.axis_mapping import DEFAULT_AXIS_MAPPING, mapping_matrix, normalized_mapping
from wt901_viz.yaw_guard import GyroOnly, PacketClock, YawGuard
from wt901_viz.so3_filter import AdaptiveTiltFilter, SO3Health, TiltEstimate, check_so3
from wt901.registers import (
    REG_ACCRANGE, REG_AXIS6, REG_BANDWIDTH, REG_CALSW, REG_GYRORANGE,
    REG_ORIENT, REG_RATE, REG_SAVE, UNLOCK_SETTLE,
)


def relative_rotation(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Current hand orientation expressed in the neutral hand frame."""
    return reference.T @ current


def model_rotation(reference: np.ndarray, current: np.ndarray,
                   mapping: tuple[str, str, str]) -> np.ndarray:
    """Express a neutral-relative sensor rotation in the hand model's axes."""
    basis = mapping_matrix(mapping)
    return basis @ relative_rotation(reference, current) @ basis.T


def xyz_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler decomposition, returned as X, Y, Z angles in degrees."""
    y = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    x = np.arctan2(matrix[2, 1], matrix[2, 2])
    z = np.arctan2(matrix[1, 0], matrix[0, 0])
    return tuple(np.degrees((x, y, z)))


def box_vertices(x0, x1, y0, y1, z0, z1):
    return np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)


def wrist_axis_points(axis: int) -> np.ndarray:
    """Arrow line segments rooted at the fixed wrist rotation pivot."""
    direction = np.eye(3, dtype=np.float32)[axis]
    side_direction = np.eye(3, dtype=np.float32)[(axis + 1) % 3]
    tip = direction * 0.42
    head = direction * 0.32
    return np.array([
        np.zeros(3), tip,
        tip, head + side_direction * 0.045,
        tip, head - side_direction * 0.045,
    ], dtype=np.float32)


BOX_FACES = np.array([
    [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
    [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
    [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
], dtype=np.uint32)

MAG_PRECHECK_MIN_SECONDS = 20.0
MAG_PRECHECK_MAX_SECONDS = 90.0
MAG_MIN_AXIS_SPAN_UT = 30.0
MAG_MAX_AXIS_SPAN_UT = 50.0

HELP_PAGES = (
    ("快速开始", """
<h2>WT901 双手手腕控制台</h2>
<p>左手默认对应 11381，右手默认对应 11408。中央模型显示每只手相对各自中立位的转动；
前臂模型固定，不能测量手臂在空间中的位置。</p>
<p>右侧切换设备标签页，先点击“读取设备参数”，再调整该设备的算法、带宽、
回传频率或量程。每项设置都会发送、保存并回读确认；离线设备无法配置。</p>
<p>新设备初次处于 AP 模式时，点击顶部“接入新设备”，按照弹窗完成接入。</p>
<p>点击顶部“热点设备”可查看当前关联到电脑热点的终端，以及已经通过 UDP 数据确认的 WT 设备 ID 和端点。</p>
<p>点击顶部“设备分配”可把当前热点中在线的 WT 设备绑定到左手或右手；换绑后该部位会以新设备的下一包数据重建中立位。</p>
<p>每只手旁的“轴映射”可设定设备 X/Y/Z 正轴在手掌模型中立位时指向的模型轴；
红/绿/蓝箭头分别表示随手掌转动的模型 X/Y/Z 轴。映射仅作用于当前 GUI，
不写入设备，重启程序后恢复默认映射。</p>
<p><b>快捷键：</b>空格同时归零；1 只归零左手；2 只归零右手；
R 重置视角；Esc/Q 关闭窗口。</p>
"""),
    ("接入新设备", """
<h2>从设备 AP 切换到电脑热点</h2>
<p>1. 点击顶部“接入新设备”。如果电脑正在开 WT901-LAN 热点，先关闭它；
已连接的旧设备会暂时离线。</p>
<p>2. 扫描并连接名称为 WT55… 的新设备热点，也可用 Ubuntu 的 Wi-Fi 菜单连接，
再点“刷新状态”。弹窗会核对当前连接名称。</p>
<p>3. 输入<strong>电脑热点</strong>的密码，确认 SSID 与已保存的电脑热点一致，
点击“写入 STA + UDP 配置”。这会向当前设备发送合并指令，设置网络及电脑接收地址。
密码不会显示在日志中。</p>
<p>4. 设备热点可能随即消失。点击“启动电脑热点”，等待弹窗检测到该设备 ID 的实时数据。
如果未接入，核对密码、设备电量和电脑热点的 2.4 GHz 设置。</p>
<p>一次只迁移一台。设置成功后，其余寄存器可在主界面的设备参数面板修改；
若新设备要代替左右手模型，可在重启程序时用 --left-id 或 --right-id 指定。</p>
"""),
    ("热点设备", """
<h2>查看当前热点终端</h2>
<p>顶部“热点设备”会打开实时列表，仅在电脑 Wi-Fi 处于热点 AP 模式时显示已关联终端。</p>
<p>“设备唯一 ID”来自实际收到的 WT 数据包；仅有 Wi-Fi 关联、但尚未发数据的终端会显示“未识别”，不能据此断定它是 WT 设备。</p>
<p>“设备端点”是传感器发包的 IP 和源端口；“电脑接收端点”是热点地址及本程序的 UDP 监听端口。列表约每 3 秒重新检查热点关联情况，在线状态随数据更新。</p>
<p>选中一个“WT 数据在线”的设备后，可点“将选中设备切回 AP/UDP”。手册指令为 WITAPUDP；设备会离开电脑热点并建立以其 ID 命名的 Wi-Fi，方便另一台电脑直连调试。发送后需在另一台电脑上验证，不能只凭列表消失判定成功。</p>
"""),
    ("设备分配", """
<h2>给身体部位分配设备</h2>
<p>顶部“设备分配”按身体部位逐行列出当前绑定。候选设备仅包含当前连接电脑热点、且最近收到有效 WT 数据的设备 ID。</p>
<p>可以同时交换两个部位的设备；同一设备不能同时分给两个部位。换绑时不能进行校准、寄存器设置或读取。</p>
<p>应用后模型会等待新设备的下一包数据，以该姿态作为该部位的新中立位；旧设备的参数显示会清空。当前分配仅在本次程序运行中生效。</p>
"""),
    ("带宽与回传", """
<h2>带宽与回传速率</h2>
<p><b>带宽</b>是内部滤波对运动变化的响应范围，不是 Wi-Fi 带宽。
低带宽使数据更平滑、噪声更少，但快速挥动会显得迟缓；
高带宽更跟手，但静止时可能更抖。可选 5、10、20、42、98、188、256 Hz，
手册默认 20 Hz。</p>
<p><b>回传速率</b>是每秒向电脑发送多少组数据。50 Hz 约每 20 ms 一包，
100 Hz 约每 10 ms 一包。调高会增加网络和程序负载，不能替代提高内部带宽；
带宽较低时，相邻数据可能十分接近。最高 200 Hz。</p>
<p>手腕演示可先试带宽 20～42 Hz、回传 50～100 Hz，再按延迟与抖动调整。
回传周期以整数毫秒保存，实际频率可能与输入值略有差异。</p>
"""),
    ("陀螺与加计量程", """
<h2>陀螺仪与加速度计量程</h2>
<p><b>陀螺量程</b>是三轴角速度可测的范围：±250、±500、±1000、±2000 °/s。
挥手过快而超过量程时，读数会截顶，动作估计可能失真。低档更适合较慢动作，
高档给快速旋转留更多余量；不确定时先保持当前值。</p>
<p><b>加计量程</b>是三轴加速度可测的范围：±2、±4、±8、±16 g。
静止时一轴接近 +1 g 是正常的支持力读数；快速甩动或碰撞可能超出小量程。
若读数经常接近边界，应提高一档。</p>
<p>量程改变的是设备内部可测范围；本设备的 54 字节网络报文仍按协议固定公式
换算成 g 和 °/s，GUI 的单位不随量程变化。每台设备的量程独立设置。</p>
"""),
    ("电池电量", """
<h2>电压与估算电量</h2>
<p>设备报文给出电池电压（单位 V），不直接给出精确的剩余容量。
GUI 按 WT901WIFI 协议手册的分档表显示“约 xx%”，不进行线性插值：</p>
<p>&gt;3.96 V：100%；3.93～3.96 V：90%；3.87～3.92 V：75%；
3.82～3.86 V：60%；3.79～3.81 V：50%；3.77～3.78 V：40%；
3.73～3.76 V：30%；3.70～3.72 V：20%；3.68～3.69 V：15%；
3.50～3.67 V：10%；3.40～3.49 V：5%；&lt;3.40 V：0%。</p>
<p>手册指出电压不高于 3.3 V 时电池已耗尽。电压会受充电、瞬时负载和温度影响，
因此百分比只是粗略档位；充电时尤其不能当作准确续航时间。</p>
"""),
    ("6轴与9轴", """
<h2>融合算法</h2>
<p><b>6 轴</b>使用加速度计和陀螺仪，航向角是相对值；短时间动作通常不受附近磁铁影响，
但长时间可能漂移。</p>
<p><b>9 轴</b>还使用磁力计修正航向，可约束长期漂移；磁场干扰或校准不良可能使航向偏移。
切换算法后，该手的 GUI 中立位会根据下一包数据重新建立。</p>
"""),
    ("俯仰角保护", """
<h2>俯仰角保护区（实验功能）</h2>
<p>每只手可独立选择显示模式。默认当 |Pitch| 达到 80° 时进入保护区，下降到 75° 时退出；
两个角度可在对应设备页自行调整，退出阈值必须低于进入阈值，以免边界反复切换。</p>
<p>对应设备页可选择四种姿态来源：厂商 RPY、俯仰保护、全程陀螺、陀螺＋加计纠倾斜。
四条链路始终同时计算；切换显示模式不会重新归零，因此可在同一次动作中直接对比。</p>
<p>保护区内不直接采用设备欧拉角 Yaw，而是以进入前的姿态为起点，用<b>三轴角速度</b>
积分完整的旋转矩阵。退出时保留航向修正，尽量避免马上跳回偏移的 Yaw。
全程陀螺从设备首包开始取单位姿态，仅用角速度和设备时间戳，不读取 RPY。
第四模式在静止约 1 秒后学习三轴陀螺零偏，并在加速度可信时缓慢修正倾斜；
它不会使用加速度修正长期航向。建议启动或换绑后先保持设备静止。
修改保护阈值会重新建立该手四条链路的中立位；不会写设备寄存器。</p>
<p>接近 90° 时，ZYX 欧拉角 Yaw 本身并不稳定。保护区内界面展示的“竖轴累计角”
是角速度在世界竖轴的积分，并不等同于严格的欧拉 Yaw；模型使用完整旋转矩阵。
长时间在保护区内会积累陀螺漂移，进入保护区前已有的错误也无法追回。
全程陀螺更可能随时间漂移，若有漏包或长时间离线，请重新开始测试。</p>
<p>界面显示的 SO(3) 距离是当前矩阵到最近正规旋转矩阵的 Frobenius 距离。
数值超出 1e-8 时才以 SVD 投影；这只修正数值失正交，不会修正传感器测量偏差。</p>
"""),
    ("安装方向", """
<h2>设备安装方向</h2>
<p>安装方向是设备内部姿态解算的设置，不是把 GUI 手掌模型旋转 90°。
默认“水平安装”（寄存器 ORIENT=0）；只有模块固定为垂直安装时才选“垂直安装”（ORIENT=1）。</p>
<p>手册指定的垂直姿态是：从水平姿态绕设备 X 轴转 90°，最终让设备标注的 <b>Y 轴箭头朝上</b>。
这不是任意侧面朝上的通用模式；手持设备自由转动时通常保持与实际固定安装方式相符的设置。</p>
<p>改动会写入并保存到设备，回读确认后 GUI 用下一包数据重建该部位中立位。
操作手册要求改动后重新校准；本程序不会自动开始校准。手册的通用加计校准说明要求模块本体水平静止，
但未单独说明垂直模式下的校准摆放。垂直安装后请先确认厂家校准步骤，勿直接套用水平校准入口。</p>
"""),
    ("校准与归零", """
<h2>校准与归零</h2>
<p><b>磁场校准：</b>远离磁吸物和大块金属。先在预检阶段充分翻转设备，
通过后继续绕三个轴转动，最后点击“结束磁场校准并保存”。</p>
<p><b>加速度计校准：</b>设备正面朝上、水平静止，直到自动结束。</p>
<p><b>Z 轴航向归零：</b>仅在 6 轴算法下可用；
<b>X/Y 角度参考归零：</b>改变设备输出的参考角。
它们与空格键的 GUI 中立位归零不同：前者写入设备，后者只影响当前界面。</p>
"""),
)


def mag_coverage(state: HandState) -> tuple[list[float], bool]:
    span = [hi - lo for lo, hi in zip(state.mag_min, state.mag_max)]
    passed = (min(span) >= MAG_MIN_AXIS_SPAN_UT
              and max(span) >= MAG_MAX_AXIS_SPAN_UT)
    return span, passed


@dataclass(frozen=True)
class RegisterSetting:
    address: int
    value: int
    label: str


@dataclass(frozen=True)
class BodyPartSpec:
    key: str
    name: str
    y_offset: float
    mirror: bool
    shortcut: str


BODY_PARTS = (
    # Add a spec for each future body-part slot; the assignment dialog is generated from it.
    BodyPartSpec("left_hand", "左手", 0.8, False, "1"),
    BodyPartSpec("right_hand", "右手", -0.8, True, "2"),
)


def qt_matrix(r: np.ndarray, y_offset: float = 0.0) -> QtGui.QMatrix4x4:
    return QtGui.QMatrix4x4(
        float(r[0, 0]), float(r[0, 1]), float(r[0, 2]), 0.0,
        float(r[1, 0]), float(r[1, 1]), float(r[1, 2]), y_offset,
        float(r[2, 0]), float(r[2, 1]), float(r[2, 2]), 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


@dataclass
class HandState:
    part_key: str
    name: str
    device_id: str
    y_offset: float
    info: QtWidgets.QLabel
    details: QtWidgets.QLabel | None = None
    parameters: QtWidgets.QLabel | None = None
    status: QtWidgets.QLabel | None = None
    algorithm_combo: QtWidgets.QComboBox | None = None
    orient_combo: QtWidgets.QComboBox | None = None
    orient_apply: QtWidgets.QPushButton | None = None
    bandwidth_combo: QtWidgets.QComboBox | None = None
    bandwidth_apply: QtWidgets.QPushButton | None = None
    rate_spin: QtWidgets.QDoubleSpinBox | None = None
    rate_apply: QtWidgets.QPushButton | None = None
    gyro_combo: QtWidgets.QComboBox | None = None
    gyro_apply: QtWidgets.QPushButton | None = None
    acc_combo: QtWidgets.QComboBox | None = None
    acc_apply: QtWidgets.QPushButton | None = None
    parts: list[gl.GLMeshItem] = field(default_factory=list)
    axes: list[gl.GLLinePlotItem] = field(default_factory=list)
    axis_mapping: tuple[str, str, str] = DEFAULT_AXIS_MAPPING
    mapping_button: QtWidgets.QPushButton | None = None
    mode_combo: QtWidgets.QComboBox | None = None
    guard_enter_spin: QtWidgets.QDoubleSpinBox | None = None
    guard_exit_spin: QtWidgets.QDoubleSpinBox | None = None
    guard_status: QtWidgets.QLabel | None = None
    yaw_guard: YawGuard | None = None
    gyro_only: GyroOnly | None = None
    tilt_filter: AdaptiveTiltFilter | None = None
    tilt_estimate: TiltEstimate | None = None
    packet_clock: PacketClock | None = None
    last_device_tick_s: float | None = None
    stream_current: dict[str, np.ndarray] = field(default_factory=dict)
    stream_reference: dict[str, np.ndarray] = field(default_factory=dict)
    stream_health: dict[str, SO3Health] = field(default_factory=dict)
    so3_projection_count: int = 0
    guarded_heading_deg: float | None = None
    guarded_active: bool = False
    guarded_offset_deg: float = 0.0
    reference: np.ndarray | None = None
    current: np.ndarray | None = None
    last_packet_time: float = 0.0
    last_details_time: float = 0.0
    calibration: str | None = None
    calibration_started: float = 0.0
    mag_min: list[float] = field(default_factory=lambda: [float("inf")] * 3)
    mag_max: list[float] = field(default_factory=lambda: [float("-inf")] * 3)
    pending_reads: list[int] = field(default_factory=list)
    waiting_address: int | None = None
    read_deadline: float = 0.0
    registers: dict[int, int] = field(default_factory=dict)
    setting: RegisterSetting | None = None
    controls: list[QtWidgets.QWidget] = field(default_factory=list)
    neutral_button: QtWidgets.QPushButton | None = None


class AxisMappingDialog(QtWidgets.QDialog):
    """Choose a right-handed device-axis to hand-model-axis mapping."""

    CHOICES = tuple((f"{sign}{axis}", f"模型 {sign}{axis}")
                    for axis in "XYZ" for sign in "+-")

    def __init__(self, state: HandState, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setWindowTitle(f"{state.name} · 设备轴映射")
        self.setMinimumWidth(390)
        layout = QtWidgets.QVBoxLayout(self)
        instruction = QtWidgets.QLabel(
            "先让手掌与模型中立姿势一致并归零，再选择设备每根正轴"
            "在模型坐标系中指向哪里。模型轴可在主画面中查看。")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        form = QtWidgets.QFormLayout()
        self.selectors = []
        for axis, initial in zip("XYZ", state.axis_mapping):
            combo = QtWidgets.QComboBox()
            for value, label in self.CHOICES:
                combo.addItem(label, value)
            combo.setCurrentIndex(combo.findData(initial))
            form.addRow(f"设备 +{axis} 轴 →", combo)
            self.selectors.append(combo)
        layout.addLayout(form)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.ok_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("保存映射")
        for combo in self.selectors:
            combo.currentIndexChanged.connect(self._validate)
        self._validate()

    def mapping(self) -> tuple[str, str, str]:
        return tuple(combo.currentData() for combo in self.selectors)

    def _validate(self, _index: int = 0) -> None:
        try:
            mapping_matrix(self.mapping())
        except ValueError as exc:
            self.message.setText(str(exc))
            self.ok_button.setEnabled(False)
        else:
            self.message.setText("映射有效：模型轴将按此关系随设备转动。")
            self.ok_button.setEnabled(True)


class WristWindow(QtWidgets.QMainWindow):
    def __init__(self, source, left_id: str, right_id: str, endpoint: str):
        super().__init__()
        self.source = source
        self.started_at = time.monotonic()
        self._last_rate_refresh = float("-inf")
        self.setWindowTitle("WT901 双手转角试验：空格双手归零，1/2 单手归零，R 重置视角")
        self.resize(1150, 740)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((24, 26, 33))
        self.view.setCameraPosition(distance=4.5, elevation=35, azimuth=-65)
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.25, 0.25)
        self.view.addItem(grid)

        def add_box(state, extents, color, moving=False, mirror=False):
            vertices = box_vertices(*extents)
            if mirror:
                vertices[:, 1] *= -1
            if not moving:
                vertices[:, 1] += state.y_offset
            item = gl.GLMeshItem(
                vertexes=vertices, faces=BOX_FACES[:, [0, 2, 1]] if mirror else BOX_FACES,
                smooth=False, color=color, drawEdges=True,
                edgeColor=(0.2, 0.2, 0.2, 0.8), glOptions="opaque",
            )
            self.view.addItem(item)
            if moving:
                state.parts.append(item)
                item.setTransform(qt_matrix(np.eye(3), state.y_offset))

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(10, 6, 10, 2)
        title = QtWidgets.QLabel("WT901 双手手腕控制台")
        title.setStyleSheet("font-size:18px; font-weight:600")
        top_row.addWidget(title)
        top_row.addStretch(1)
        self.axes_checkbox = QtWidgets.QCheckBox("显示模型轴（红 X · 绿 Y · 蓝 Z）")
        self.axes_checkbox.setChecked(True)
        self.axes_checkbox.toggled.connect(self._toggle_axes)
        top_row.addWidget(self.axes_checkbox)
        self.assignment_button = QtWidgets.QPushButton("设备分配")
        self.assignment_button.setToolTip("给各身体部位指定最近收到有效数据的在线 WT 设备")
        self.assignment_button.clicked.connect(self.open_device_assignment)
        top_row.addWidget(self.assignment_button)
        self.devices_button = QtWidgets.QPushButton("在线设备")
        self.devices_button.setToolTip("查看电脑热点、外部 Wi-Fi 或局域网中正在发送数据的 WT 设备")
        self.devices_button.clicked.connect(self.open_connected_devices)
        top_row.addWidget(self.devices_button)
        self.setup_button = QtWidgets.QPushButton("接入新设备")
        self.setup_button.setToolTip("关闭/恢复电脑热点，并将新 WT901 从设备 AP 配成 STA 模式")
        self.setup_button.clicked.connect(self.open_device_setup)
        top_row.addWidget(self.setup_button)
        help_button = QtWidgets.QPushButton("帮助 / Help")
        help_button.setToolTip("查看带宽、回传速率、量程、6/9 轴和校准操作说明")
        help_button.clicked.connect(self.show_help)
        top_row.addWidget(help_button)
        layout.addLayout(top_row)
        self.receive_rate_label = QtWidgets.QLabel("有效收包频率：等待数据")
        self.receive_rate_label.setWordWrap(True)
        self.receive_rate_label.setStyleSheet("padding:6px 10px; background:#edf3f8; color:#152b40")
        self.receive_rate_label.setToolTip(
            "按设备统计近 2 秒本程序实际读到的有效姿态包数 ÷ 2 秒，每 0.25 秒刷新。"
            "开始收包后先采样 2 秒；未分配设备也显示。"
            "不是设备配置频率、画面帧率或无线到达时间；UI 忙碌时批量读包会影响短时统计。"
            "无序号基准，不能据此直接计算丢包率。")
        layout.addWidget(self.receive_rate_label)
        from wt901_viz.receive_diagnostics import ReceiveDiagnosticsPanel
        self.receive_diagnostics = ReceiveDiagnosticsPanel(source, self)
        self._closing_for_receive_diagnostics = False
        self.receive_diagnostics.finished.connect(self._receive_diagnostics_finished)
        layout.addWidget(self.receive_diagnostics)
        self.controls_enabled = hasattr(source, "write_register")
        self.assignment_button.setEnabled(self.controls_enabled)
        self.devices_button.setEnabled(self.controls_enabled)
        self.setup_button.setEnabled(self.controls_enabled)
        self.setup_dialog = None
        self.devices_dialog = None
        self.assignment_dialog = None
        controls = QtWidgets.QTabWidget()
        controls.setMinimumWidth(320)
        controls.setMaximumWidth(400)
        self.hands: dict[str, HandState] = {}
        self.part_states: dict[str, HandState] = {}
        self.shortcuts = []
        initial_ids = {"left_hand": left_id, "right_hand": right_id}
        for spec in BODY_PARTS:
            name, device_id = spec.name, initial_ids.get(spec.key, "")
            y_offset, mirror, key = spec.y_offset, spec.mirror, spec.shortcut
            info = QtWidgets.QLabel(f"{name} {device_id}：等待数据（{endpoint}）")
            info.setStyleSheet("font-size:15px; padding:8px; background:#202028; color:white")
            state = HandState(spec.key, name, device_id, y_offset, info)
            self.part_states[spec.key] = state
            if device_id:
                self.hands[device_id] = state
            header = QtWidgets.QHBoxLayout()
            header.addWidget(info, 1)
            mapping_button = QtWidgets.QPushButton("轴映射")
            state.mapping_button = mapping_button
            mapping_button.clicked.connect(
                lambda _checked=False, part=state: self.show_axis_mapping(part))
            header.addWidget(mapping_button)
            button = QtWidgets.QPushButton(f"{name}归零（{key}）")
            state.neutral_button = button
            button.clicked.connect(lambda checked=False, part=state: self.set_neutral(part.device_id))
            header.addWidget(button)
            layout.addLayout(header)
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(lambda part=state: self.set_neutral(part.device_id))
            self.shortcuts.append(shortcut)

            # Each wrist is a separate origin; the right thumb is mirrored.
            add_box(state, (-1.1, -0.04, -0.15, 0.15, -0.13, 0.13),
                    (0.48, 0.56, 0.67, 1), mirror=mirror)
            add_box(state, (-0.04, 0.06, -0.18, 0.18, -0.15, 0.15),
                    (0.9, 0.75, 0.35, 1), mirror=mirror)
            add_box(state, (0.06, 0.55, -0.20, 0.20, -0.065, 0.065),
                    (0.97, 0.59, 0.35, 1), True, mirror)
            for i in range(4):
                y0 = -0.18 + i * 0.09
                add_box(state, (0.55, 0.80, y0, y0 + 0.065, -0.05, 0.05),
                        (0.97, 0.64, 0.40, 1), True, mirror)
            add_box(state, (0.18, 0.42, 0.20, 0.39, -0.05, 0.05),
                    (0.97, 0.64, 0.40, 1), True, mirror)
            for axis, color in enumerate(((1.0, 0.2, 0.2, 1.0),
                                          (0.2, 1.0, 0.2, 1.0),
                                          (0.25, 0.45, 1.0, 1.0))):
                item = gl.GLLinePlotItem(pos=wrist_axis_points(axis), color=color,
                                         width=3, mode="lines", antialias=True,
                                         glOptions="additive")
                item.setTransform(qt_matrix(np.eye(3), state.y_offset))
                self.view.addItem(item)
                state.axes.append(item)

            panel = QtWidgets.QWidget()
            panel_layout = QtWidgets.QVBoxLayout(panel)
            state.details = QtWidgets.QLabel("等待实时数据")
            state.parameters = QtWidgets.QLabel("寄存器参数：尚未读取")
            state.status = QtWidgets.QLabel("就绪" if self.controls_enabled else "TCP 模式不支持无线配置")
            for label in (state.details, state.parameters, state.status):
                label.setWordWrap(True)
                label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
                panel_layout.addWidget(label)
            guard_group = QtWidgets.QGroupBox("姿态来源 · 四轨迹对比")
            guard_layout = QtWidgets.QVBoxLayout(guard_group)
            state.mode_combo = QtWidgets.QComboBox()
            state.mode_combo.addItem("厂商 RPY（默认）", "raw")
            state.mode_combo.addItem("俯仰角保护区", "guard")
            state.mode_combo.addItem("全程陀螺 SO(3)", "gyro")
            state.mode_combo.addItem("陀螺 + 加计纠倾斜", "fusion")
            state.mode_combo.currentIndexChanged.connect(
                lambda _index, part=state: self._render_selected_pose(part))
            guard_layout.addWidget(state.mode_combo)
            guard_angles = QtWidgets.QHBoxLayout()
            state.guard_enter_spin = QtWidgets.QDoubleSpinBox()
            state.guard_enter_spin.setRange(10.5, 89.5)
            state.guard_enter_spin.setSingleStep(0.5)
            state.guard_enter_spin.setSuffix("° 进入")
            state.guard_enter_spin.setValue(80.0)
            state.guard_exit_spin = QtWidgets.QDoubleSpinBox()
            state.guard_exit_spin.setRange(10.0, 89.0)
            state.guard_exit_spin.setSingleStep(0.5)
            state.guard_exit_spin.setSuffix("° 退出")
            state.guard_exit_spin.setValue(75.0)
            guard_angles.addWidget(state.guard_enter_spin)
            guard_angles.addWidget(state.guard_exit_spin)
            guard_layout.addLayout(guard_angles)
            state.guard_enter_spin.valueChanged.connect(
                lambda value, part=state: self._guard_enter_changed(part, value))
            state.guard_exit_spin.valueChanged.connect(
                lambda value, part=state: self._guard_exit_changed(part, value))
            state.guard_status = QtWidgets.QLabel("未启用；使用设备原始姿态")
            state.guard_status.setWordWrap(True)
            guard_layout.addWidget(state.guard_status)
            panel_layout.addWidget(guard_group)
            read_button = QtWidgets.QPushButton("读取设备参数")
            read_button.clicked.connect(lambda checked=False, part=state: self.read_parameters(part.device_id))
            panel_layout.addWidget(read_button)
            algorithm_row = QtWidgets.QHBoxLayout()
            state.algorithm_combo = QtWidgets.QComboBox()
            state.algorithm_combo.addItem("9 轴：磁场辅助航向", 0)
            state.algorithm_combo.addItem("6 轴：相对航向", 1)
            algorithm_row.addWidget(state.algorithm_combo, 1)
            algorithm_button = QtWidgets.QPushButton("切换")
            algorithm_button.clicked.connect(
                lambda checked=False, part=state: self.switch_algorithm(
                    part.device_id, part.algorithm_combo.currentData()))
            algorithm_row.addWidget(algorithm_button)
            panel_layout.addLayout(algorithm_row)
            orient_row = QtWidgets.QHBoxLayout()
            state.orient_combo = QtWidgets.QComboBox()
            state.orient_combo.addItem("水平安装（默认）", 0)
            state.orient_combo.addItem("垂直安装（Y 轴向上）", 1)
            orient_row.addWidget(state.orient_combo, 1)
            state.orient_apply = QtWidgets.QPushButton("设置安装方向")
            state.orient_apply.clicked.connect(
                lambda checked=False, part=state: self.set_orientation(part.device_id))
            orient_row.addWidget(state.orient_apply)
            panel_layout.addLayout(orient_row)
            bandwidth_row = QtWidgets.QHBoxLayout()
            state.bandwidth_combo = QtWidgets.QComboBox()
            for code, hz in enumerate((256, 188, 98, 42, 20, 10, 5)):
                state.bandwidth_combo.addItem(f"带宽 {hz} Hz", code)
            bandwidth_row.addWidget(state.bandwidth_combo, 1)
            state.bandwidth_apply = QtWidgets.QPushButton("设置")
            state.bandwidth_apply.clicked.connect(
                lambda checked=False, part=state: self.set_bandwidth(part.device_id))
            bandwidth_row.addWidget(state.bandwidth_apply)
            panel_layout.addLayout(bandwidth_row)
            rate_row = QtWidgets.QHBoxLayout()
            state.rate_spin = QtWidgets.QDoubleSpinBox()
            state.rate_spin.setRange(0.2, 200.0)
            state.rate_spin.setDecimals(1)
            state.rate_spin.setSingleStep(5.0)
            state.rate_spin.setSuffix(" Hz 回传")
            rate_row.addWidget(state.rate_spin, 1)
            state.rate_apply = QtWidgets.QPushButton("设置")
            state.rate_apply.clicked.connect(
                lambda checked=False, part=state: self.set_rate(part.device_id))
            rate_row.addWidget(state.rate_apply)
            panel_layout.addLayout(rate_row)
            gyro_row = QtWidgets.QHBoxLayout()
            state.gyro_combo = QtWidgets.QComboBox()
            for code, limit in enumerate((250, 500, 1000, 2000)):
                state.gyro_combo.addItem(f"陀螺 ±{limit} °/s", code)
            gyro_row.addWidget(state.gyro_combo, 1)
            state.gyro_apply = QtWidgets.QPushButton("设置")
            state.gyro_apply.clicked.connect(
                lambda checked=False, part=state: self.set_gyro_range(part.device_id))
            gyro_row.addWidget(state.gyro_apply)
            panel_layout.addLayout(gyro_row)
            acc_row = QtWidgets.QHBoxLayout()
            state.acc_combo = QtWidgets.QComboBox()
            for code, limit in enumerate((2, 4, 8, 16)):
                state.acc_combo.addItem(f"加计 ±{limit} g", code)
            acc_row.addWidget(state.acc_combo, 1)
            state.acc_apply = QtWidgets.QPushButton("设置")
            state.acc_apply.clicked.connect(
                lambda checked=False, part=state: self.set_acc_range(part.device_id))
            acc_row.addWidget(state.acc_apply)
            panel_layout.addLayout(acc_row)
            mag_start = QtWidgets.QPushButton("开始磁场校准")
            mag_start.clicked.connect(lambda checked=False, part=state: self.start_calibration(part.device_id, "mag"))
            panel_layout.addWidget(mag_start)
            mag_stop = QtWidgets.QPushButton("结束磁场校准并保存")
            mag_stop.clicked.connect(lambda checked=False, part=state: self.stop_calibration(part.device_id))
            panel_layout.addWidget(mag_stop)
            acc_button = QtWidgets.QPushButton("水平静止：加速度计校准")
            acc_button.clicked.connect(lambda checked=False, part=state: self.start_calibration(part.device_id, "acc"))
            panel_layout.addWidget(acc_button)
            heading_button = QtWidgets.QPushButton("Z 轴航向归零（仅 6 轴算法）")
            heading_button.clicked.connect(lambda checked=False, part=state: self.set_device_reference(part.device_id, 0x04))
            panel_layout.addWidget(heading_button)
            angle_button = QtWidgets.QPushButton("X/Y 角度参考归零")
            angle_button.clicked.connect(lambda checked=False, part=state: self.set_device_reference(part.device_id, 0x08))
            panel_layout.addWidget(angle_button)
            for button in (read_button, algorithm_button, mag_start, mag_stop, acc_button,
                           heading_button, angle_button):
                button.setEnabled(self.controls_enabled and bool(device_id))
                state.controls.append(button)
            state.algorithm_combo.setEnabled(self.controls_enabled and bool(device_id))
            state.orient_combo.setEnabled(self.controls_enabled and bool(device_id))
            state.bandwidth_combo.setEnabled(self.controls_enabled and bool(device_id))
            state.rate_spin.setEnabled(self.controls_enabled and bool(device_id))
            state.gyro_combo.setEnabled(self.controls_enabled and bool(device_id))
            state.acc_combo.setEnabled(self.controls_enabled and bool(device_id))
            state.controls.extend((state.algorithm_combo, state.orient_combo,
                                   state.bandwidth_combo,
                                   state.rate_spin, state.gyro_combo, state.acc_combo))
            state.neutral_button.setEnabled(bool(device_id))
            # Read current values first so the controls never start at a misleading default.
            state.orient_apply.setEnabled(False)
            state.bandwidth_apply.setEnabled(False)
            state.rate_apply.setEnabled(False)
            state.gyro_apply.setEnabled(False)
            state.acc_apply.setEnabled(False)
            panel_layout.addStretch(1)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setWidget(panel)
            controls.addTab(scroll, name)

        both_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        both_shortcut.activated.connect(self.set_neutral)
        self.shortcuts.append(both_shortcut)
        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.view, 1)
        body.addWidget(controls)
        layout.addLayout(body, 1)
        self.setCentralWidget(central)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    def show_help(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("WT901 快速帮助")
        dialog.resize(720, 520)
        layout = QtWidgets.QVBoxLayout(dialog)
        tabs = QtWidgets.QTabWidget()
        for title, html in HELP_PAGES:
            content = QtWidgets.QTextBrowser()
            content.setHtml(html)
            tabs.addTab(content, title)
        layout.addWidget(tabs)
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec()

    def _toggle_axes(self, visible: bool) -> None:
        for state in self.part_states.values():
            for axis in state.axes:
                axis.setVisible(visible)

    def _set_hand_transform(self, state: HandState, rotation: np.ndarray) -> None:
        transform = qt_matrix(rotation, state.y_offset)
        for part in (*state.parts, *state.axes):
            part.setTransform(transform)

    def _reset_orientation_streams(self, state: HandState) -> None:
        state.yaw_guard = None
        state.gyro_only = None
        state.tilt_filter = None
        state.tilt_estimate = None
        state.packet_clock = None
        state.last_device_tick_s = None
        state.stream_current.clear()
        state.stream_reference.clear()
        state.stream_health.clear()
        state.so3_projection_count = 0
        state.guarded_heading_deg = None
        state.guarded_active = False
        state.guarded_offset_deg = 0.0
        state.reference = None
        state.current = None
        self._set_hand_transform(state, np.eye(3))
        state.guard_status.setText("等待下一包数据为四种姿态建立共同中立时刻")

    def _guard_enter_changed(self, state: HandState, value: float) -> None:
        state.guard_exit_spin.setMaximum(value - 0.5)
        self._reset_orientation_streams(state)

    def _guard_exit_changed(self, state: HandState, value: float) -> None:
        state.guard_enter_spin.setMinimum(value + 0.5)
        self._reset_orientation_streams(state)

    def _update_orientation_streams(self, state: HandState, packet) -> None:
        if state.packet_clock is None:
            state.packet_clock = PacketClock()
            state.yaw_guard = YawGuard(state.guard_enter_spin.value(),
                                       state.guard_exit_spin.value())
            state.gyro_only = GyroOnly()
            state.tilt_filter = AdaptiveTiltFilter()
        t = state.packet_clock.elapsed(packet)
        rate = (packet.gyro_x, packet.gyro_y, packet.gyro_z)
        raw = rotation_matrix(packet.roll, packet.pitch, packet.yaw, "ZYX")
        raw, raw_health = check_so3(raw)
        result = state.yaw_guard.update(t, packet.roll, packet.pitch, packet.yaw, rate)
        gyro = state.gyro_only.update(t, rate)
        fused = state.tilt_filter.update(
            t, rate, (packet.acc_x, packet.acc_y, packet.acc_z))
        state.tilt_estimate = fused
        state.stream_health = {
            "raw": raw_health, "guard": state.yaw_guard.last_health,
            "gyro": state.gyro_only.last_health, "fusion": fused.health,
        }
        state.so3_projection_count += sum(
            health.projected for health in state.stream_health.values())
        for mode, rotation in (("raw", raw), ("guard", result.rotation),
                               ("gyro", gyro), ("fusion", fused.rotation)):
            state.stream_current[mode] = rotation
            if mode not in state.stream_reference:
                state.stream_reference[mode] = rotation.copy()
        state.guarded_heading_deg = result.heading_deg
        state.guarded_active = result.active
        state.guarded_offset_deg = result.yaw_offset_deg

    def _render_selected_pose(self, state: HandState) -> None:
        mode = state.mode_combo.currentData()
        state.current = state.stream_current.get(mode)
        state.reference = state.stream_reference.get(mode)
        if state.current is None or state.reference is None:
            self._set_hand_transform(state, np.eye(3))
            return
        rotation = model_rotation(state.reference, state.current, state.axis_mapping)
        self._set_hand_transform(state, rotation)
        x, y, z = xyz_angles(rotation)
        labels = {"raw": "厂商 RPY", "guard": "俯仰保护",
                  "gyro": "全程陀螺", "fusion": "陀螺+加计"}
        state.info.setText(
            f"{state.name} {state.device_id} [{labels[mode]}]："
            f"X {x:+6.1f}°  Y {y:+6.1f}°  Z {z:+6.1f}°"
            + (" · 保护中" if mode == "guard" and state.guarded_active else ""))

    def show_axis_mapping(self, state: HandState) -> None:
        dialog = AxisMappingDialog(state, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        state.axis_mapping = normalized_mapping(dialog.mapping())
        state.mapping_button.setText(
            "轴映射*" if state.axis_mapping != DEFAULT_AXIS_MAPPING else "轴映射")
        state.mapping_button.setToolTip("；".join(
            f"设备 +{axis} → 模型 {target}"
            for axis, target in zip("XYZ", state.axis_mapping)))
        self._render_selected_pose(state)
        state.status.setText("GUI 轴映射已更新；设备寄存器未改变")

    def open_device_setup(self):
        if not self.controls_enabled:
            return
        if self.setup_dialog is None:
            from wt901_viz.onboarding import DeviceSetupDialog
            self.setup_dialog = DeviceSetupDialog(self.source, self)
        else:
            self.setup_dialog.refresh_timer.start()
            self.setup_dialog.join_timer.start()
            self.setup_dialog.refresh()
        self.setup_dialog.show()
        self.setup_dialog.raise_()
        self.setup_dialog.activateWindow()

    def open_connected_devices(self):
        if not self.controls_enabled:
            return
        if self.devices_dialog is None:
            from wt901_viz.device_list import ConnectedDevicesDialog
            self.devices_dialog = ConnectedDevicesDialog(
                self.source, self, can_switch=self.can_switch_device_to_ap)
        else:
            self.devices_dialog.timer.start()
            self.devices_dialog.refresh()
        self.devices_dialog.show()
        self.devices_dialog.raise_()
        self.devices_dialog.activateWindow()

    def can_switch_device_to_ap(self, device_id: str) -> bool:
        state = self.hands.get(device_id)
        if state is not None and (state.calibration or state.setting is not None
                                  or state.waiting_address is not None or state.pending_reads):
            QtWidgets.QMessageBox.warning(
                self, "当前不能切换网络", f"{device_id} 正在校准或读写参数，请先完成当前操作。")
            return False
        return True

    def open_device_assignment(self):
        if not self.controls_enabled:
            return
        if self.assignment_dialog is None:
            from wt901_viz.assignment import DeviceAssignmentDialog
            self.assignment_dialog = DeviceAssignmentDialog(
                self.source, self.part_states, self.apply_assignments, self)
        else:
            if not self.assignment_dialog.isVisible():
                self.assignment_dialog.reset_choices()
            self.assignment_dialog.timer.start()
            self.assignment_dialog.refresh()
        self.assignment_dialog.show()
        self.assignment_dialog.raise_()
        self.assignment_dialog.activateWindow()

    def apply_assignments(self, assignments: dict[str, str]) -> bool:
        if set(assignments) != set(self.part_states):
            return False
        used = [device_id for device_id in assignments.values() if device_id]
        if len(used) != len(set(used)):
            return False
        if any(state.calibration or state.setting is not None or state.waiting_address is not None
               or state.pending_reads for state in self.part_states.values()):
            QtWidgets.QMessageBox.warning(
                self, "暂不能换绑", "请先完成当前设备的校准、参数设置或读取。")
            return False
        for key, state in self.part_states.items():
            device_id = assignments[key]
            if device_id == state.device_id:
                continue
            state.device_id = device_id
            self._reset_orientation_streams(state)
            state.last_packet_time = state.last_details_time = 0.0
            state.registers.clear()
            state.parameters.setText("寄存器参数：尚未读取")
            state.details.setText("等待实时数据" if device_id else "尚未分配设备")
            state.status.setText("已换绑；等待新设备建立中立位" if device_id else "尚未分配设备")
            state.info.setText(f"{state.name} {device_id}：等待数据" if device_id
                               else f"{state.name}：尚未分配设备")
            for control in state.controls:
                control.setEnabled(self.controls_enabled and bool(device_id))
            for button in (state.orient_apply, state.bandwidth_apply, state.rate_apply,
                           state.gyro_apply, state.acc_apply):
                button.setEnabled(False)
            state.neutral_button.setEnabled(bool(device_id))
        self.hands = {state.device_id: state for state in self.part_states.values()
                      if state.device_id}
        return True

    def _update_receive_rates(self):
        now = time.monotonic()
        if now - self._last_rate_refresh < 0.25:
            return
        self._last_rate_refresh = now
        if not hasattr(self.source, "receive_stats"):
            self.receive_rate_label.setText("有效收包频率：当前数据源不支持（请使用 UDP 模式）")
            return
        stats = self.source.receive_stats()
        assigned = {state.device_id: state.name for state in self.part_states.values()
                    if state.device_id}
        ids = sorted(set(assigned) | {key for key, value in stats.items() if value.age_ms <= 3000})
        rows = []
        for device_id in ids:
            name = f"{assigned[device_id]} " if device_id in assigned else "未分配 "
            value = stats.get(device_id)
            if value is None:
                text = "尚未收包"
            else:
                rate = "采样中…" if value.rate_hz is None else f"{value.rate_hz:.1f} Hz"
                text = f"{rate} · 距末包 {value.age_ms:.0f} ms"
            rows.append(f"{name}{device_id}：{text}")
        self.receive_rate_label.setText(
            "有效收包（近 2 秒，非画面帧率）：\n" + ("  |  ".join(rows) or "等待 WT 数据"))

    def poll(self):
        latest = {}
        now = time.monotonic()
        for packet in self.source.poll():
            state = self.hands.get(packet.device_id)
            if state is None:
                continue
            tick = (packet.hour * 3600 + packet.minute * 60 + packet.second
                    + packet.millisecond / 1000.0)
            gap = tick - state.last_device_tick_s if state.last_device_tick_s is not None else 0.0
            if gap < -43200:
                gap += 86400.0
            if gap > 1.0:
                # A real device-time gap, not a temporarily slow GUI render.
                # Unobserved rotation cannot be reconstructed reliably.
                self._reset_orientation_streams(state)
                state.status.setText("设备时间间断超过 1 秒；三条姿态已重新建立中立位")
            state.last_device_tick_s = tick
            latest[packet.device_id] = packet
            # The QTimer may receive several packets at once. Integrate every
            # gyro sample, not only the last packet rendered in this tick.
            self._update_orientation_streams(state, packet)
            state.last_packet_time = now
            if state.calibration in ("mag_precheck", "mag"):
                for i, value in enumerate((packet.mag_x, packet.mag_y, packet.mag_z)):
                    state.mag_min[i] = min(state.mag_min[i], value)
                    state.mag_max[i] = max(state.mag_max[i], value)
        self._update_receive_rates()
        if self.controls_enabled:
            for device_id, values in self.source.take_replies():
                state = self.hands.get(device_id)
                if state is not None and state.waiting_address is not None:
                    for i, value in enumerate(values):
                        state.registers[state.waiting_address + i] = value
                    if state.setting is not None and state.waiting_address == state.setting.address:
                        actual = values[0]
                        setting = state.setting
                        state.setting = None
                        state.waiting_address = None
                        if actual == setting.value:
                            self._show_parameters(state)
                            if setting.address in (REG_AXIS6, REG_ORIENT):
                                self._reset_orientation_streams(state)
                                latest.pop(device_id, None)
                            state.status.setText(
                                f"{state.name} {setting.label}：已保存并回读确认"
                                + ("；下一包数据将作为新的手腕中立位"
                                   if setting.address in (REG_AXIS6, REG_ORIENT) else "")
                                + ("；手册要求重新校准，尚未自动执行"
                                   if setting.address == REG_ORIENT else "")
                            )
                        else:
                            state.status.setText(
                                f"{setting.label}未确认：期望寄存器值 {setting.value}，"
                                f"设备回读 {actual}；请重新读取参数"
                            )
                        continue
                    state.waiting_address = None
                    self._request_next_register(state)
        for device_id, state in self.hands.items():
            if state.waiting_address is not None and now > state.read_deadline:
                state.waiting_address = None
                state.pending_reads.clear()
                if state.setting is not None:
                    label = state.setting.label
                    state.setting = None
                    state.status.setText(f"{label}命令已发送，但回读超时；请重新读取参数确认")
                else:
                    state.status.setText("读取参数超时；可重试")
            packet = latest.get(device_id)
            if packet is None:
                if state.current is not None and now - state.last_packet_time > 1.0:
                    state.info.setText(f"{state.name} {device_id}：数据中断；保持最后姿势")
                elif state.current is None and now - self.started_at > 2.0:
                    state.info.setText(f"{state.name} {device_id}：未连接，等待数据")
                if state.calibration == "mag_precheck" and now - state.last_packet_time > 3.0:
                    state.calibration = None
                    state.status.setText("预检时设备离线，未写入；请检查热点连接")
                continue
            self._render_selected_pose(state)
            if now - state.last_details_time >= 0.2:
                state.last_details_time = now
                ip = self.source.device_ip(device_id) if self.controls_enabled else "TCP"
                battery_percent = packet.battery_percent
                battery = f"约 {battery_percent}%" if battery_percent is not None else "未知"
                warning = "；低电量" if battery_percent is not None and battery_percent <= 5 else ""
                state.details.setText(
                    f"IP: {ip}    RSSI: {packet.rssi} dBm    "
                    f"电池: {packet.battery_v:.2f} V（{battery}{warning}）\n"
                    f"原始角度 R/P/Y: {packet.roll:+.1f} / {packet.pitch:+.1f} / {packet.yaw:+.1f} °\n"
                    f"加速度 X/Y/Z: {packet.acc_x:+.3f} / {packet.acc_y:+.3f} / {packet.acc_z:+.3f} g\n"
                    f"角速度 X/Y/Z: {packet.gyro_x:+.1f} / {packet.gyro_y:+.1f} / {packet.gyro_z:+.1f} °/s\n"
                    f"磁场 X/Y/Z: {packet.mag_x:+.1f} / {packet.mag_y:+.1f} / {packet.mag_z:+.1f} µT\n"
                    f"温度: {packet.temperature:.1f} °C    固件: {packet.version}"
                )
                guard_mode = "保护中：三轴陀螺积分" if state.guarded_active else "未进入保护区"
                fused = state.tilt_estimate
                bias = "/".join(f"{value:+.2f}" for value in fused.gyro_bias_dps)
                selected_health = state.stream_health[state.mode_combo.currentData()]
                state.guard_status.setText(
                    f"四条姿态同时计算；|Pitch| {abs(packet.pitch):.1f}°；"
                    f"{guard_mode}；保护航向修正 {state.guarded_offset_deg:+.1f}°。\n"
                    f"融合零偏 X/Y/Z：{bias} °/s"
                    f"（{'已建立' if fused.bias_ready else '待静止约 1 秒'}）；"
                    f"加计权重 {fused.accelerometer_weight:.2f}。\n"
                    f"当前模式距 SO(3)：{selected_health.distance_to_so3:.2e}；"
                    f"SVD 投影累计 {state.so3_projection_count} 次。")
            if state.calibration in ("mag_precheck", "mag"):
                span, covered = mag_coverage(state)
                if state.calibration == "mag_precheck":
                    elapsed = now - state.calibration_started
                    weak_axes = [axis for axis, value in zip("XYZ", span)
                                 if value < MAG_MIN_AXIS_SPAN_UT]
                    reasons = ([f"{axis}轴不足30 µT" for axis in weak_axes]
                               + (["最大轴跨度不足50 µT"] if max(span) < MAG_MAX_AXIS_SPAN_UT else []))
                    missing = "；".join(reasons) if reasons else "无"
                    state.status.setText(
                        f"磁场预检 {elapsed:.0f} 秒；三轴跨度 "
                        f"{span[0]:.1f}/{span[1]:.1f}/{span[2]:.1f} µT；"
                        f"未达标项：{missing}。"
                        + ("请继续充分翻转设备。" if elapsed < MAG_PRECHECK_MIN_SECONDS or not covered
                           else "覆盖度已达标。")
                    )
                    if elapsed >= MAG_PRECHECK_MIN_SECONDS and covered:
                        self._begin_calibration(state, "mag")
                    elif elapsed >= MAG_PRECHECK_MAX_SECONDS:
                        state.calibration = None
                        state.status.setText(
                            f"预检超过 {MAG_PRECHECK_MAX_SECONDS:.0f} 秒仍未达标，未写入。"
                            f"三轴跨度 {span[0]:.1f}/{span[1]:.1f}/{span[2]:.1f} µT；"
                            f"未达标项：{missing}。"
                        )
                else:
                    state.status.setText(
                        f"磁校准中 {now - state.calibration_started:.0f} 秒；"
                        f"磁场跨度 X/Y/Z: {span[0]:.1f}/{span[1]:.1f}/{span[2]:.1f} µT。"
                        "绕三轴充分转动，完成后点击结束并保存。"
                    )

    def read_parameters(self, device_id: str):
        state = self.hands[device_id]
        if state.calibration or state.setting is not None or state.waiting_address is not None:
            state.status.setText("请先完成当前操作，再读取参数")
            return
        try:
            self.source.device_ip(device_id)
        except RuntimeError as exc:
            state.status.setText(str(exc))
            return
        state.registers.clear()
        state.orient_apply.setEnabled(False)
        state.bandwidth_apply.setEnabled(False)
        state.rate_apply.setEnabled(False)
        state.gyro_apply.setEnabled(False)
        state.acc_apply.setEnabled(False)
        # Reading ORIENT also returns AXIS6 in the following register slot.
        state.pending_reads = [REG_CALSW, REG_BANDWIDTH, REG_ORIENT, REG_RATE]
        state.waiting_address = None
        state.status.setText("正在读取设备参数…")
        self._request_next_register(state)

    def _show_parameters(self, state: HandState):
        values = state.registers
        bandwidth = ["256", "188", "98", "42", "20", "10", "5"]
        gyro_range = [250, 500, 1000, 2000]
        acc_range = [2, 4, 8, 16]
        bw = values.get(REG_BANDWIDTH, -1)
        gr = values.get(REG_GYRORANGE, -1)
        ar = values.get(REG_ACCRANGE, -1)
        rate_ms = values.get(REG_RATE, 0)
        lines = [f"回传频率: {1000 / rate_ms:.1f} Hz（{rate_ms} ms）"
                 if rate_ms else "回传频率: 未知"]
        algorithm = {0: "9 轴", 1: "6 轴"}.get(values.get(REG_AXIS6), "未知")
        lines.append(f"融合算法: {algorithm}")
        orient = {0: "水平安装", 1: "垂直安装（Y 轴向上）"}.get(values.get(REG_ORIENT), "未知")
        lines.append(f"安装方向: {orient}")
        lines.append(f"带宽: {bandwidth[bw]} Hz" if 0 <= bw < len(bandwidth) else "带宽: 未知")
        lines.append(f"陀螺量程: ±{gyro_range[gr]} °/s"
                     if 0 <= gr < len(gyro_range) else "陀螺量程: 未知")
        lines.append(f"加计量程: ±{acc_range[ar]} g"
                     if 0 <= ar < len(acc_range) else "加计量程: 未知")
        lines.append(f"校准模式: 0x{values.get(REG_CALSW, 0):04X}")
        state.parameters.setText("\n".join(lines))
        mode = values.get(REG_AXIS6)
        if mode in (0, 1):
            state.algorithm_combo.setCurrentIndex(state.algorithm_combo.findData(mode))
        orientation = values.get(REG_ORIENT)
        if orientation in (0, 1):
            state.orient_combo.setCurrentIndex(state.orient_combo.findData(orientation))
            state.orient_apply.setEnabled(self.controls_enabled)
        if 0 <= bw <= 6:
            state.bandwidth_combo.setCurrentIndex(state.bandwidth_combo.findData(bw))
            state.bandwidth_apply.setEnabled(self.controls_enabled)
        if rate_ms and 0.2 <= 1000 / rate_ms <= 200:
            state.rate_spin.setValue(1000 / rate_ms)
            state.rate_apply.setEnabled(self.controls_enabled)
        if 0 <= gr <= 3:
            state.gyro_combo.setCurrentIndex(state.gyro_combo.findData(gr))
            state.gyro_apply.setEnabled(self.controls_enabled)
        if 0 <= ar <= 3:
            state.acc_combo.setCurrentIndex(state.acc_combo.findData(ar))
            state.acc_apply.setEnabled(self.controls_enabled)

    def _request_next_register(self, state: HandState):
        if not state.pending_reads:
            self._show_parameters(state)
            state.status.setText("设备参数读取完成")
            return
        address = state.pending_reads.pop(0)
        try:
            self.source.read_registers(state.device_id, address)
        except (RuntimeError, OSError) as exc:
            state.pending_reads.clear()
            state.status.setText(f"读取失败：{exc}")
            return
        state.waiting_address = address
        state.read_deadline = time.monotonic() + 2.0

    def switch_algorithm(self, device_id: str, target: int):
        if target not in (0, 1):
            self.hands[device_id].status.setText("无效的算法选择")
            return
        name = "6 轴（相对航向）" if target else "9 轴（磁场辅助航向）"
        self.apply_setting(device_id, RegisterSetting(REG_AXIS6, target, f"切换为{name}"))

    def set_orientation(self, device_id: str):
        state = self.hands[device_id]
        target = state.orient_combo.currentData()
        if target not in (0, 1):
            state.status.setText("无效的安装方向")
            return
        if state.registers.get(REG_ORIENT) == target:
            state.status.setText("安装方向已经是所选模式，无需写入")
            return
        label = "垂直安装（设备 Y 轴箭头朝上）" if target else "水平安装"
        self.apply_setting(device_id, RegisterSetting(REG_ORIENT, target, f"设置为{label}"))

    def set_bandwidth(self, device_id: str):
        target = self.hands[device_id].bandwidth_combo.currentData()
        bandwidths = (256, 188, 98, 42, 20, 10, 5)
        if target not in range(len(bandwidths)):
            self.hands[device_id].status.setText("无效的带宽档位")
            return
        self.apply_setting(device_id, RegisterSetting(
            REG_BANDWIDTH, target, f"设置带宽为 {bandwidths[target]} Hz"))

    def set_rate(self, device_id: str):
        requested_hz = self.hands[device_id].rate_spin.value()
        if not 0.2 <= requested_hz <= 200:
            self.hands[device_id].status.setText("回传频率必须在 0.2～200 Hz")
            return
        period_ms = max(5, min(5000, round(1000 / requested_hz)))
        actual_hz = 1000 / period_ms
        self.apply_setting(device_id, RegisterSetting(
            REG_RATE, period_ms,
            f"设置回传频率为约 {actual_hz:.1f} Hz（周期 {period_ms} ms）"))

    def set_gyro_range(self, device_id: str):
        target = self.hands[device_id].gyro_combo.currentData()
        limits = (250, 500, 1000, 2000)
        if target not in range(len(limits)):
            self.hands[device_id].status.setText("无效的陀螺量程")
            return
        self.apply_setting(device_id, RegisterSetting(
            REG_GYRORANGE, target, f"设置陀螺量程为 ±{limits[target]} °/s"))

    def set_acc_range(self, device_id: str):
        target = self.hands[device_id].acc_combo.currentData()
        limits = (2, 4, 8, 16)
        if target not in range(len(limits)):
            self.hands[device_id].status.setText("无效的加计量程")
            return
        self.apply_setting(device_id, RegisterSetting(
            REG_ACCRANGE, target, f"设置加计量程为 ±{limits[target]} g"))

    def apply_setting(self, device_id: str, setting: RegisterSetting):
        state = self.hands[device_id]
        if state.calibration or state.waiting_address is not None or state.setting is not None:
            state.status.setText("请先完成当前校准或参数读取")
            return
        try:
            self.source.device_ip(device_id)
        except (RuntimeError, OSError) as exc:
            state.status.setText(str(exc))
            return
        extra = ("\n手册要求切换安装方向后重新校准；本操作不会自动启动校准。"
                 if setting.address == REG_ORIENT else "")
        if QtWidgets.QMessageBox.question(
                self, "确认设备设置",
                f"确认对 {state.name}（{device_id}）{setting.label}？\n"
                f"设置会保存到设备，随后自动回读确认。{extra}") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        state.setting = setting
        state.status.setText(f"正在{setting.label}…")
        try:
            self.source.unlock(device_id)
        except (RuntimeError, OSError) as exc:
            state.setting = None
            state.status.setText(f"解锁失败：{exc}")
            return

        def write_setting():
            if state.setting is not setting:
                return
            try:
                self.source.write_register(device_id, setting.address, setting.value)
            except (RuntimeError, OSError) as exc:
                state.setting = None
                state.status.setText(f"写入失败：{exc}")
                return
            QtCore.QTimer.singleShot(350, save_setting)

        def save_setting():
            if state.setting is not setting:
                return
            try:
                self.source.write_register(device_id, REG_SAVE, 0)
            except (RuntimeError, OSError) as exc:
                state.setting = None
                state.status.setText(f"设置可能已生效但保存失败：{exc}；请读取参数确认")
                return
            QtCore.QTimer.singleShot(350, verify_setting)

        def verify_setting():
            if state.setting is not setting:
                return
            state.waiting_address = setting.address
            state.read_deadline = time.monotonic() + 2.0
            try:
                self.source.read_registers(device_id, setting.address)
            except (RuntimeError, OSError) as exc:
                state.waiting_address = None
                state.setting = None
                state.status.setText(f"设置命令已发送但回读失败：{exc}；请读取参数确认")

        QtCore.QTimer.singleShot(int(UNLOCK_SETTLE * 1000) + 50, write_setting)

    def start_calibration(self, device_id: str, kind: str):
        state = self.hands[device_id]
        if state.calibration or state.setting is not None:
            state.status.setText("该设备已有操作，先完成它")
            return
        if state.waiting_address is not None:
            state.status.setText("请等待参数读取结束")
            return
        if kind == "acc" and state.registers.get(REG_ORIENT) == 1:
            state.status.setText(
                "当前为垂直安装；手册未明确该模式下的加计校准摆放，"
                "本“水平静止”入口暂不执行。请先确认厂家校准步骤。")
            return
        try:
            self.source.device_ip(device_id)
        except RuntimeError as exc:
            state.status.setText(str(exc))
            return
        if kind == "mag":
            message = (f"确认校准 {state.name}（{device_id}）的磁场？\n"
                       "先进行至少 20 秒三轴覆盖预检，若未达标可继续翻转至多 90 秒。"
                       "正式校准时继续转动，完成时手动点击‘结束并保存’。")
        else:
            message = (f"确认校准 {state.name}（{device_id}）的加速度计？\n"
                       "将设备正面朝上、水平放稳，之后 5 秒内不要移动。")
        if QtWidgets.QMessageBox.question(self, "确认校准", message) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if kind == "mag":
            state.calibration = "mag_precheck"
            state.calibration_started = time.monotonic()
            state.mag_min = [float("inf")] * 3
            state.mag_max = [float("-inf")] * 3
            state.status.setText("磁场预检中：持续绕 X/Y/Z 轴充分翻转设备，达标后自动开始校准")
            return
        self._begin_calibration(state, kind)

    def _begin_calibration(self, state: HandState, kind: str):
        device_id = state.device_id
        try:
            self.source.unlock(device_id)
        except (RuntimeError, OSError) as exc:
            state.calibration = None
            state.status.setText(f"无法开始校准：{exc}")
            return
        state.calibration = f"{kind}_pending"
        state.status.setText("已发送解锁，等待进入校准模式…")

        def enter():
            if state.calibration != f"{kind}_pending":
                return
            try:
                self.source.write_register(device_id, REG_CALSW,
                                           0x07 if kind == "mag" else 0x01)
            except (RuntimeError, OSError) as exc:
                state.calibration = None
                state.status.setText(f"进入校准失败：{exc}")
                return
            state.calibration = kind
            state.calibration_started = time.monotonic()
            state.mag_min = [float("inf")] * 3
            state.mag_max = [float("-inf")] * 3
            if kind == "mag":
                state.status.setText("磁场校准中：绕 X/Y/Z 轴转动，完成后点击结束并保存")
            else:
                state.status.setText("加速度计校准中：请保持水平静止 5 秒")
                QtCore.QTimer.singleShot(5000, lambda: self.stop_calibration(device_id))

        QtCore.QTimer.singleShot(int(UNLOCK_SETTLE * 1000) + 50, enter)

    def set_device_reference(self, device_id: str, mode: int):
        state = self.hands[device_id]
        if state.calibration or state.waiting_address is not None or state.setting is not None:
            state.status.setText("请先完成当前校准或参数读取")
            return
        if mode == 0x04 and state.registers.get(REG_AXIS6) != 1:
            state.status.setText("Z 轴航向归零只适用于 6 轴算法；请先读取参数确认")
            return
        try:
            self.source.device_ip(device_id)
        except RuntimeError as exc:
            state.status.setText(str(exc))
            return
        action = "Z 轴航向归零" if mode == 0x04 else "X/Y 角度参考归零"
        if QtWidgets.QMessageBox.question(
                self, "确认设备操作",
                f"确认对 {state.name}（{device_id}）执行{action}？\n"
                "这会改变设备输出的姿态角；GUI 中可再按 1/2 设定手腕中立位。") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self.source.unlock(device_id)
        except (RuntimeError, OSError) as exc:
            state.status.setText(f"操作失败：{exc}")
            return
        state.calibration = "reference_pending"
        state.status.setText(f"正在执行{action}…")

        def apply_reference():
            if state.calibration != "reference_pending":
                return
            try:
                self.source.write_register(device_id, REG_CALSW, mode)
            except (RuntimeError, OSError) as exc:
                state.calibration = None
                state.status.setText(f"操作失败：{exc}")
                return
            state.calibration = "reference"
            QtCore.QTimer.singleShot(400, lambda: self.stop_calibration(device_id))

        QtCore.QTimer.singleShot(int(UNLOCK_SETTLE * 1000) + 50, apply_reference)

    def stop_calibration(self, device_id: str):
        state = self.hands[device_id]
        if state.calibration is None:
            state.status.setText("该设备当前没有正在进行的校准")
            return
        if state.calibration.endswith("_pending") or state.calibration == "mag_precheck":
            state.calibration = None
            state.status.setText("已取消，尚未进入校准模式")
            return
        if state.calibration == "stopping":
            return
        previous_mode = state.calibration
        state.calibration = "stopping"
        state.status.setText("正在退出校准并保存…")
        try:
            # Calibration firmware may pause streaming; use the IP already
            # verified when this calibration was started.
            self.source.unlock(device_id, max_age=None)
        except (RuntimeError, OSError) as exc:
            state.status.setText(f"退出校准失败：{exc}；请保持设备在线并重试")
            state.calibration = previous_mode
            return

        def finish():
            try:
                self.source.write_register(device_id, REG_CALSW, 0, max_age=None)
                QtCore.QTimer.singleShot(350, save_result)
            except (RuntimeError, OSError) as exc:
                state.status.setText(f"退出校准失败：{exc}；请重试")
                state.calibration = previous_mode

        def save_result():
            try:
                self.source.write_register(device_id, REG_SAVE, 0, max_age=None)
                state.status.setText("已发送退出校准与保存指令；建议重新读取参数确认模式为 0")
                state.calibration = None
            except (RuntimeError, OSError) as exc:
                state.status.setText(f"保存失败：{exc}；请重试结束并保存")
                state.calibration = previous_mode

        QtCore.QTimer.singleShot(int(UNLOCK_SETTLE * 1000) + 50, finish)

    def _receive_diagnostics_finished(self):
        if self._closing_for_receive_diagnostics:
            QtCore.QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        if self.setup_dialog is not None and self.setup_dialog._process is not None:
            self.setup_dialog.state_label.setText("网络切换正在进行，请等待完成后再关闭主窗口。")
            event.ignore()
            return
        if self.devices_dialog is not None and self.devices_dialog._process is not None:
            self.devices_dialog.status.setText("设备列表正在刷新，请等待完成后再关闭主窗口。")
            event.ignore()
            return
        if self.assignment_dialog is not None and self.assignment_dialog._process is not None:
            self.assignment_dialog.status.setText("设备分配列表正在刷新，请等待完成后再关闭主窗口。")
            event.ignore()
            return
        if any(state.setting is not None for state in self.hands.values()):
            for state in self.hands.values():
                if state.setting is not None:
                    state.status.setText("设备设置正在保存/回读，请等待结束后再关闭窗口")
            event.ignore()
            return
        if self.receive_diagnostics.recorder is not None:
            self._closing_for_receive_diagnostics = True
            self.receive_diagnostics.stop("window_close")
            event.ignore()
            return
        # Best-effort emergency exit if the user closes the window mid-calibration.
        for state in self.hands.values():
            if state.calibration and state.calibration != "mag_precheck" and not state.calibration.endswith("_pending"):
                try:
                    self.source.unlock(state.device_id, max_age=None)
                    time.sleep(UNLOCK_SETTLE)
                    self.source.write_register(state.device_id, REG_CALSW, 0, max_age=None)
                    time.sleep(0.3)
                    self.source.write_register(state.device_id, REG_SAVE, 0, max_age=None)
                except (RuntimeError, OSError) as exc:
                    print(f"[警告] {state.device_id} 退出校准失败: {exc}", file=sys.stderr)
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.set_neutral()
        elif event.key() == QtCore.Qt.Key.Key_1:
            self.set_neutral(self.part_states["left_hand"].device_id)
        elif event.key() == QtCore.Qt.Key.Key_2:
            self.set_neutral(self.part_states["right_hand"].device_id)
        elif event.key() == QtCore.Qt.Key.Key_R:
            self.view.setCameraPosition(distance=4.5, elevation=35, azimuth=-65)
        elif event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)

    def set_neutral(self, device_id: str | None = None):
        for state in self.hands.values():
            if device_id is not None and state.device_id != device_id:
                continue
            if not state.stream_current:
                continue
            state.stream_reference = {mode: rotation.copy()
                                      for mode, rotation in state.stream_current.items()}
            self._render_selected_pose(state)


def main() -> int:
    parser = argparse.ArgumentParser(description="WT901 双手手腕转角 3D 试验")
    parser.add_argument("--net", choices=("udp", "tcp"), default="udp",
                        help="接收协议；默认 UDP")
    parser.add_argument("--tcp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--listen-ip", default="0.0.0.0",
                        help="本机监听地址；默认所有网卡，允许在电脑热点与设备 AP 间切换")
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--left-id", default="WT5500011381", help="左手设备 ID")
    parser.add_argument("--right-id", default="WT5500011408", help="右手设备 ID")
    parser.add_argument("--device-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device_id:
        args.left_id = args.device_id  # Compatibility with the single-hand version.
    if args.left_id == args.right_id:
        parser.error("左右手设备 ID 不能相同")
    from wt901.net_source import TcpAttitudeSource
    from wt901.gui_udp import GuiUdpSource

    network = "tcp" if args.tcp else args.net
    source_type = TcpAttitudeSource if network == "tcp" else GuiUdpSource
    # On this machine Qt's xcb backend does not give PyOpenGL a usable
    # context under Wayland. The existing attitude viewer uses the same fix.
    if (os.environ.get("WAYLAND_DISPLAY")
            and os.environ.get("QT_QPA_PLATFORM", "").startswith("xcb")):
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        print("[提示] 已切换 Qt 到 Wayland，以便显示 3D 模型", file=sys.stderr)
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        source = source_type(args.listen_port, host=args.listen_ip)
    except OSError as exc:
        parser.exit(1, f"无法监听 {args.listen_ip}:{args.listen_port} ({network.upper()}): {exc}\n"
                       "请确认热点已启动、监听地址正确，且端口未被其他程序占用。\n")
    with source:
        window = WristWindow(source, args.left_id, args.right_id,
                             f"{network.upper()} {args.listen_ip}:{args.listen_port}")
        window.show()
        return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
