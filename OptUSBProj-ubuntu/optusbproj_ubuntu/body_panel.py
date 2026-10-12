"""Main-window panel construction; keeps widget attributes and layout order stable."""
from PySide6 import QtCore, QtGui, QtWidgets
from .skeleton import PARTS
from .pose import GYRO_TILT, VENDOR_RPY, GYRO_RGBD


def build_body_panel(window, layout):
    panel = QtWidgets.QWidget()
    panel.setMinimumWidth(390)
    panel.setMaximumWidth(540)
    panel_layout = QtWidgets.QVBoxLayout(panel)
    layout.addWidget(panel, 2)

    title = QtWidgets.QLabel("全身设备分配与姿态")
    font = title.font()
    font.setPointSize(16)
    font.setBold(True)
    title.setFont(font)
    panel_layout.addWidget(title)
    window.overview = QtWidgets.QLabel("等待 WT901 UDP 数据…")
    window.overview.setWordWrap(True)
    panel_layout.addWidget(window.overview)

    toolbar = QtWidgets.QHBoxLayout()
    for label, callback in (("热点设备", window._show_devices),
                            ("设备参数", window._show_settings),
                            ("接入新设备", window._show_setup),
                            ("全部归零", window._zero_all)):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(callback)
        if window.replay_mode and label != "全部归零":
            button.setEnabled(False)
        toolbar.addWidget(button)
    panel_layout.addLayout(toolbar)
    timed_zero_row = QtWidgets.QHBoxLayout()
    timed_zero_row.addWidget(QtWidgets.QLabel("延时"))
    window.zero_delay = QtWidgets.QSpinBox()
    window.zero_delay.setRange(3, 120)
    window.zero_delay.setValue(10)
    window.zero_delay.setSuffix(" 秒")
    window.zero_delay.setToolTip("点击开始后，可在倒计时期间穿戴设备并摆好中立姿势")
    timed_zero_row.addWidget(window.zero_delay)
    window.timed_zero_button = QtWidgets.QPushButton("定时全部归零")
    window.timed_zero_button.clicked.connect(window._start_timed_zero)
    timed_zero_row.addWidget(window.timed_zero_button, 1)
    window.cancel_zero_button = QtWidgets.QPushButton("取消")
    window.cancel_zero_button.setEnabled(False)
    window.cancel_zero_button.clicked.connect(window._cancel_timed_zero)
    timed_zero_row.addWidget(window.cancel_zero_button)
    panel_layout.addLayout(timed_zero_row)
    window.zero_countdown = QtWidgets.QLabel("定时归零未启动")
    window.zero_countdown.setWordWrap(True)
    panel_layout.addWidget(window.zero_countdown)
    window.calibration_button = QtWidgets.QPushButton("穿戴标定…（躯干 / 双臂 / 双手）")
    window.calibration_button.clicked.connect(window._show_calibration)
    window.calibration_button.setEnabled(not window.replay_mode)
    panel_layout.addWidget(window.calibration_button)
    window.bilateral_calibration_button = QtWidgets.QPushButton("双臂联合标定…（左右上臂 / 前臂）")
    window.bilateral_calibration_button.clicked.connect(window._show_bilateral_calibration)
    window.bilateral_calibration_button.setEnabled(not window.replay_mode)
    panel_layout.addWidget(window.bilateral_calibration_button)
    window.wearing_options_button = QtWidgets.QPushButton('穿戴标定参数…（弱先验 λ）')
    window.wearing_options_button.clicked.connect(window._show_wearing_options)
    window.wearing_options_button.setEnabled(not window.replay_mode)
    panel_layout.addWidget(window.wearing_options_button)
    window.natural_hand_button=QtWidgets.QPushButton('左手自然掌面参考…（手动原型）')
    window.natural_hand_button.clicked.connect(window._show_natural_hand)
    window.natural_hand_button.setEnabled(not window.replay_mode);panel_layout.addWidget(window.natural_hand_button)
    window.x1_preview_button = QtWidgets.QPushButton("X1 双臂实时预览…（RViz）")
    window.x1_preview_button.clicked.connect(window._show_x1_preview)
    window.x1_preview_button.setEnabled(not window.replay_mode)
    panel_layout.addWidget(window.x1_preview_button)
    window.scene_robot_button = QtWidgets.QPushButton("新机器人双臂实时预览…（CoppeliaSim）")
    window.scene_robot_button.clicked.connect(window._show_scene_robot)
    window.scene_robot_button.setEnabled(not window.replay_mode)
    panel_layout.addWidget(window.scene_robot_button)
    window.gripper_gesture_button = QtWidgets.QPushButton('左右手 IMU 夹爪配置…（四设备 / RViz）')
    window.gripper_gesture_button.clicked.connect(window._show_gripper_gesture)
    window.gripper_gesture_button.setEnabled(not window.replay_mode)
    panel_layout.addWidget(window.gripper_gesture_button)
    window._timed_zero_deadline: float | None = None
    window.zero_countdown_timer = QtCore.QTimer(window)
    window.zero_countdown_timer.setInterval(100)
    window.zero_countdown_timer.timeout.connect(window._update_zero_countdown)
    window.axes_checkbox = QtWidgets.QCheckBox("显示部位坐标轴（红 X · 绿 Y · 蓝 Z）")
    window.axes_checkbox.setChecked(True)
    window.axes_checkbox.toggled.connect(window._toggle_axes)
    panel_layout.addWidget(window.axes_checkbox)
    window.record_button = QtWidgets.QPushButton("开始录制 CSV")
    window.record_button.clicked.connect(window._toggle_recording)
    window.record_button.setEnabled(not window.replay_mode)
    panel_layout.addWidget(window.record_button)

    group = QtWidgets.QGroupBox("设备 ID → 身体部位（支持部分设备先行测试）")
    grid = QtWidgets.QGridLayout(group)
    section_color = ("#76D7FF" if group.palette().color(
        QtGui.QPalette.ColorRole.Window).lightness() < 128 else "#006D9B")
    grid.addWidget(QtWidgets.QLabel("部位"), 0, 0)
    grid.addWidget(QtWidgets.QLabel("轴映射"), 0, 1)
    grid.addWidget(QtWidgets.QLabel("设备 ID"), 0, 2)
    grid.addWidget(QtWidgets.QLabel("中立位"), 0, 3)
    row = 1
    section = ""
    for part in (p for p in PARTS if p.sensor_slot):
        next_section = ("骨盆 / 躯干 / 头部" if part.key in ("pelvis", "torso", "head")
                        else "左臂" if part.key.startswith("left_") and
                        part.key in ("left_upper_arm", "left_forearm", "left_hand")
                        else "右臂" if part.key.startswith("right_") and
                        part.key in ("right_upper_arm", "right_forearm", "right_hand")
                        else "左腿" if part.key.startswith("left_") else "右腿")
        if next_section != section:
            section = next_section
            heading = QtWidgets.QLabel(section)
            heading.setObjectName("bodySectionHeading")
            heading.setStyleSheet(
                f"color: {section_color}; font-weight: bold; "
                "padding-top: 10px; padding-bottom: 2px;")
            grid.addWidget(heading, row, 0, 1, 4)
            row += 1
        grid.addWidget(QtWidgets.QLabel(part.label), row, 0)
        mapping_button = QtWidgets.QPushButton("轴映射")
        mapping_button.clicked.connect(
            lambda _checked=False, key=part.key: window._show_axis_mapping(key))
        grid.addWidget(mapping_button, row, 1)
        window.mapping_buttons[part.key] = mapping_button
        window._update_mapping_button(part.key)
        combo = QtWidgets.QComboBox()
        combo.setMinimumWidth(165)
        combo.currentIndexChanged.connect(
            lambda _index, key=part.key: window._assignment_changed(key))
        grid.addWidget(combo, row, 2)
        window.combos[part.key] = combo
        zero = QtWidgets.QPushButton("归零")
        zero.clicked.connect(lambda _checked=False, key=part.key: window._zero(key))
        grid.addWidget(zero, row, 3)
        status = QtWidgets.QLabel("未分配")
        status.setWordWrap(True)
        algorithm = QtWidgets.QComboBox()
        algorithm.addItem("厂商 R/P/Y", VENDOR_RPY)
        algorithm.addItem("陀螺＋加计", GYRO_TILT)
        if part.key in ('left_upper_arm','left_forearm','right_upper_arm','right_forearm'):
            algorithm.addItem("陀螺＋加计＋RGB-D（需相机标定）",GYRO_RGBD)
        algorithm.setCurrentIndex(algorithm.findData(window.engine.algorithm_modes[part.key]))
        algorithm.setToolTip("仅切换本部位软件姿态估计；不改变设备的 6/9 轴寄存器设置")
        algorithm.currentIndexChanged.connect(
            lambda _index, key=part.key: window._algorithm_changed(key))
        grid.addWidget(algorithm, row + 1, 0, 1, 2)
        window.algorithm_combos[part.key] = algorithm
        grid.addWidget(status, row + 1, 2, 1, 2)
        window.status_labels[part.key] = status
        row += 2
    assignment_scroll = QtWidgets.QScrollArea()
    assignment_scroll.setWidgetResizable(True)
    assignment_scroll.setHorizontalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    assignment_scroll.setWidget(group)
    assignment_scroll.setMinimumHeight(280)
    panel_layout.addWidget(assignment_scroll, 1)
    refresh = QtWidgets.QPushButton("刷新在线设备列表")
    refresh.clicked.connect(window._refresh_choices)
    panel_layout.addWidget(refresh)

    hint = QtWidgets.QLabel(
        "模型 15 个部件均可绑定 IMU；不必一次接满。"
        "未绑定部位跟随父部位。每只设备首次收到数据时自动建立中立位。"
        "彩色箭头是模型各部位的局部轴，不是 WT901 外壳上的轴。"
        "归零只重设姿态基准，不改变设备轴映射；更换绑法后需重新设置或标定。"
        "空格：立即全部归零；定时归零可留出穿戴和摆姿势的时间；R：重置视角。\n"
        "本窗口占用 UDP 1399，运行时请关闭手腕控制台。")
    hint.setWordWrap(True)
    panel_layout.addWidget(hint)
    window.statusBar().showMessage("就绪；等待设备数据")
    window._refresh_choices()
    return panel
