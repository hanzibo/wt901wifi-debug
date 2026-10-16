"""Fusion UI and rendering, delegating lifecycle/application/recording coordination."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyvista as pv

from .camera_gui import CameraWorker
from .camera_preview import CameraPreview, CameraPreviewWindow
from .camera_calibration_gui import CameraCalibrationDialog
from .orbbec_worker import OrbbecWorker, DEFAULT_PYTHON
from .rgbd_pose_worker import RGBDPoseWorker
from .pose_2d import Pose2DConfig,DEFAULT_YOLO
from .person_target import PersonTarget
from .rgbd_pose_gui import RGBDPoseDialog, draw_pose_rgb
from .fusion_capture import CameraObservation
from .fusion_recording import FusionSessionRecorder
from .fusion_replay import FusionSessionReplay
from .gui import BodyWindow, MODEL_DIR, _axis_triad
from .pose import PoseEngine,GYRO_RGBD,GYRO_TILT
from .live_rgbd_fusion import LiveRGBDFusion, ARM_SEGMENTS
from .skeleton import PARTS
from .fusion_camera_controller import FusionCameraController
from .fusion_vision_controller import FusionVisionController
from .fusion_calibration_controller import FusionCalibrationController
from .fusion_recording_controller import FusionRecordingController


class FusionWindow(BodyWindow):
    def __init__(self, source: object, settings_path: Path,
                 assignments: dict[str, str], axis_mappings: dict,
                 *, camera_index: int = 0, backend: str = "rtmw3d",
                 camera_source: str = "webcam", orbbec_python: Path = DEFAULT_PYTHON,
                 rgbd_fps: float = 10., rgbd_warmup: float = 2.,
                 rgbd_pose: bool = False, pose_backend: str = 'rtmw',
                 yolo_model: Path = DEFAULT_YOLO, pose_imgsz: int = 640,
                 algorithm_modes: dict[str, str] | None = None,
                 model_path: Path | None = None,
                 calibration_candidates: list[dict] | None = None,
                 comparison_shared_calibrations: list[dict] | None = None,
                 manual_mapping_source: str = "当前配置") -> None:
        super().__init__(source, settings_path, assignments, axis_mappings,
                         algorithm_modes)
        if isinstance(source, FusionSessionReplay) and source.camera_calibration is not None:
            camera_source = "orbbec"
        self.setWindowTitle("WT901 + 单目相机 · 同步采集基线")
        self.camera_source = camera_source
        if camera_source == "orbbec":
            self.setWindowTitle("WT901 + Gemini 336L RGB-D · 同步采集（未融合）")
        self.orbbec_python = Path(orbbec_python)
        self.rgbd_fps, self.rgbd_warmup = rgbd_fps, rgbd_warmup
        self.last_rgbd_observation = None
        self.pose_worker = None
        self.person_target=PersonTarget()
        self._person_candidates=[]
        self._person_frame_ns=None
        self._person_preview_frame_ns=None
        import uuid
        self.vision_config=Pose2DConfig(pose_backend,Path(yolo_model),pose_imgsz)
        self.vision_metadata=self.vision_config.metadata()
        self.vision_epoch=uuid.uuid4().hex
        self._vision_changed=False
        if isinstance(source,FusionSessionReplay) and source.manifest.get('vision_config'):
            saved=source.manifest['vision_config']
            model=Path(saved['weights'][0]['path']) if saved['backend']=='yolo26' else Path(yolo_model)
            self.vision_config=Pose2DConfig(saved['backend'],model,saved.get('imgsz') or 640)
            self.vision_metadata=saved;self.vision_epoch=source.manifest.get('vision_epoch')
        self.pose_dialog = None
        self.camera_calibration_dialog = None
        self._heading_calibration_dialog = None
        self._visual_wearing_dialog = None
        self._reuse_calibration_dialog = None
        self._combined_calibration_dialog = None
        self.live_fusion = LiveRGBDFusion(self.engine)
        self._fusion_events_file = None
        self._live_camera_epoch = None
        self._person_restore_vision_signature = None
        self.last_pose_estimate = None
        self.palm_visual_dialog = None
        self._bridge_skipped_start = 0
        self.rgbd_timer = QtCore.QTimer(self)
        self.rgbd_timer.setInterval(20)
        self.rgbd_timer.timeout.connect(self._poll_rgbd)
        self.candidate_engine: PoseEngine | None = None
        self.candidate_actors = []
        self.candidate_axis_actors = []
        self.live_compare_actors = []
        self.live_compare_axis_actors = []
        self._live_compare_enabled = False
        self._single_camera = None
        self.visual_wearing_candidates = {c['part']: c for c in calibration_candidates or []
                                          if c.get('method') == 'visual_bone_v1'}
        self.visual_direction_actors = []
        self.comparison_neutral_devices = {c['device_id'] for c in
            list(calibration_candidates or []) + list(comparison_shared_calibrations or [])}
        if isinstance(source, FusionSessionReplay):
            for key, matrix in source.manifest.get('continuous_basis_at_start', {}).items():
                self.engine.set_continuous_basis(key, matrix)
        if calibration_candidates:
            if not self.replay_mode:
                raise ValueError("候选标定只能用于回放预览")
            self.candidate_engine = PoseEngine(assignments, axis_mappings,
                                               algorithm_modes)
            if isinstance(source, FusionSessionReplay):
                for key, matrix in source.manifest.get('continuous_basis_at_start', {}).items():
                    self.candidate_engine.set_continuous_basis(key, matrix)
            for shared in comparison_shared_calibrations or []:
                key = shared["part"]
                if self.engine.assignments.get(key) != shared["device_id"]:
                    raise ValueError(f"{key} 的共同标定设备与回放分配不一致")
                for engine in (self.engine, self.candidate_engine):
                    engine.apply_calibration(
                        key, shared["device_id"], shared["basis_matrix"],
                        shared["neutral_matrix"],
                        paired_neutrals=shared.get("neutral_matrices_by_mode"))
            for candidate in calibration_candidates:
                key = candidate["part"]
                if self.candidate_engine.assignments.get(key) != candidate["device_id"]:
                    raise ValueError(f"{key} 的候选设备与回放分配不一致")
                if "neutral_matrix" in candidate:
                    neutral = np.asarray(candidate["neutral_matrix"], dtype=float)
                    self.candidate_engine.apply_calibration(
                        key, candidate["device_id"], candidate["basis_matrix"], neutral,
                        paired_neutrals=candidate.get("neutral_matrices_by_mode"))
                    # Both avatars must use the same neutral hold; otherwise
                    # differences in the first (possibly moving) packet bias
                    # the visual A/B comparison.
                    for mode, matrix in candidate.get(
                            "neutral_matrices_by_mode", {}).items():
                        self.engine.set_neutral_for_mode(
                            candidate["device_id"], mode, matrix)
                    active_mode = self.engine.algorithm_modes[key]
                    self.engine.set_neutral_for_mode(
                        candidate["device_id"],
                        active_mode,
                        candidate.get("neutral_matrices_by_mode", {}).get(
                            active_mode, neutral))
                else:
                    self.candidate_engine.set_continuous_basis(key, candidate["basis_matrix"])
            self._install_comparison_view(manual_mapping_source)
            self.setWindowTitle("WT901 姿态映射对比 · 同一数据 / 双模型")
            for combo in self.combos.values():
                combo.setEnabled(False)
            for button in self.mapping_buttons.values():
                button.setEnabled(False)
            for combo in self.algorithm_combos.values():
                combo.setEnabled(False)
        self.record_button.hide()  # The legacy CSV cannot contain camera frames.
        self.camera_index = camera_index
        self.backend = backend
        self.model_path = model_path or Path()
        self.camera_worker: CameraWorker | None = None
        self.session: FusionSessionRecorder | None = None
        # Public state stays here for calibration dialogs and replay.
        self._camera_controller = FusionCameraController(
            self, camera_factory=lambda *a, **k: CameraWorker(*a, **k),
            orbbec_type=OrbbecWorker,
            pose_factory=lambda *a, **k: RGBDPoseWorker(*a, **k))
        self._vision_controller = FusionVisionController(self)
        self._calibration_controller = FusionCalibrationController(self)
        self._recording_controller = FusionRecordingController(
            self, recorder_factory=lambda *a, **k: FusionSessionRecorder(*a, **k),
            orbbec_type=OrbbecWorker,
            warn=lambda title, message: QtWidgets.QMessageBox.warning(self, title, message))
        self.camera_count = 0
        self.visible_points = 0
        self._camera_failed = False
        self._record_start_deadline: float | None = None
        self._record_stop_deadline: float | None = None
        self.record_schedule_timer = QtCore.QTimer(self)
        self.record_schedule_timer.setInterval(100)
        self.record_schedule_timer.timeout.connect(self._update_record_schedule)

        group = QtWidgets.QGroupBox("相机 + IMU（RGB-D可选低频航向纠偏）")
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        camera_row = QtWidgets.QHBoxLayout()
        self.camera_source_combo = QtWidgets.QComboBox()
        self.camera_source_combo.addItem("普通摄像头 + 姿态检测", "webcam")
        self.camera_source_combo.addItem("Gemini 336L RGB-D", "orbbec")
        self.camera_source_combo.setCurrentIndex(1 if camera_source == "orbbec" else 0)
        self.camera_source_combo.currentIndexChanged.connect(self._change_camera_source)
        camera_row.addWidget(self.camera_source_combo)
        self.rgbd_rate = QtWidgets.QSpinBox()
        self.rgbd_rate.setRange(1, 30)
        self.rgbd_rate.setValue(round(rgbd_fps))
        self.rgbd_rate.setSuffix(" 帧/秒")
        self.rgbd_rate.setToolTip("RGB-D 输出/录制帧率上限；源采流使用默认 profile，实际输出可能略低。停止相机后更改。")
        camera_row.addWidget(self.rgbd_rate)
        group_layout.addLayout(camera_row)
        self.depth_preview = QtWidgets.QCheckBox("预览配准深度（伪彩色仅用于显示）")
        group_layout.addWidget(self.depth_preview)
        pose_row = QtWidgets.QHBoxLayout()
        self.rgbd_pose_enabled = QtWidgets.QCheckBox("三维观测：二维关键点＋真实深度")
        self.rgbd_pose_enabled.setEnabled(camera_source == "orbbec" and not self.replay_mode)
        self.rgbd_pose_enabled.toggled.connect(self._set_rgbd_pose)
        pose_row.addWidget(self.rgbd_pose_enabled)
        self.pose_details = QtWidgets.QPushButton("三维 / 可靠性详情")
        self.pose_details.clicked.connect(self._show_pose_details)
        pose_row.addWidget(self.pose_details)
        group_layout.addLayout(pose_row)
        model_row=QtWidgets.QHBoxLayout()
        model_row.addWidget(QtWidgets.QLabel('二维模型'))
        self.pose_backend_combo=QtWidgets.QComboBox()
        self.pose_backend_combo.addItem('RTMW（现有）','rtmw');self.pose_backend_combo.addItem('YOLO26 Pose','yolo26')
        self.pose_backend_combo.setCurrentIndex(self.pose_backend_combo.findData(self.vision_config.backend))
        model_row.addWidget(self.pose_backend_combo)
        self.pose_imgsz_combo=QtWidgets.QComboBox()
        for size in (640,960):self.pose_imgsz_combo.addItem(f'{size} 像素',size)
        self.pose_imgsz_combo.setCurrentIndex(self.pose_imgsz_combo.findData(self.vision_config.imgsz))
        model_row.addWidget(self.pose_imgsz_combo)
        self.pose_model_button=QtWidgets.QPushButton('YOLO权重…')
        self.pose_model_button.setToolTip(str(yolo_model));model_row.addWidget(self.pose_model_button)
        group_layout.addLayout(model_row)
        self.palm_visual_button=QtWidgets.QPushButton('左手视觉 C0…（与手动估计对比）')
        self.palm_visual_button.setEnabled(camera_source=='orbbec' and not self.replay_mode)
        self.palm_visual_button.clicked.connect(self._show_palm_visual)
        group_layout.addWidget(self.palm_visual_button)
        self.pose_backend_combo.currentIndexChanged.connect(self._change_pose_config)
        self.pose_imgsz_combo.currentIndexChanged.connect(self._change_pose_config)
        self.pose_model_button.clicked.connect(self._choose_pose_model)
        self.pose_status = QtWidgets.QLabel("视觉观测关闭；原人体模型仍由 IMU 驱动")
        self.pose_status.setWordWrap(True)
        self.pose_status.setMinimumHeight(52)
        group_layout.addWidget(self.pose_status)
        person_row=QtWidgets.QHBoxLayout()
        self.person_combo=QtWidgets.QComboBox()
        self.person_combo.setToolTip('编号仅用于当前帧；确认选择的人佩戴当前分配的 IMU。')
        person_row.addWidget(self.person_combo,1)
        self.person_select=QtWidgets.QPushButton('锁定所选人物')
        self.person_select.clicked.connect(lambda:self._select_person(self.person_combo.currentData()))
        self.person_combo.activated.connect(lambda _index:self._select_person(self.person_combo.currentData()))
        person_row.addWidget(self.person_select)
        group_layout.addLayout(person_row)
        person_actions=QtWidgets.QHBoxLayout()
        self.person_auto=QtWidgets.QPushButton('单人自动（重新锁定）')
        self.person_auto.clicked.connect(lambda:self._apply_person_target(None,'auto'))
        person_actions.addWidget(self.person_auto)
        self.person_cancel=QtWidgets.QPushButton('取消选择 / 暂停视觉')
        self.person_cancel.clicked.connect(lambda:self._apply_person_target(None,'paused'))
        person_actions.addWidget(self.person_cancel)
        group_layout.addLayout(person_actions)
        self.person_restore_ack=QtWidgets.QCheckBox('确认所选为原标定者，相机和设备绑带未移动')
        group_layout.addWidget(self.person_restore_ack)
        self.person_restore=QtWidgets.QPushButton('确认原标定者 · 保留标定并恢复纠偏')
        self.person_restore.clicked.connect(self._restore_original_person)
        self.person_restore_ack.toggled.connect(self._sync_pose_controls)
        group_layout.addWidget(self.person_restore)
        self.person_status=QtWidgets.QLabel('人物：单人首次自动锁定；多人请点击彩色画面中的人物框。切换目标需重新相机标定。')
        self.person_status.setWordWrap(True)
        group_layout.addWidget(self.person_status)
        self._sync_pose_controls()
        limit_row=QtWidgets.QHBoxLayout()
        limit_row.addWidget(QtWidgets.QLabel('累计纠偏范围 ±'))
        self.heading_limit=QtWidgets.QSpinBox();self.heading_limit.setRange(5,90)
        self.heading_limit.setSingleStep(5);self.heading_limit.setValue(30);self.heading_limit.setSuffix(' °')
        self.heading_limit.setToolTip('四个臂段共用累计范围；默认±30°，可试±60°。创新门限30°和纠偏速率3°/s不变；更大范围可能追随错误视觉。仅本次运行有效。')
        limit_row.addWidget(self.heading_limit)
        self.heading_unlimited=QtWidgets.QCheckBox('不限累计角度（实验）')
        self.heading_unlimited.setToolTip('只取消累计上限；创新门限30°、限速3°/s及观测质量检查保留。可能累计错误视觉修正。默认关闭，重启恢复±30°。')
        self.heading_unlimited.toggled.connect(lambda checked:self.heading_limit.setEnabled(not checked and self._pose_change_error() is None))
        limit_row.addWidget(self.heading_unlimited)
        self.heading_limit_apply=QtWidgets.QPushButton('应用范围')
        self.heading_limit_apply.clicked.connect(self._apply_heading_limit);limit_row.addWidget(self.heading_limit_apply)
        group_layout.addLayout(limit_row)
        if isinstance(source,FusionSessionReplay):
            recorded_limit=source.manifest.get('heading_max_offset_deg',30)
            self.heading_unlimited.setChecked(recorded_limit is None)
            if recorded_limit is not None:self.heading_limit.setValue(round(recorded_limit))
            self.live_fusion.set_max_offset(recorded_limit)
        self._sync_pose_controls()
        self.camera_preview = CameraPreview()
        self.camera_preview.image_clicked.connect(self._click_person)
        self.camera_preview.setMinimumWidth(240)
        self.camera_preview.setFixedHeight(120)
        self.camera_preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                         QtWidgets.QSizePolicy.Policy.Fixed)
        self.camera_preview.setStyleSheet("background: #191b20; color: white")
        group_layout.addWidget(self.camera_preview)
        self._camera_preview_layout = group_layout
        self.camera_preview_window = None
        self.camera_preview_placeholder = QtWidgets.QLabel("相机画面正在独立窗口显示，可拖到另一台显示器。")
        self.camera_preview_placeholder.setWordWrap(True)
        self.camera_preview_placeholder.hide()
        group_layout.addWidget(self.camera_preview_placeholder)
        self.detach_camera_button = QtWidgets.QPushButton("相机画面独立窗口 ↗")
        self.detach_camera_button.setToolTip("仅移动显示画面，不重新打开相机；支持彩色、深度和回放画面。")
        self.detach_camera_button.clicked.connect(self._toggle_camera_preview_window)
        group_layout.addWidget(self.detach_camera_button)
        self.camera_calibration_button = QtWidgets.QPushButton("相机标定 · 引导录制")
        self.camera_calibration_button.setEnabled(not self.replay_mode)
        self.camera_calibration_button.clicked.connect(self._show_camera_calibration)
        group_layout.addWidget(self.camera_calibration_button)
        self.visual_wearing_button = QtWidgets.QPushButton("视觉辅助穿戴标定 · 双臂 / 单臂（实验）")
        self.visual_wearing_button.setEnabled(not self.replay_mode and camera_source == 'orbbec')
        self.visual_wearing_button.clicked.connect(self._show_visual_wearing)
        group_layout.addWidget(self.visual_wearing_button)
        self.combined_calibration_button = QtWidgets.QPushButton('双臂一体标定 · 穿戴＋相机（一次动作）')
        self.combined_calibration_button.setEnabled(not self.replay_mode and camera_source == 'orbbec')
        self.combined_calibration_button.clicked.connect(self._show_combined_calibration)
        group_layout.addWidget(self.combined_calibration_button)
        self.reuse_calibration_button = QtWidgets.QPushButton('历史标定 · 保存 / 快速恢复')
        self.reuse_calibration_button.setEnabled(not self.replay_mode and camera_source == 'orbbec')
        self.reuse_calibration_button.clicked.connect(self._show_calibration_reuse)
        group_layout.addWidget(self.reuse_calibration_button)
        self.freeze_fusion=QtWidgets.QCheckBox('冻结视觉纠偏（保留修正，继续IMU传播）')
        group_layout.addWidget(self.freeze_fusion)
        self.live_compare_checkbox = QtWidgets.QCheckBox('双人体实时对比：陀螺＋加计 vs 陀螺＋加计＋RGB-D')
        self.live_compare_checkbox.setEnabled(not self.replay_mode)
        self.live_compare_checkbox.setToolTip('仅改变显示；四个臂段使用同一数据、同一穿戴映射、各自中立参考。'
            '要持续纠偏，请先应用相机标定并将对应部位选为RGB-D融合模式。回放请使用原对比窗口。')
        self.live_compare_checkbox.toggled.connect(self._toggle_live_comparison)
        group_layout.addWidget(self.live_compare_checkbox)
        self.live_fusion_status=QtWidgets.QLabel('实时纠偏未标定；不影响当前算法')
        self.live_fusion_status.setWordWrap(True);group_layout.addWidget(self.live_fusion_status)
        invalidate=QtWidgets.QPushButton('相机/穿戴变化：停用本次相机标定')
        invalidate.clicked.connect(lambda:self.live_fusion.invalidate('相机/穿戴改变，需重标定'))
        group_layout.addWidget(invalidate)
        self.camera_status = QtWidgets.QLabel(f"识别后端：{backend}；等待启动")
        self.camera_status.setWordWrap(True)
        group_layout.addWidget(self.camera_status)
        controls = QtWidgets.QHBoxLayout()
        self.start_camera_button = QtWidgets.QPushButton("启动相机")
        self.start_camera_button.clicked.connect(self.start_camera)
        controls.addWidget(self.start_camera_button)
        self.stop_camera_button = QtWidgets.QPushButton("停止相机")
        self.stop_camera_button.clicked.connect(self.stop_camera)
        self.stop_camera_button.setEnabled(False)
        controls.addWidget(self.stop_camera_button)
        self.session_button = QtWidgets.QPushButton("开始联合录制")
        self.session_button.clicked.connect(self.toggle_session)
        self.session_button.setEnabled(not self.replay_mode)
        controls.addWidget(self.session_button)
        group_layout.addLayout(controls)
        timed_controls = QtWidgets.QHBoxLayout()
        timed_controls.addWidget(QtWidgets.QLabel("延迟"))
        self.record_delay = QtWidgets.QSpinBox()
        self.record_delay.setRange(0, 120)
        self.record_delay.setValue(10)
        self.record_delay.setSuffix(" 秒")
        timed_controls.addWidget(self.record_delay)
        timed_controls.addWidget(QtWidgets.QLabel("录制"))
        self.record_duration = QtWidgets.QSpinBox()
        self.record_duration.setRange(5, 600)
        self.record_duration.setValue(30)
        self.record_duration.setSuffix(" 秒")
        timed_controls.addWidget(self.record_duration)
        self.timed_session_button = QtWidgets.QPushButton("定时录制")
        self.timed_session_button.clicked.connect(self._start_timed_session)
        self.timed_session_button.setEnabled(not self.replay_mode)
        timed_controls.addWidget(self.timed_session_button)
        self.cancel_session_button = QtWidgets.QPushButton("取消预约")
        self.cancel_session_button.clicked.connect(self._cancel_record_schedule)
        self.cancel_session_button.setEnabled(False)
        timed_controls.addWidget(self.cancel_session_button)
        group_layout.addLayout(timed_controls)
        self.record_schedule_status = QtWidgets.QLabel(
            "定时录制未启动；手动或定时结束时均剔除末尾 3 秒")
        group_layout.addWidget(self.record_schedule_status)
        note = QtWidgets.QLabel(
            "录制包含未绘线的相机帧、原始 2D 关键点及置信度、WT901 数据和主机采集时间。"
            "当前仅用于数据检查；视觉校正将在后续阶段加入。")
        note.setWordWrap(True)
        group.setToolTip(note.text())
        main_layout = self.centralWidget().layout()
        panel = main_layout.itemAt(1).widget()
        # This window has camera and recording controls in addition to the
        # original device panel; give them room without changing run.py.
        panel.setMinimumWidth(520)
        panel.setMaximumWidth(740)
        main_layout.setStretch(0, 3)
        main_layout.setStretch(1, 2)
        panel.layout().insertWidget(2, group)
        # The whole control panel must scroll on shorter screens: forcing the
        # camera's minimum height alone would overlap downstream controls.
        panel.layout().setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        main_layout.removeWidget(panel)
        panel.setMinimumWidth(0)
        panel.setMaximumWidth(16777215)
        control_scroll = QtWidgets.QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setMinimumWidth(520)
        control_scroll.setMaximumWidth(740)
        control_scroll.setWidget(panel)
        main_layout.addWidget(control_scroll, 2)
        self.control_scroll = control_scroll
        if not self.replay_mode:
            self._install_live_comparison_header()
        if isinstance(source, FusionSessionReplay):
            import json
            manifest=json.loads((source.path/'manifest.json').read_text())
            for key,value in manifest.get('continuous_basis_at_start',{}).items():self.engine.set_continuous_basis(key,value)
            self.rgbd_pose_enabled.setEnabled(source.has_vision)
            self.rgbd_pose_enabled.setChecked(source.has_vision)
            if not source.has_vision:
                self.pose_status.setText("此录制没有三维视觉结果；可离线分析原始 RGB-D")
            self.camera_source_combo.setEnabled(False)
            self.rgbd_rate.setEnabled(False)
            self.start_camera_button.setEnabled(False)
            self.camera_status.setText(
                (f"左：你的手动轴映射（{manual_mapping_source}）；右：我的候选标定。"
                 "两者使用同一批 IMU 包；"
                 "相机不驱动模型。\n" if calibration_candidates else "")
                + f"正在回放：{source.path}")
        else:
            self.start_camera()
            self.rgbd_pose_enabled.setChecked(rgbd_pose and camera_source == "orbbec")

    def _install_comparison_view(self, manual_mapping_source: str) -> None:
        """Keep one VTK viewport/camera and place both avatars side by side."""
        layout = self.centralWidget().layout()
        layout.takeAt(0)
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)
        labels = QtWidgets.QHBoxLayout()
        baseline_caption = (f"对比基线 · {manual_mapping_source}" if self.visual_wearing_candidates else
                            f"你的映射 · 手动 XYZ（{manual_mapping_source}）")
        for caption, color in ((baseline_caption, "#76D7FF"),
                               ("视觉辅助候选（仅回放）" if self.visual_wearing_candidates else
                                "我的候选 · 连续标定（仅回放）", "#FFBD72")):
            label = QtWidgets.QLabel(caption)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
                f"background: #242936; color: {color}; font-weight: bold;"
                "padding: 6px; font-size: 13px;")
            labels.addWidget(label, 1)
        container_layout.addLayout(labels)
        if self.visual_wearing_candidates:
            failed = any(r.get('status') not in ('通过', '通过，但需人工复核') for r in self.visual_wearing_candidates.values())
            review = any(r.get('status') == '通过，但需人工复核' for r in self.visual_wearing_candidates.values())
            note = QtWidgets.QLabel(('候选含未通过结果，禁止实时应用。' if failed else
                                    '候选需人工复核，不能仅凭方向误差认定轴向动作正确。' if review else '') +
                '紫色箭头：RGB-D骨段方向，经中立归一化后显示；锚点取各模型关节。不是原始三维位置或GT；长轴扭转需另行检查。')
            note.setWordWrap(True)
            container_layout.addWidget(note)
        container_layout.addWidget(self.plotter.interactor, 1)
        layout.insertWidget(0, container, 3)
        triad = _axis_triad(0.105)
        for part in PARTS:
            mesh = pv.read(MODEL_DIR / part.mesh)
            mesh.points *= 0.001
            self.candidate_actors.append(self.plotter.add_mesh(
                mesh, color=part.color, smooth_shading=False, show_edges=False))
            axis_actor = self.plotter.add_mesh(
                triad, scalars="axis_rgb", rgb=True, lighting=False,
                show_scalar_bar=False, pickable=False)
            axis_actor.SetVisibility(self.axes_checkbox.isChecked())
            self.candidate_axis_actors.append(axis_actor)
        self._render_pose()
        self._front_view()

    def _install_live_comparison_header(self):
        layout = self.centralWidget().layout()
        layout.takeAt(0)
        container = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        self.live_compare_header = QtWidgets.QWidget()
        header = QtWidgets.QHBoxLayout(self.live_compare_header)
        header.setContentsMargins(0, 0, 0, 0)
        self.live_compare_labels = []
        for color in ('#76D7FF', '#FFBD72'):
            label = QtWidgets.QLabel()
            label.setWordWrap(True)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f'background:#242936;color:{color};padding:6px;font-weight:bold;')
            header.addWidget(label, 1)
            self.live_compare_labels.append(label)
        column.addWidget(self.live_compare_header)
        self.live_compare_header.hide()
        column.addWidget(self.plotter.interactor, 1)
        layout.insertWidget(0, container, 3)

    def _toggle_live_comparison(self, enabled):
        # Replay candidate comparison is a different feature, never mix them.
        if self.replay_mode:
            return
        if enabled and not self.live_compare_actors:
            triad = _axis_triad(.105)
            for part, original in zip(PARTS, self.actors):
                self.live_compare_actors.append(self.plotter.add_mesh(
                    original.mapper.dataset, color=part.color, smooth_shading=False,
                    show_edges=False, render=False))
                self.live_compare_axis_actors.append(self.plotter.add_mesh(
                    triad, scalars='axis_rgb', rgb=True, lighting=False,
                    show_scalar_bar=False, pickable=False, render=False))
        if enabled:
            self._single_camera = self.plotter.camera.copy()
        self._live_compare_enabled = enabled
        self.live_compare_header.setVisible(enabled)
        for actor in self.live_compare_actors:
            actor.SetVisibility(enabled)
        for actor in self.live_compare_axis_actors:
            actor.SetVisibility(enabled and self.axes_checkbox.isChecked())
        if enabled:
            self._front_view()
        elif self._single_camera is not None:
            self.plotter.camera = self._single_camera
        self._render_pose()

    def _update_live_comparison_status(self):
        if not self._live_compare_enabled:
            return
        names = {'left_upper_arm':'左上臂', 'left_forearm':'左前臂',
                 'right_upper_arm':'右上臂', 'right_forearm':'右前臂'}
        raw, fused = [], []
        for part in ARM_SEGMENTS:
            device = self.engine.assignments.get(part)
            if not device:
                raw.append(names[part]+'未绑定'); fused.append(names[part]+'未绑定')
                continue
            if not self.engine.is_live(device):
                raw.append(names[part]+'数据过期'); fused.append(names[part]+'数据过期')
            if self.engine.current_for(device, GYRO_TILT) is None or self.engine.neutral_for(device, GYRO_TILT) is None:
                raw.append(names[part]+'姿态未就绪')
            elif not self.engine._tilt_filters[device].bias_ready:
                raw.append(names[part]+'静止学习未完成')
            if self.engine.current_for(device, GYRO_RGBD) is None or self.engine.neutral_for(device, GYRO_RGBD) is None:
                fused.append(names[part]+'姿态未就绪')
            elif device not in self.engine.fusion_ready:
                fused.append(names[part]+'未标定/已停用（保留缓存）')
            elif self.engine.algorithm_modes[part] != GYRO_RGBD:
                fused.append(names[part]+'未选融合模式，纠偏待机')
            else:
                state = self.live_fusion.states.get(part, {})
                fused.append(names[part]+f" {state.get('offset_deg', 0):+.1f}° · "+state.get('reason', '等待视觉'))
        if self.freeze_fusion.isChecked() or not self.rgbd_pose_enabled.isChecked():
            fused.insert(0, '视觉冻结/关闭：保留已有修正')
        elif self.last_pose_estimate is None or time.monotonic_ns()-self.last_pose_estimate.captured_mono_ns > 500_000_000:
            fused.insert(0, '视觉未就绪/过期：保留已有修正')
        self.live_compare_labels[0].setText('左 · 陀螺＋加计（不使用视觉修正）\n'+('；'.join(raw) or '四臂段实时传播'))
        self.live_compare_labels[1].setText('右 · 陀螺＋加计＋RGB-D（非GT）\n'+'；'.join(fused))

    def _render_pose(self) -> None:
        if getattr(self, '_live_compare_enabled', False):
            for actors, axes, mode, y in (
                    (self.actors, self.axis_actors, GYRO_TILT, -1.),
                    (self.live_compare_actors, self.live_compare_axis_actors, GYRO_RGBD, 1.)):
                offset = np.eye(4); offset[1, 3] = y
                matrices = self.engine.transforms(mode_overrides={p:mode for p in ARM_SEGMENTS})
                for actor, axis, matrix in zip(actors, axes, matrices):
                    actor.user_matrix = offset @ matrix
                    axis.user_matrix = offset @ matrix
            self._update_live_comparison_status()
            self.plotter.render()
            self._render_dirty = False
            self._last_render = time.monotonic()
            return
        if getattr(self, "candidate_engine", None) is None:
            super()._render_pose()
            return
        spacing = 1.0 if self.visual_wearing_candidates else .8
        for actors, axis_actors, engine, offset_y in (
                (self.actors, self.axis_actors, self.engine, -spacing),
                (self.candidate_actors, self.candidate_axis_actors,
                 self.candidate_engine, +spacing)):
            offset = np.eye(4)
            offset[1, 3] = offset_y
            for actor, axis_actor, matrix in zip(
                    actors, axis_actors, engine.transforms()):
                positioned = offset @ matrix
                actor.user_matrix = positioned
                axis_actor.user_matrix = positioned
        if self.visual_wearing_candidates:
            self._render_visual_direction_targets()
        self.plotter.render()
        self._render_dirty = False
        self._last_render = time.monotonic()

    def _front_view(self):
        if getattr(self, '_live_compare_enabled', False):
            self.plotter.camera_position = [(4., 0., .15), (0., 0., .15), (0., 0., 1.)]
            self.plotter.reset_camera(bounds=(-1., 1., -2.2, 2.2, -1., 1.3))
            self.plotter.camera.zoom(1.45)
        elif getattr(self, 'visual_wearing_candidates', None) and self.candidate_engine is not None:
            # Reserve the full bilateral T-pose envelope, not only the small
            # neutral-pose bounds at startup. R restores this fixed view.
            self.plotter.camera_position = [(4., 0., .15), (0., 0., .15), (0., 0., 1.)]
            self.plotter.reset_camera(bounds=(-1., 1., -2., 2., -1., 1.3))
            self.plotter.camera.zoom(1.2)
        else:
            super()._front_view()

    def _render_visual_direction_targets(self):
        """Magenta directions are neutral-normalized observations, NOT raw GT."""
        from .rgbd_fusion import visual_direction
        from .skeleton import PART_BY_KEY
        for actor in self.visual_direction_actors:
            self.plotter.remove_actor(actor, render=False)
        self.visual_direction_actors = []
        estimate = getattr(self, 'last_pose_estimate', None)
        if estimate is None:
            return
        row = estimate.to_record()
        # Use the recorded source-frame timestamp, not a live wall clock.
        stamp = estimate.captured_mono_ns
        for engine, offset_y in ((self.engine, -1.), (self.candidate_engine, 1.)):
            transforms = engine.transforms()
            for part, result in self.visual_wearing_candidates.items():
                device = result['device_id']
                # Hide an old observation if the replay IMU has advanced >0.5s.
                current_stamp = getattr(self, '_last_replay_stamp', stamp)
                if abs(current_stamp - stamp) > 500_000_000:
                    continue
                direction, _ = visual_direction(row, part, result['bone_length_m'])
                if direction is None:
                    continue
                direction = (np.array(result['L_neutral_normalization']) @
                             np.array(result['K_model_from_camera']) @ direction)
                index = PART_BY_KEY[part]
                start = transforms[index][:3, 3].copy()
                start[1] += offset_y
                length = .26349 if part.endswith('_upper_arm') else .2724
                actor = self.plotter.add_mesh(pv.Arrow(start=start, direction=direction, scale=length),
                                             color='#ef70db', opacity=.75, pickable=False,
                                             render=False)
                self.visual_direction_actors.append(actor)

    def _toggle_axes(self, visible: bool) -> None:
        super()._toggle_axes(visible)
        for actor in getattr(self, "candidate_axis_actors", ()):
            actor.SetVisibility(visible)
        for actor in getattr(self, 'live_compare_axis_actors', ()):
            actor.SetVisibility(visible and self._live_compare_enabled)
        self._render_dirty = True

    def _zero(self, key: str) -> None:
        super()._zero(key)
        if self.candidate_engine is not None:
            self.candidate_engine.zero(key)

    def _zero_all(self) -> None:
        super()._zero_all()
        if self.candidate_engine is not None:
            self.candidate_engine.zero()

    def start_camera(self) -> None:
        return self._camera_controller.start_camera()

    def _change_camera_source(self):
        return self._camera_controller._change_camera_source()

    def _set_rgbd_pose(self, enabled):
        return self._camera_controller._set_rgbd_pose(enabled)

    def _pose_change_error(self):
        return self._vision_controller._pose_change_error()

    def _sync_pose_controls(self):
        if not hasattr(self,'pose_backend_combo'):return
        self.pose_backend_combo.blockSignals(True);self.pose_imgsz_combo.blockSignals(True)
        self.pose_backend_combo.setCurrentIndex(self.pose_backend_combo.findData(self.vision_config.backend))
        self.pose_imgsz_combo.setCurrentIndex(self.pose_imgsz_combo.findData(self.vision_config.imgsz))
        self.pose_backend_combo.blockSignals(False);self.pose_imgsz_combo.blockSignals(False)
        permitted=self._pose_change_error() is None
        self.pose_backend_combo.setEnabled(permitted)
        self.pose_imgsz_combo.setEnabled(permitted and self.vision_config.backend=='yolo26')
        self.pose_model_button.setEnabled(permitted and self.vision_config.backend=='yolo26')
        self.pose_model_button.setToolTip(str(self.vision_config.model))
        if hasattr(self,'person_select'):
            enabled=permitted and self.camera_source=='orbbec'
            self.person_combo.setEnabled(enabled)
            self.person_select.setEnabled(enabled and self.rgbd_pose_enabled.isChecked() and bool(self._person_candidates))
            self.person_auto.setEnabled(enabled);self.person_cancel.setEnabled(enabled)
            if hasattr(self,'person_restore'):
                error=self._person_restore_error()
                self.person_restore.setEnabled(error is None and self.person_restore_ack.isChecked())
                self.person_restore.setToolTip(error or '仅恢复已确认的原穿戴者；不重置补偿角，不自动切换算法或启动机器人')
        if hasattr(self,'heading_limit_apply'):
            self.heading_limit.setEnabled(permitted and not self.heading_unlimited.isChecked())
            self.heading_unlimited.setEnabled(permitted);self.heading_limit_apply.setEnabled(permitted)

    def _restore_heading_limit_controls(self):
        value=self.live_fusion.max_offset_deg
        self.heading_unlimited.setChecked(value is None)
        if value is not None:self.heading_limit.setValue(round(value))

    def _apply_heading_limit(self):
        error=self._pose_change_error()
        if error:
            self._restore_heading_limit_controls()
            self.pose_status.setText(error);return
        try:self.live_fusion.set_max_offset(None if self.heading_unlimited.isChecked() else self.heading_limit.value())
        except ValueError as exc:
            self._restore_heading_limit_controls()
            QtWidgets.QMessageBox.warning(self,'范围未修改',str(exc));return
        self._update_live_fusion_status()
        label='不限累计角度（实验）' if self.live_fusion.max_offset_deg is None else f'±{self.live_fusion.max_offset_deg:g}°'
        self.statusBar().showMessage(f'累计纠偏范围已改为{label}；创新门限30°、限速3°/s不变',10000)

    def _choose_pose_model(self):
        error=self._pose_change_error()
        if error:self.pose_status.setText(error);return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择已下载的YOLO26 Pose权重',str(self.vision_config.model.parent),'模型 (*.pt)')
        if path:self._change_pose_config(model=Path(path))

    def _select_person(self,item):
        if item is None:return
        age=time.monotonic_ns()-item['captured_mono_ns']
        if not self.rgbd_pose_enabled.isChecked() or not 0<=age<=1_000_000_000:
            self.person_status.setText('人物框已过期，请等待新的检测画面再选择');return
        self._apply_person_target(item,'manual')

    def _click_person(self,x,y):
        if self.depth_preview.isChecked():
            self.person_status.setText('请切回彩色画面后点击人物框，或使用人物下拉列表');return
        if not self.rgbd_pose_enabled.isChecked() or self._person_preview_frame_ns!=self._person_frame_ns:
            self.person_status.setText('等待带人物框的当前彩色检测画面再选择');return
        hits=[p for p in self._person_candidates if p['box'][0]<=x<=p['box'][2] and p['box'][1]<=y<=p['box'][3]]
        if len(hits)==1:self._select_person(hits[0])
        elif len(hits)>1:self.person_status.setText('点击处有重叠人物框，请使用下拉列表明确选择；有歧义时仍会暂停跟踪')

    def _apply_person_target(self,anchor,mode):
        return self._vision_controller._apply_person_target(anchor, mode)

    def _person_restore_error(self):
        return self._vision_controller._person_restore_error()

    def _restore_original_person(self):
        return self._vision_controller._restore_original_person()

    def _update_person_candidates(self,estimate):
        info=estimate.diagnostics
        self._person_candidates=info.get('person_candidates',[])
        self._person_frame_ns=estimate.captured_mono_ns
        chosen=info.get('selected_index')
        # Freeze entries while the popup is open. Each entry carries its source
        # frame/box, not an index to look up again in a reordered future frame.
        if not self.person_combo.view().isVisible():
            self.person_combo.blockSignals(True);self.person_combo.clear()
            for person in self._person_candidates:
                depth=person.get('depth_m')
                suffix=f' · 表面深度 {depth:.2f} m' if depth is not None else ' · 深度不可用'
                self.person_combo.addItem(f"本帧人物 #{person['index']+1}"+suffix,person.copy())
                if person['index']==chosen:self.person_combo.setCurrentIndex(self.person_combo.count()-1)
            self.person_combo.blockSignals(False)
        self.person_status.setText('人物：'+info.get('target_state','历史记录未保存人物选择')+
            (f" · 当前框 #{chosen+1}" if chosen is not None else '')+
            (' · OSNet 恢复已启用' if info.get('reid_enabled') else ' · 外观恢复未启用')+
            (f" · 外观相似度 {info['reid_similarity']:.2f} / 候选间隔 {info['reid_margin']:.2f}"
             if 'reid_similarity' in info else '')+
            '。编号不是永久身份；须选择佩戴当前 IMU 的人。')

    def _change_pose_config(self,_index=None,*,model=None):
        return self._vision_controller._change_pose_config(_index, model=model)

    def _on_pose_error(self, message):
        return self._camera_controller._on_pose_error(message)

    def _on_pose_finished(self):
        return self._camera_controller._on_pose_finished()

    def _show_palm_visual(self):
        if self.replay_mode:return
        from .palm_visual_gui import PalmVisualDialog
        if self.palm_visual_dialog is None:self.palm_visual_dialog=PalmVisualDialog(self)
        self.palm_visual_dialog.show();self.palm_visual_dialog.raise_()

    def _show_pose_details(self):
        if self.pose_dialog is None:
            self.pose_dialog = RGBDPoseDialog(self)
        if self.last_pose_estimate is not None:
            self.pose_dialog.update_estimate(self.last_pose_estimate)
        self.pose_dialog.show()
        self.pose_dialog.raise_()

    def _poll_pose(self):
        self._sync_pose_controls()
        if self.pose_worker is None:
            return
        estimate = self.pose_worker.take_latest()
        if estimate is not None and estimate.diagnostics.get('vision_epoch')!=self.vision_epoch:return
        if estimate is not None and self.rgbd_pose_enabled.isChecked():
            self._on_pose_estimate(estimate)
        elif self.last_pose_estimate is not None and time.monotonic_ns()-self.last_pose_estimate.captured_mono_ns > 1_000_000_000:
            self.last_pose_estimate = None
            self.pose_status.setText("视觉观测已过期，不作当前有效数据；原始采集继续")
            if self.pose_dialog is not None:
                self.pose_dialog.canvas.estimate = None
                self.pose_dialog.canvas.update()
                self.pose_dialog.status.setText("观测过期；表格是最后一次历史结果")

    def _on_pose_estimate(self, estimate):
        if not self.replay_mode and 'vision_epoch' in estimate.diagnostics and estimate.diagnostics['vision_epoch']!=self.vision_epoch:return
        self.last_pose_estimate = estimate
        self._update_person_candidates(estimate)
        if self.palm_visual_dialog is not None:self.palm_visual_dialog.on_body(estimate)
        if self.visual_wearing_candidates:
            self._render_dirty = True
        if not self.replay_mode:
            if self._reuse_calibration_dialog is not None:
                self._reuse_calibration_dialog.on_estimate(estimate)
            guide=self.camera_calibration_dialog
            allow=(self.rgbd_pose_enabled.isChecked() and not self.freeze_fusion.isChecked()
                   and self.camera_worker is not None and getattr(self.camera_worker,'capture_epoch',None)==self._live_camera_epoch
                   and not (guide is not None and guide.active)
                   and not (self.calibration_dialog is not None and self.calibration_dialog.isVisible()))
            self.live_fusion.observe(estimate,allow=allow)
            if any(s.get('applied') for s in self.live_fusion.states.values()):self._render_dirty=True
            self._update_live_fusion_status()
        age_ms = (time.monotonic_ns()-estimate.captured_mono_ns)/1e6
        fresh = self.replay_mode or age_ms < 1000
        skipped = self.pose_worker.input_skipped if self.pose_worker is not None else 0
        self.pose_status.setText(
            f"视觉 {int(estimate.valid.sum())}/8 点有效 · 推理 {estimate.inference_ms:.0f} ms"
            + (" · 回放保存结果" if self.replay_mode else f" · 观测龄期 {age_ms:.0f} ms · 输入跳过 {skipped}")
            + " · " + estimate.diagnostics.get("reject_reason", "真实深度；纠偏仅作用于已启用融合部位"))
        if not fresh:
            self.pose_status.setText("推理结果过期，未显示为当前观测；原始采集继续")
        elif not self.depth_preview.isChecked() and estimate.frame_rgb is not None:
            rgb = draw_pose_rgb(estimate)
            palm=self.palm_visual_dialog
            if palm is not None and palm.active and palm.latest is not None and abs(palm.latest['captured_mono_ns']-estimate.captured_mono_ns)<=150_000_000:
                from .palm_worker import draw_palm_diagnostics
                rgb=draw_palm_diagnostics(rgb,palm.latest)
            h, w = rgb.shape[:2]
            self._display_image(QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy(),
                                person_frame_ns=estimate.captured_mono_ns)
        if self.pose_dialog is not None:
            self.pose_dialog.update_estimate(estimate)
            if not fresh:
                self.pose_dialog.canvas.estimate = None
                self.pose_dialog.canvas.update()
                self.pose_dialog.status.setText("结果过期；仅展示历史数值")
        if self.session is not None:
            try:
                self.session.write_vision(estimate)
            except (OSError, RuntimeError, ValueError) as exc:
                self._stop_session()
                self.statusBar().showMessage(f"视觉结果写入失败：{exc}")

    def _poll_rgbd(self):
        self._update_live_fusion_status()
        self._poll_pose()
        worker = self.camera_worker
        if not isinstance(worker, OrbbecWorker):
            return
        observation = worker.take_latest()
        if observation is None:
            return
        self.last_rgbd_observation = observation
        if self.palm_visual_dialog is not None:self.palm_visual_dialog.submit(observation)
        self._on_observation(observation)
        if self.rgbd_pose_enabled.isChecked() and self.pose_worker is not None:
            self.pose_worker.submit(observation)
        timing = observation.rgbd.timing
        delta_ms = abs(timing["color_device_us"] - timing["depth_device_us"]) / 1000
        self.camera_status.setText(
            f"Gemini RGB-D · 输出上限 {worker.fps:g} FPS · 已收 {self.camera_count} 帧"
            f" · 深度有效 {timing['valid_depth_fraction']:.0%} · RGB-D Δt {delta_ms:.3f} ms"
            f" · GUI跳过 {worker.consumer_skipped}" + (" · RGB-D融合状态见下方" if GYRO_RGBD in self.engine.algorithm_modes.values() else " · 只采集，不做视觉纠偏"))
        if self.depth_preview.isChecked() or not self.rgbd_pose_enabled.isChecked() or self.last_pose_estimate is None:
            self._display_rgbd(observation)

    def _display_rgbd(self, observation):
        image = observation.frame_rgb
        if self.depth_preview.isChecked():
            import cv2
            d = observation.rgbd.aligned_depth_mm
            gray = np.uint8(np.clip(d / 6000 * 255, 0, 255))
            image = cv2.cvtColor(cv2.applyColorMap(255 - gray, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            image[d <= 0] = 0
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        frame = QtGui.QImage(image.data, w, h, image.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        self._display_image(frame)

    def _update_live_fusion_status(self):
        self._update_live_comparison_status()
        text=[]
        for part,state in self.live_fusion.states.items():
            reason=state['reason']
            if self.freeze_fusion.isChecked() or not self.rgbd_pose_enabled.isChecked():reason='视觉冻结/关闭；保留修正'
            elif self.last_pose_estimate is None or time.monotonic_ns()-self.last_pose_estimate.captured_mono_ns>500_000_000:reason='视觉过期；保留修正'
            name={'left_upper_arm':'左上臂','left_forearm':'左前臂','right_upper_arm':'右上臂','right_forearm':'右前臂'}.get(part,part)
            binding=self.live_fusion.bindings.get(part)
            warning=binding[1].limit_status() if binding and hasattr(binding[1],'limit_status') else state.get('limit_status')
            text.append(f"{name}：{state['offset_deg']:+.2f}° · {reason}"+(f' · ⚠ {warning}' if warning else ''))
        if text:self.live_fusion_status.setText('\n'.join(text))
        events=self.live_fusion.drain_events()
        if self._fusion_events_file is not None:
            import json
            try:
                for event in events:self._fusion_events_file.write(json.dumps(event,ensure_ascii=False)+'\n')
            except OSError as exc:
                if self.session is not None:self.session.error='纠偏事件写入失败：'+str(exc)
                self._stop_session()

    def apply_camera_report(self,report):
        return self._calibration_controller.apply_camera_report(report)

    def prepare_camera_gravity(self,devices,start_ns,end_ns):
        return self._calibration_controller.prepare_camera_gravity(devices, start_ns, end_ns)

    def _check_vision_report(self,report):
        return self._calibration_controller._check_vision_report(report)

    def apply_visual_wearing_report(self, report, parts, *, review_acknowledged=False):
        return self._calibration_controller.apply_visual_wearing_report(report, parts, review_acknowledged=review_acknowledged)

    def restore_visual_wearing(self, backup):
        return self._calibration_controller.restore_visual_wearing(backup)

    def _algorithm_changed(self,key):
        mode=self.algorithm_combos[key].currentData()
        if (mode==GYRO_RGBD or self.engine.algorithm_modes[key]==GYRO_RGBD) and self.x1_preview_dialog is not None and self.x1_preview_dialog.sender is not None:
            combo=self.algorithm_combos[key];combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(self.engine.algorithm_modes[key]));combo.blockSignals(False)
            QtWidgets.QMessageBox.warning(self,'先验证人体模型','请先停止X1随动，再启用新的融合算法。')
            return
        super()._algorithm_changed(key)

    def stop_camera(self) -> None:
        return self._camera_controller.stop_camera()

    def _on_camera_error(self, message: str) -> None:
        return self._camera_controller._on_camera_error(message)

    def _on_camera_finished(self) -> None:
        return self._camera_controller._on_camera_finished()

    def _on_preview(self, frame: QtGui.QImage, _points: object,
                    _visibility: object, fps: float) -> None:
        self._display_image(frame)
        self.camera_status.setText(
            f"{self.backend} · {fps:.1f} 帧/秒 · 已收到 {self.camera_count} 帧"
            f" · 可信关键点 {self.visible_points}")

    def _display_image(self, frame: QtGui.QImage, *, person_frame_ns=None) -> None:
        self._person_preview_frame_ns=person_frame_ns
        self.camera_preview.display_image(frame)

    def _toggle_camera_preview_window(self):
        if self.camera_preview_window is not None and self.camera_preview_window.isVisible():
            self.camera_preview_window.close()
            return
        if self.camera_preview_window is None:
            self.camera_preview_window = CameraPreviewWindow(self)
            self.camera_preview_window.returning.connect(self._return_camera_preview)
        self._camera_preview_layout.removeWidget(self.camera_preview)
        self.camera_preview.setMinimumHeight(160)
        self.camera_preview.setMaximumHeight(16777215)
        self.camera_preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                         QtWidgets.QSizePolicy.Policy.Ignored)
        self.camera_preview_window.preview_layout.insertWidget(0, self.camera_preview, 1)
        self.camera_preview_placeholder.show()
        self.detach_camera_button.setText("相机画面返回主界面 ↙")
        self.camera_preview_window.show()
        self.camera_preview.show()
        self.camera_preview_window.raise_()
        self.camera_preview_window.activateWindow()

    def _return_camera_preview(self):
        if self.camera_preview.parentWidget() is not self.camera_preview_window:
            return
        self.camera_preview_window.preview_layout.removeWidget(self.camera_preview)
        self.camera_preview.setFixedHeight(120)
        self.camera_preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                         QtWidgets.QSizePolicy.Policy.Fixed)
        index = self._camera_preview_layout.indexOf(self.camera_preview_placeholder)
        self._camera_preview_layout.insertWidget(index, self.camera_preview)
        self.camera_preview_placeholder.hide()
        self.camera_preview.show()
        self.detach_camera_button.setText("相机画面独立窗口 ↗")

    def _on_observation(self, observation: CameraObservation) -> None:
        self.camera_count += 1
        self.visible_points = (0 if observation.confidence is None else
                               sum(float(score) >= 0.35 for score in observation.confidence))
        if self.session is not None:
            try:
                self.session.write_camera(observation)
            except (OSError, RuntimeError, ValueError) as exc:
                self._stop_session()
                self.statusBar().showMessage(f"联合录制中断：{exc}")

    def _poll(self) -> None:
        poll_started=time.monotonic()
        from .real_robot_timing import acquisition_poll, acquisition_feed
        acquisition_poll(self,poll_started)
        try:
            timed_packets = self.source.poll_timed()
        except (OSError, ValueError) as exc:
            self.poll_timer.stop()
            self.statusBar().showMessage(f"数据接收/回放失败：{exc}")
            return
        acquisition_poll(self,poll_started,timed_packets)
        packets = []
        runtime=self.source.take_runtime() if self.replay_mode and hasattr(self.source,'take_runtime') else {}
        for received_ns, packet in timed_packets:
            packets.append(packet)
            received_at = time.monotonic() if self.replay_mode else received_ns / 1e9
            self._check_calibration_reconnect(packet.device_id, received_at)
            feed_started = time.monotonic()
            self.engine.feed(packet, received_at=received_at)
            acquisition_feed(self,packet,received_ns,feed_started)
            self._feed_gripper_packet(round(received_at*1e9), packet)
            recorded_runtime = runtime.get((received_ns, packet.device_id))
            if recorded_runtime is not None:
                if self.candidate_engine is None:
                    self.engine.restore_runtime(recorded_runtime)
                else:
                    # A/B uses the SAME actually recorded rotations, while
                    # preserving the common fitted neutral and chosen B.
                    from copy import deepcopy
                    replay_row = deepcopy(recorded_runtime)
                    if packet.device_id in self.comparison_neutral_devices:
                        replay_row['neutral_by_mode'] = {m: r.tolist() for m, r in
                            self.engine.mode_neutral[packet.device_id].items()}
                    self.engine.restore_runtime(replay_row)
            self._last_replay_stamp = received_ns
            if not self.replay_mode:self.live_fusion.feed(packet.device_id,received_ns)
            if self.calibration_dialog is not None:
                self.calibration_dialog.on_packet(received_ns, packet)
            if self.candidate_engine is not None:
                self.candidate_engine.feed(packet, received_at=received_at)
                if recorded_runtime is not None:
                    from copy import deepcopy
                    replay_row = deepcopy(recorded_runtime)
                    if packet.device_id in self.comparison_neutral_devices:
                        replay_row['neutral_by_mode'] = {m: r.tolist() for m, r in
                            self.candidate_engine.mode_neutral[packet.device_id].items()}
                    self.candidate_engine.restore_runtime(replay_row)
            if self.session is not None:
                try:
                    self.session.write_imu(received_ns, packet,self.engine.runtime_snapshot(packet.device_id,received_ns))
                except (OSError, RuntimeError) as exc:
                    self._stop_session()
                    self.statusBar().showMessage(f"联合录制中断：{exc}")
        if self.settings_dialog is not None or self.calibration_dialog is not None:
            replies = self.source.take_replies()
        else:
            replies = []
        if self.settings_dialog is not None:
            self.settings_dialog.on_packets(packets)
            self.settings_dialog.on_replies(replies)
        if self.calibration_dialog is not None:
            self.calibration_dialog.on_replies(replies)
        if packets:
            self._render_dirty = True
        if self.replay_mode:
            for observation in self.source.take_camera():
                self._on_observation(observation)
                if observation.rgbd is not None:
                    self._display_rgbd(observation)
                    self.camera_status.setText(f"回放 RGB-D · 第 {self.camera_count} 帧 · 深度可切换预览")
                    continue
                rgb = observation.frame_rgb
                height, width = rgb.shape[:2]
                frame = QtGui.QImage(rgb.data, width, height, rgb.strides[0],
                                     QtGui.QImage.Format.Format_RGB888).copy()
                self._display_image(frame)
                self.camera_status.setText(
                    f"回放 {observation.backend} · 第 {self.camera_count} 帧")
            for estimate in self.source.take_vision():
                if self.rgbd_pose_enabled.isChecked():
                    self._on_pose_estimate(estimate)
        now = time.monotonic()
        if self._render_dirty and now - self._last_render >= 1 / 30:
            self._render_pose()

        if hasattr(self.source,'diagnostic_event'):
            self.source.diagnostic_event('gui_poll_end',duration_ms=(time.monotonic()-poll_started)*1000,
                                         packets=len(timed_packets))

    def toggle_session(self) -> None:
        return self._recording_controller.toggle_session()

    def _show_camera_calibration(self):
        if self.camera_calibration_dialog is not None and (self.camera_calibration_dialog.active or
                                                          self.camera_calibration_dialog.state == 'analyzing'):
            self.camera_calibration_dialog.raise_()
            return
        if self._heading_calibration_dialog is None:
            self._heading_calibration_dialog = CameraCalibrationDialog(self)
        if self.camera_calibration_dialog is not None and self.camera_calibration_dialog is not self._heading_calibration_dialog:
            self.camera_calibration_dialog.hide()
        self.camera_calibration_dialog = self._heading_calibration_dialog
        self.camera_calibration_dialog.show()
        self.camera_calibration_dialog.raise_()
        self.camera_calibration_dialog.activateWindow()

    def _show_calibration_reuse(self):
        if self.camera_calibration_dialog is not None and self.camera_calibration_dialog.active:
            self.camera_calibration_dialog.raise_();return
        error=self._pose_change_error()
        if error:self.statusBar().showMessage(error,8000);return
        from .calibration_reuse_gui import CalibrationReuseDialog
        if self._reuse_calibration_dialog is None:self._reuse_calibration_dialog=CalibrationReuseDialog(self)
        if self.camera_calibration_dialog is not None:self.camera_calibration_dialog.hide()
        self.camera_calibration_dialog=self._reuse_calibration_dialog
        self.camera_calibration_dialog.show();self.camera_calibration_dialog.raise_()
        self.camera_calibration_dialog.activateWindow()

    def _show_combined_calibration(self):
        if self.camera_calibration_dialog is not None and self.camera_calibration_dialog.active:
            self.camera_calibration_dialog.raise_();return
        error=self._pose_change_error()
        if error:self.statusBar().showMessage(error,8000);return
        from .combined_calibration_gui import CombinedCalibrationDialog
        if self._combined_calibration_dialog is None:self._combined_calibration_dialog=CombinedCalibrationDialog(self)
        if self.camera_calibration_dialog is not None:self.camera_calibration_dialog.hide()
        self.camera_calibration_dialog=self._combined_calibration_dialog
        self.camera_calibration_dialog.show();self.camera_calibration_dialog.raise_()
        self.camera_calibration_dialog.activateWindow()

    def _show_visual_wearing(self):
        from .visual_wearing_gui import VisualWearingDialog
        if self.camera_calibration_dialog is not None and (self.camera_calibration_dialog.active or
                                                          self.camera_calibration_dialog.state == 'analyzing'):
            self.camera_calibration_dialog.raise_()
            return
        if self._visual_wearing_dialog is None:
            self._visual_wearing_dialog = VisualWearingDialog(self)
        if self.camera_calibration_dialog is not None and self.camera_calibration_dialog is not self._visual_wearing_dialog:
            self.camera_calibration_dialog.hide()
        self.camera_calibration_dialog = self._visual_wearing_dialog
        self.camera_calibration_dialog.show()
        self.camera_calibration_dialog.raise_()
        self.camera_calibration_dialog.activateWindow()

    def _start_session(self) -> bool:
        return self._recording_controller._start_session()

    def _start_timed_session(self) -> None:
        return self._recording_controller._start_timed_session()

    def _cancel_record_schedule(self) -> None:
        return self._recording_controller._cancel_record_schedule()

    def _update_record_schedule(self) -> None:
        return self._recording_controller._update_record_schedule()

    def _stop_session(self, *, auto: bool = False) -> None:
        return self._recording_controller._stop_session(auto=auto)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Request physical command braking before any blocking camera shutdown.
        if self.real_robot_dialog is not None and not self.real_robot_dialog.close():
            event.ignore()
            return
        if self.palm_visual_dialog is not None:
            if not self.palm_visual_dialog.shutdown():event.ignore();return
            self.palm_visual_dialog.close()
        if self._combined_calibration_dialog is not None and self._combined_calibration_dialog._cleanup_running():
            self.statusBar().showMessage('原始录制正在删除，请等待完成后关闭程序。')
            event.ignore();return
        dialogs = {dialog for dialog in (self.camera_calibration_dialog,
                    self._heading_calibration_dialog, self._visual_wearing_dialog, self._reuse_calibration_dialog,
                    self._combined_calibration_dialog) if dialog is not None}
        for dialog in dialogs:
            dialog.shutdown()
        self._cancel_record_schedule()
        self._stop_session()  # Fix the recording cutoff before SDK shutdown latency.
        self.rgbd_pose_enabled.blockSignals(True)
        self.rgbd_pose_enabled.setChecked(False)
        self.rgbd_pose_enabled.blockSignals(False)
        if self.pose_worker is not None:
            self.pose_worker.stop()
            if not self.pose_worker.wait(5000):
                self.pose_status.setText("视觉模型正在退出，请稍后关闭")
                event.ignore()
                return
        if self.pose_dialog is not None:
            self.pose_dialog.close()
        if self.camera_preview_window is not None:
            self.camera_preview_window.close()
        if self.camera_worker is not None:
            self.camera_worker.stop()
            if not self.camera_worker.wait(5000):
                self.camera_status.setText("相机正在关闭，请稍后再试")
                event.ignore()
                return
        super().closeEvent(event)
