"""Single-window camera + WT901 acquisition baseline (fusion comes later)."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyvista as pv

from .camera_gui import CameraWorker
from .fusion_capture import CameraObservation
from .fusion_recording import FusionSessionRecorder
from .fusion_replay import FusionSessionReplay
from .gui import BodyWindow, MODEL_DIR, _axis_triad
from .pose import PoseEngine
from .skeleton import PARTS


class FusionWindow(BodyWindow):
    def __init__(self, source: object, settings_path: Path,
                 assignments: dict[str, str], axis_mappings: dict,
                 *, camera_index: int = 0, backend: str = "rtmw3d",
                 model_path: Path | None = None,
                 calibration_candidates: list[dict] | None = None,
                 comparison_shared_calibrations: list[dict] | None = None,
                 manual_mapping_source: str = "当前配置") -> None:
        super().__init__(source, settings_path, assignments, axis_mappings)
        self.setWindowTitle("WT901 + 单目相机 · 同步采集基线")
        self.candidate_engine: PoseEngine | None = None
        self.candidate_actors = []
        self.candidate_axis_actors = []
        if calibration_candidates:
            if not self.replay_mode:
                raise ValueError("候选标定只能用于回放预览")
            self.candidate_engine = PoseEngine(assignments, axis_mappings)
            for shared in comparison_shared_calibrations or []:
                key = shared["part"]
                if self.engine.assignments.get(key) != shared["device_id"]:
                    raise ValueError(f"{key} 的共同标定设备与回放分配不一致")
                for engine in (self.engine, self.candidate_engine):
                    engine.apply_calibration(
                        key, shared["device_id"], shared["basis_matrix"],
                        shared["neutral_matrix"])
            for candidate in calibration_candidates:
                key = candidate["part"]
                if self.candidate_engine.assignments.get(key) != candidate["device_id"]:
                    raise ValueError(f"{key} 的候选设备与回放分配不一致")
                if "neutral_matrix" in candidate:
                    neutral = np.asarray(candidate["neutral_matrix"], dtype=float)
                    self.candidate_engine.apply_calibration(
                        key, candidate["device_id"], candidate["basis_matrix"], neutral)
                    # Both avatars must use the same neutral hold; otherwise
                    # differences in the first (possibly moving) packet bias
                    # the visual A/B comparison.
                    self.engine.neutral[candidate["device_id"]] = neutral.copy()
                else:
                    self.candidate_engine.set_continuous_basis(key, candidate["basis_matrix"])
            self._install_comparison_view(manual_mapping_source)
            self.setWindowTitle("WT901 姿态映射对比 · 同一数据 / 双模型")
            for combo in self.combos.values():
                combo.setEnabled(False)
            for button in self.mapping_buttons.values():
                button.setEnabled(False)
        self.record_button.hide()  # The legacy CSV cannot contain camera frames.
        self.camera_index = camera_index
        self.backend = backend
        self.model_path = model_path or Path()
        self.camera_worker: CameraWorker | None = None
        self.session: FusionSessionRecorder | None = None
        self.camera_count = 0
        self.visible_points = 0
        self._camera_failed = False
        self._record_start_deadline: float | None = None
        self._record_stop_deadline: float | None = None
        self.record_schedule_timer = QtCore.QTimer(self)
        self.record_schedule_timer.setInterval(100)
        self.record_schedule_timer.timeout.connect(self._update_record_schedule)

        group = QtWidgets.QGroupBox("相机 + IMU 采集（当前模型仅由 IMU 驱动）")
        group_layout = QtWidgets.QVBoxLayout(group)
        self.camera_preview = QtWidgets.QLabel("等待相机画面")
        self.camera_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.camera_preview.setMinimumSize(240, 120)
        self.camera_preview.setMaximumHeight(160)
        self.camera_preview.setStyleSheet("background: #191b20; color: white")
        group_layout.addWidget(self.camera_preview)
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
        if isinstance(source, FusionSessionReplay):
            self.start_camera_button.setEnabled(False)
            self.camera_status.setText(
                (f"左：你的手动轴映射（{manual_mapping_source}）；右：我的候选标定。"
                 "两者使用同一批 IMU 包；"
                 "相机不驱动模型。\n" if calibration_candidates else "")
                + f"正在回放：{source.path}")
        else:
            self.start_camera()

    def _install_comparison_view(self, manual_mapping_source: str) -> None:
        """Keep one VTK viewport/camera and place both avatars side by side."""
        layout = self.centralWidget().layout()
        layout.takeAt(0)
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)
        labels = QtWidgets.QHBoxLayout()
        for caption, color in ((f"你的映射 · 手动 XYZ（{manual_mapping_source}）", "#76D7FF"),
                               ("我的候选 · 连续标定（仅回放）", "#FFBD72")):
            label = QtWidgets.QLabel(caption)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
                f"background: #242936; color: {color}; font-weight: bold;"
                "padding: 6px; font-size: 13px;")
            labels.addWidget(label, 1)
        container_layout.addLayout(labels)
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

    def _render_pose(self) -> None:
        if getattr(self, "candidate_engine", None) is None:
            super()._render_pose()
            return
        for actors, axis_actors, engine, offset_y in (
                (self.actors, self.axis_actors, self.engine, -0.8),
                (self.candidate_actors, self.candidate_axis_actors,
                 self.candidate_engine, +0.8)):
            offset = np.eye(4)
            offset[1, 3] = offset_y
            for actor, axis_actor, matrix in zip(
                    actors, axis_actors, engine.transforms()):
                positioned = offset @ matrix
                actor.user_matrix = positioned
                axis_actor.user_matrix = positioned
        self.plotter.render()
        self._render_dirty = False
        self._last_render = time.monotonic()

    def _toggle_axes(self, visible: bool) -> None:
        super()._toggle_axes(visible)
        for actor in getattr(self, "candidate_axis_actors", ()):
            actor.SetVisibility(visible)
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
        if self.camera_worker is not None or self.replay_mode:
            return
        worker = CameraWorker(self.camera_index, self.model_path,
                              backend=self.backend, parent=self,
                              capture_observations=True)
        worker.frame_ready.connect(self._on_preview)
        worker.observation_ready.connect(self._on_observation)
        worker.error.connect(self._on_camera_error)
        worker.finished.connect(self._on_camera_finished)
        self.camera_worker = worker
        self._camera_failed = False
        self.start_camera_button.setEnabled(False)
        self.stop_camera_button.setEnabled(True)
        self.camera_status.setText(f"正在打开摄像头 {self.camera_index} / {self.backend}…")
        worker.start()

    def stop_camera(self) -> None:
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.stop_camera_button.setEnabled(False)
            self.camera_status.setText("正在停止相机…")

    def _on_camera_error(self, message: str) -> None:
        self._camera_failed = True
        self.camera_status.setText(message)

    def _on_camera_finished(self) -> None:
        worker = self.camera_worker
        self.camera_worker = None
        if worker is not None:
            worker.deleteLater()
        self.start_camera_button.setEnabled(not self.replay_mode)
        self.stop_camera_button.setEnabled(False)
        if not self._camera_failed:
            self.camera_status.setText("相机已停止")

    def _on_preview(self, frame: QtGui.QImage, _points: object,
                    _visibility: object, fps: float) -> None:
        self._display_image(frame)
        self.camera_status.setText(
            f"{self.backend} · {fps:.1f} 帧/秒 · 已收到 {self.camera_count} 帧"
            f" · 可信关键点 {self.visible_points}")

    def _display_image(self, frame: QtGui.QImage) -> None:
        self.camera_preview.setPixmap(QtGui.QPixmap.fromImage(frame).scaled(
            self.camera_preview.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))

    def _on_observation(self, observation: CameraObservation) -> None:
        self.camera_count += 1
        self.visible_points = (0 if observation.confidence is None else
                               sum(float(score) >= 0.35 for score in observation.confidence))
        if self.session is not None:
            try:
                self.session.write_camera(observation)
            except (OSError, RuntimeError) as exc:
                self._stop_session()
                self.statusBar().showMessage(f"联合录制中断：{exc}")

    def _poll(self) -> None:
        try:
            timed_packets = self.source.poll_timed()
        except (OSError, ValueError) as exc:
            self.poll_timer.stop()
            self.statusBar().showMessage(f"数据接收/回放失败：{exc}")
            return
        packets = []
        for received_ns, packet in timed_packets:
            packets.append(packet)
            received_at = time.monotonic() if self.replay_mode else received_ns / 1e9
            self._check_calibration_reconnect(packet.device_id, received_at)
            self.engine.feed(packet, received_at=received_at)
            if self.calibration_dialog is not None:
                self.calibration_dialog.on_packet(received_ns, packet)
            if self.candidate_engine is not None:
                self.candidate_engine.feed(packet, received_at=received_at)
            if self.session is not None:
                try:
                    self.session.write_imu(received_ns, packet)
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
                rgb = observation.frame_rgb
                height, width = rgb.shape[:2]
                frame = QtGui.QImage(rgb.data, width, height, rgb.strides[0],
                                     QtGui.QImage.Format.Format_RGB888).copy()
                self._display_image(frame)
                self.camera_status.setText(
                    f"回放 {observation.backend} · 第 {self.camera_count} 帧")
        now = time.monotonic()
        if self._render_dirty and now - self._last_render >= 1 / 30:
            self._render_pose()

    def toggle_session(self) -> None:
        if self.session is not None:
            self._stop_session()
            return
        self._cancel_record_schedule()
        self._start_session()

    def _start_session(self) -> bool:
        default_parent = Path(__file__).resolve().parent.parent / "recordings"
        try:
            self.session = FusionSessionRecorder(
                default_parent, assignments=self.engine.assignments,
                axis_mappings=self.engine.axis_mappings)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "无法开始录制", str(exc))
            return False
        self.session_button.setText("结束联合录制")
        self.record_schedule_status.setText("录制中；结束时末尾 3 秒不保存")
        self.statusBar().showMessage(f"正在录制：{self.session.path}")
        return True

    def _start_timed_session(self) -> None:
        if self.replay_mode or self.session is not None or self._record_start_deadline is not None:
            return
        self._record_start_deadline = time.monotonic() + self.record_delay.value()
        self.timed_session_button.setEnabled(False)
        self.cancel_session_button.setEnabled(True)
        self.record_delay.setEnabled(False)
        self.record_duration.setEnabled(False)
        self.record_schedule_timer.start()
        self._update_record_schedule()

    def _cancel_record_schedule(self) -> None:
        if self._record_start_deadline is None:
            return
        self._record_start_deadline = None
        self.record_schedule_timer.stop()
        self.timed_session_button.setEnabled(not self.replay_mode)
        self.cancel_session_button.setEnabled(False)
        self.record_delay.setEnabled(True)
        self.record_duration.setEnabled(True)
        self.record_schedule_status.setText("定时录制已取消；没有创建录制文件")

    def _update_record_schedule(self) -> None:
        now = time.monotonic()
        if self._record_start_deadline is not None:
            remaining = self._record_start_deadline - now
            if remaining > 0:
                self.record_schedule_status.setText(
                    f"还有 {remaining:.1f} 秒开始录制；可离开电脑摆好姿势")
                return
            self._record_start_deadline = None
            self.cancel_session_button.setEnabled(False)
            if not self._start_session():
                self.record_schedule_timer.stop()
                self.timed_session_button.setEnabled(not self.replay_mode)
                self.record_delay.setEnabled(True)
                self.record_duration.setEnabled(True)
                self.record_schedule_status.setText("定时录制未能启动")
                return
            self._record_stop_deadline = now + self.record_duration.value()
        if self._record_stop_deadline is not None:
            remaining = self._record_stop_deadline - now
            if remaining <= 0:
                self._stop_session(auto=True)
            else:
                self.record_schedule_status.setText(
                    f"录制中；{remaining:.1f} 秒后自动结束，末尾 3 秒不保存")

    def _stop_session(self, *, auto: bool = False) -> None:
        self._record_start_deadline = None
        self._record_stop_deadline = None
        self.record_schedule_timer.stop()
        self.timed_session_button.setEnabled(not self.replay_mode)
        self.cancel_session_button.setEnabled(False)
        self.record_delay.setEnabled(True)
        self.record_duration.setEnabled(True)
        session, self.session = self.session, None
        if session is None:
            return
        self.session_button.setText("开始联合录制")
        try:
            session.close()
        except OSError as exc:
            self.record_schedule_status.setText(f"录制不完整：{exc}")
            self.statusBar().showMessage(f"录制不完整：{exc}；目录 {session.path}")
        else:
            self.record_schedule_status.setText(
                ("定时录制完成" if auto else "录制已结束")
                + f"；末尾 3 秒已剔除（IMU {session.trimmed_imu_packets} 包、"
                f"相机 {session.trimmed_camera_frames} 帧）")
            self.statusBar().showMessage(
                f"已录制 {session.imu_count} 包 / {session.frame_count} 帧"
                f"（丢 {session.dropped_frames} 帧）：{session.path}", 10000)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.camera_worker is not None:
            self.camera_worker.stop()
            if not self.camera_worker.wait(5000):
                self.camera_status.setText("相机正在关闭，请稍后再试")
                event.ignore()
                return
        self._cancel_record_schedule()
        self._stop_session()
        super().closeEvent(event)
