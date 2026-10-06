"""Camera-only upper-body prototype, isolated from the WT901 UDP viewer."""

from __future__ import annotations

import math
import importlib.util
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor

from .camera_pose import CameraPoseEngine, UPPER_BODY_KEYS
from .camera_rtmw3d import RTMWPoseStabilizer, adapt_rtmw3d, best_person_index
from .fusion_capture import CameraObservation
from .gui import MODEL_DIR, _axis_triad
from .skeleton import PARTS


PREVIEW_EDGES = (
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19),
    (12, 14), (14, 16), (16, 18), (16, 20),
)
RTMW_PREVIEW_EDGES = (
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (9, 96), (9, 108), (10, 117), (10, 129),
)


class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(QtGui.QImage, object, object, float)
    observation_ready = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(self, camera_index: int, model_path: Path,
                 backend: str = "mediapipe",
                 stabilize_rtmw: bool = True,
                 parent: QtCore.QObject | None = None,
                 capture_observations: bool = False) -> None:
        super().__init__(parent)
        self.camera_index = camera_index
        self.model_path = model_path
        self.backend = backend
        self._stabilize_rtmw = stabilize_rtmw
        self._reset_stabilizer = False
        self._running = True
        self.capture_observations = capture_observations

    def _emit_observation(self, captured_ns: int, rgb: np.ndarray,
                          keypoints_xy: np.ndarray | None,
                          confidence: np.ndarray | None) -> None:
        if self.capture_observations:
            self.observation_ready.emit(CameraObservation(
                captured_ns, self.backend, rgb.copy(),
                None if keypoints_xy is None else keypoints_xy.copy(),
                None if confidence is None else confidence.copy()))

    def stop(self) -> None:
        self._running = False

    def set_stabilization(self, enabled: bool) -> None:
        if self._stabilize_rtmw != enabled:
            self._stabilize_rtmw = enabled
            self._reset_stabilizer = True

    def run(self) -> None:
        import cv2

        camera = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            camera.release()
            self.error.emit(f"无法打开摄像头编号 {self.camera_index}")
            return
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            if self.backend == "mediapipe":
                self._run_mediapipe(camera, cv2)
            elif self.backend == "rtmw3d":
                self._run_rtmw3d(camera, cv2)
            else:
                raise ValueError(f"未知姿态后端：{self.backend}")
        except Exception as exc:
            self.error.emit(f"{self.backend} 相机识别中断：{exc}")
        finally:
            camera.release()

    def _emit_frame(self, rgb: np.ndarray, points: np.ndarray | None,
                    visibility: np.ndarray | None, previous_at: float) -> float:
        height, width = rgb.shape[:2]
        preview = QtGui.QImage(rgb.data, width, height, rgb.strides[0],
                               QtGui.QImage.Format.Format_RGB888).copy()
        now = time.monotonic()
        self.frame_ready.emit(preview, points, visibility,
                              1.0 / max(now - previous_at, 1e-3))
        return now

    def _run_mediapipe(self, camera: object, cv2: object) -> None:
        import mediapipe as mp

        last_timestamp = -1
        last_frame_at = time.monotonic()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            failed_reads = 0
            while self._running:
                ok, bgr = camera.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads >= 30:
                        self.error.emit("摄像头连续读取失败，请检查权限或是否被其他程序占用")
                        break
                    self.msleep(30)
                    continue
                failed_reads = 0
                captured_ns = time.monotonic_ns()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp = max(last_timestamp + 1, int(time.monotonic() * 1000))
                last_timestamp = timestamp
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(rgb))
                result = detector.detect_for_video(image, timestamp)
                points = visibility = None
                pixel_points = pixel_scores = None
                if result.pose_world_landmarks and result.pose_landmarks:
                    world = result.pose_world_landmarks[0]
                    points = np.asarray([(p.x, p.y, p.z) for p in world], dtype=float)
                    visibility = np.asarray([p.visibility for p in world], dtype=float)
                    image_landmarks = result.pose_landmarks[0]
                    height, width = rgb.shape[:2]
                    pixel_points = np.asarray(
                        [(p.x * width, p.y * height) for p in image_landmarks],
                        dtype=float)
                    pixel_scores = np.asarray(
                        [p.visibility for p in image_landmarks], dtype=float)
                self._emit_observation(captured_ns, rgb, pixel_points, pixel_scores)
                if pixel_points is not None:
                    for a, b in PREVIEW_EDGES:
                        pa, pb = image_landmarks[a], image_landmarks[b]
                        if min(pa.visibility, pb.visibility) < 0.5:
                            continue
                        start = (round(pa.x * width), round(pa.y * height))
                        end = (round(pb.x * width), round(pb.y * height))
                        cv2.line(rgb, start, end, (75, 225, 125), 2)
                last_frame_at = self._emit_frame(rgb, points, visibility,
                                                 last_frame_at)

    def _run_rtmw3d(self, camera: object, cv2: object) -> None:
        import onnxruntime as ort
        from rtmlib import Wholebody3d

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("未发现 CUDAExecutionProvider，请在组合环境安装 GPU 版 ONNX Runtime")
        detector = Wholebody3d(backend="onnxruntime", device="cuda")
        for model in (detector.det_model, detector.pose_model):
            if "CUDAExecutionProvider" not in model.session.get_providers():
                raise RuntimeError("RTMW3D 未启用 CUDA；请检查运行库")

        failed_reads = 0
        last_frame_at = time.monotonic()
        stabilizer = RTMWPoseStabilizer()
        empty_keypoints = np.empty((0, 133, 3), dtype=float)
        empty_scores = np.empty((0, 133), dtype=float)
        while self._running:
            ok, bgr = camera.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= 30:
                    self.error.emit("摄像头连续读取失败，请检查权限或是否被其他程序占用")
                    break
                self.msleep(30)
                continue
            failed_reads = 0
            captured_ns = time.monotonic_ns()
            if self._reset_stabilizer:
                stabilizer.reset()
                self._reset_stabilizer = False
            bboxes = detector.det_model(bgr)
            points = visibility = None
            image_points = scores = None
            if len(bboxes):
                keypoints, scores, _, image_points = detector.pose_model(
                    bgr, bboxes=bboxes)
                if self._stabilize_rtmw:
                    points, visibility = stabilizer.update(keypoints, scores)
                else:
                    points, visibility = adapt_rtmw3d(keypoints, scores)
            elif self._stabilize_rtmw:
                stabilizer.update(empty_keypoints, empty_scores)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if image_points is not None and len(image_points):
                # Draw only the upper-body points used by the avatar.
                index = best_person_index(scores)
                person = image_points[index]
                confidence = scores[index]
                self._emit_observation(captured_ns, rgb, person, confidence)
                for a, b in RTMW_PREVIEW_EDGES:
                    if min(confidence[a], confidence[b]) < 0.35:
                        continue
                    start = tuple(np.rint(person[a]).astype(int))
                    end = tuple(np.rint(person[b]).astype(int))
                    cv2.line(rgb, start, end, (75, 225, 125), 2)
            else:
                self._emit_observation(captured_ns, rgb, None, None)
            last_frame_at = self._emit_frame(rgb, points, visibility,
                                             last_frame_at)


class CameraWindow(QtWidgets.QMainWindow):
    def __init__(self, model_path: Path, camera_index: int = 0,
                 *, backend: str = "mediapipe", autostart: bool = True) -> None:
        super().__init__()
        if backend not in ("mediapipe", "rtmw3d"):
            raise ValueError(f"未知姿态后端：{backend}")
        self.model_path = model_path
        self.camera_index = camera_index
        self.engine = CameraPoseEngine(min_visibility=0.35 if backend == "rtmw3d" else 0.5)
        self.worker: CameraWorker | None = None
        self._active_backend = backend
        self._pending_restart = False
        self._worker_failed = False
        self.mesh_actors = []
        self.axis_actors = []
        self._last_render = 0.0
        self._last_detection = 0.0
        self._closing = False
        self._timed_zero_deadline: float | None = None
        self.zero_timer = QtCore.QTimer(self)
        self.zero_timer.setInterval(100)
        self.zero_timer.timeout.connect(self._update_timed_zero)

        self.setWindowTitle("单目相机 · 上身 7 部位随动实验")
        self.resize(1500, 920)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        self.plotter = QtInteractor(central)
        layout.addWidget(self.plotter.interactor, 3)
        self._load_model()

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(390)
        panel.setMaximumWidth(550)
        controls = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(panel, 2)
        title = QtWidgets.QLabel("摄像头上身姿态实验")
        title_font = title.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        controls.addWidget(title)
        self.preview = QtWidgets.QLabel("等待摄像头画面…")
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(260)
        self.preview.setStyleSheet("background: #15181d; color: #d1d5db;")
        controls.addWidget(self.preview)
        self.status = QtWidgets.QLabel("尚未启动相机")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)

        backend_row = QtWidgets.QHBoxLayout()
        backend_row.addWidget(QtWidgets.QLabel("姿态后端"))
        self.backend_choice = QtWidgets.QComboBox()
        self.backend_choice.addItem("MediaPipe", "mediapipe")
        self.backend_choice.addItem("RTMW3D（GPU）", "rtmw3d")
        self.backend_choice.setCurrentIndex(0 if backend == "mediapipe" else 1)
        self.backend_choice.setToolTip("切换时会重启摄像头并清空当前中立位；请重新归零")
        self.backend_choice.currentIndexChanged.connect(self._backend_changed)
        backend_row.addWidget(self.backend_choice, 1)
        controls.addLayout(backend_row)
        self.rtmw_stabilize = QtWidgets.QCheckBox("RTMW3D 防抖（可关闭对比）")
        self.rtmw_stabilize.setChecked(True)
        self.rtmw_stabilize.setEnabled(backend == "rtmw3d")
        self.rtmw_stabilize.setToolTip(
            "平滑肩宽尺度和关键点，过滤孤立跳点；仅作用于 RTMW3D")
        self.rtmw_stabilize.toggled.connect(self._set_rtmw_stabilization)
        controls.addWidget(self.rtmw_stabilize)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("启动相机")
        self.start_button.clicked.connect(self.start_camera)
        buttons.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton("停止相机")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.stop_button)
        self.zero_button = QtWidgets.QPushButton("当前姿势归零")
        self.zero_button.clicked.connect(self.zero)
        buttons.addWidget(self.zero_button)
        controls.addLayout(buttons)
        timed_zero_row = QtWidgets.QHBoxLayout()
        timed_zero_row.addWidget(QtWidgets.QLabel("延时"))
        self.zero_delay = QtWidgets.QSpinBox()
        self.zero_delay.setRange(3, 120)
        self.zero_delay.setValue(10)
        self.zero_delay.setSuffix(" 秒")
        self.zero_delay.setToolTip("倒计时期间可离开电脑，摆好中立姿势")
        timed_zero_row.addWidget(self.zero_delay)
        self.timed_zero_button = QtWidgets.QPushButton("定时归零")
        self.timed_zero_button.clicked.connect(self.start_timed_zero)
        timed_zero_row.addWidget(self.timed_zero_button, 1)
        self.cancel_timed_zero_button = QtWidgets.QPushButton("取消")
        self.cancel_timed_zero_button.setEnabled(False)
        self.cancel_timed_zero_button.clicked.connect(self.cancel_timed_zero)
        timed_zero_row.addWidget(self.cancel_timed_zero_button)
        controls.addLayout(timed_zero_row)
        self.zero_countdown = QtWidgets.QLabel("定时归零未启动")
        self.zero_countdown.setWordWrap(True)
        controls.addWidget(self.zero_countdown)
        options = QtWidgets.QHBoxLayout()
        self.mirror = QtWidgets.QCheckBox("镜像预览（不改变控制方向）")
        options.addWidget(self.mirror)
        self.axes = QtWidgets.QCheckBox("显示局部轴")
        self.axes.setChecked(False)
        self.axes.toggled.connect(self._toggle_axes)
        options.addWidget(self.axes)
        controls.addLayout(options)

        group = QtWidgets.QGroupBox("识别状态 · 躯干和左右手臂 7 部位")
        group_layout = QtWidgets.QVBoxLayout(group)
        self.part_labels: dict[str, QtWidgets.QLabel] = {}
        for part in PARTS:
            if part.key in UPPER_BODY_KEYS:
                label = QtWidgets.QLabel(f"{part.label}：等待识别")
                group_layout.addWidget(label)
                self.part_labels[part.key] = label
        controls.addWidget(group)
        hint = QtWidgets.QLabel(
            "请面向摄像头，保证肩、肘、腕和双手都在画面里；启动后先自然站立，"
            "再点击“当前姿势归零”，或设置延时后用“定时归零”。"
            "若腰/髋也入镜，躯干俯仰估计会更完整；"
            "否则仅粗略估计其朝向。骨盆和双腿保持模型默认姿态。"
            "手掌自转与遮挡后的姿态只作粗略演示。"
            "切换识别后端会清空中立位，需要重新归零。"
            "RTMW3D 的深度与图像坐标经过近似尺度适配，不是精确三维测量。"
            "此窗口不占用 WT901 UDP 1399。")
        hint.setWordWrap(True)
        controls.addWidget(hint)
        controls.addStretch(1)

        reset_view = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        reset_view.activated.connect(self.plotter.reset_camera)
        shortcut_zero = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        shortcut_zero.activated.connect(self.zero)
        if autostart:
            self.start_camera()

    def _load_model(self) -> None:
        missing = [part.mesh for part in PARTS if not (MODEL_DIR / part.mesh).is_file()]
        if missing:
            raise FileNotFoundError(f"缺少人体 STL 模型：{', '.join(missing)}")
        triad = _axis_triad(0.095)
        for part in PARTS:
            mesh = pv.read(MODEL_DIR / part.mesh)
            mesh.points *= 0.001
            actor = self.plotter.add_mesh(mesh, color=part.color,
                                          smooth_shading=False, show_edges=False)
            self.mesh_actors.append(actor)
            if part.key in UPPER_BODY_KEYS:
                axis = self.plotter.add_mesh(
                    triad, scalars="axis_rgb", rgb=True, lighting=False,
                    show_scalar_bar=False, pickable=False)
                axis.SetVisibility(False)
                self.axis_actors.append(axis)
            else:
                self.axis_actors.append(None)
        self.plotter.set_background("#1a1d23")
        self.plotter.show_axes()
        self.plotter.camera_position = [(2.5, 0, 0.15), (0, 0, 0.15), (0, 0, 1)]
        self.plotter.reset_camera()
        self._render_pose()

    def _render_pose(self) -> None:
        for actor, axis, matrix in zip(
                self.mesh_actors, self.axis_actors, self.engine.transforms()):
            actor.user_matrix = matrix
            if axis is not None:
                axis.user_matrix = matrix
        self.plotter.render()
        self._last_render = time.monotonic()

    def _toggle_axes(self, visible: bool) -> None:
        for axis in self.axis_actors:
            if axis is not None:
                axis.SetVisibility(visible)
        self.plotter.render()

    def _backend_ready(self, backend: str) -> str | None:
        if backend == "mediapipe":
            if importlib.util.find_spec("mediapipe") is None:
                return "当前环境缺少 mediapipe；请先安装后再切换"
            if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
                return f"缺少 MediaPipe 模型文件：{self.model_path}"
        elif (importlib.util.find_spec("rtmlib") is None
              or importlib.util.find_spec("onnxruntime") is None):
            return "当前环境缺少 rtmlib 或 onnxruntime-gpu；请先安装后再切换"
        return None

    def _reset_for_backend(self, backend: str) -> None:
        self.engine = CameraPoseEngine(min_visibility=0.35 if backend == "rtmw3d" else 0.5)
        self._last_detection = 0.0
        for key, label in self.part_labels.items():
            name = next(part.label for part in PARTS if part.key == key)
            label.setText(f"{name}：等待识别")
        self._render_pose()

    def _backend_changed(self, _index: int) -> None:
        backend = self.backend_choice.currentData()
        problem = self._backend_ready(backend)
        if problem:
            self.status.setText(problem)
            self.backend_choice.blockSignals(True)
            self.backend_choice.setCurrentIndex(
                self.backend_choice.findData(self._active_backend))
            self.backend_choice.blockSignals(False)
            return
        self.rtmw_stabilize.setEnabled(backend == "rtmw3d")
        self.cancel_timed_zero(silent=True)
        if self.worker is None:
            self._active_backend = backend
            self._reset_for_backend(backend)
            self.status.setText(f"已选择 {self.backend_choice.currentText()}；启动相机后请重新归零")
        else:
            self._pending_restart = True
            self.backend_choice.setEnabled(False)
            self.worker.stop()
            self.status.setText(f"正在切换到 {self.backend_choice.currentText()}…")

    def _set_rtmw_stabilization(self, enabled: bool) -> None:
        if self.worker is not None and self._active_backend == "rtmw3d":
            self.worker.set_stabilization(enabled)
        if self.backend_choice.currentData() == "rtmw3d":
            self.status.setText("RTMW3D 防抖已开启" if enabled else "RTMW3D 防抖已关闭")

    def start_camera(self) -> None:
        if self.worker is not None:
            return
        backend = self.backend_choice.currentData()
        problem = self._backend_ready(backend)
        if problem:
            self.status.setText(problem)
            return
        if backend != self._active_backend:
            self._reset_for_backend(backend)
        self._active_backend = backend
        self._worker_failed = False
        self.worker = CameraWorker(self.camera_index, self.model_path, backend,
                                   self.rtmw_stabilize.isChecked(), self)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_stopped)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText(f"正在用 {self.backend_choice.currentText()} 打开摄像头 {self.camera_index}…")
        self.worker.start()

    def stop_camera(self) -> None:
        if self.worker is not None:
            self._pending_restart = False
            self.backend_choice.setEnabled(True)
            self.cancel_timed_zero(silent=True)
            self.worker.stop()
            self.status.setText("正在停止摄像头…")

    def _on_stopped(self) -> None:
        self.cancel_timed_zero(silent=True)
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        if not self._closing:
            if self._pending_restart:
                self._pending_restart = False
                self.backend_choice.setEnabled(True)
                self.start_camera()
                return
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            if not self._worker_failed:
                self.status.setText("摄像头已停止；模型保持最后姿态")

    def _on_error(self, message: str) -> None:
        self.cancel_timed_zero(silent=True)
        self._worker_failed = True
        self.status.setText(message)

    def _on_frame(self, image: QtGui.QImage, points: object,
                  visibility: object, fps: float) -> None:
        if self._closing or self._pending_restart:
            return
        if self.mirror.isChecked():
            image = image.mirrored(True, False)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.preview.setPixmap(pixmap.scaled(
            self.preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))
        tracked = self.engine.update(points, visibility)
        now = time.monotonic()
        backend_label = self.backend_choice.currentText()
        if self._active_backend == "rtmw3d":
            backend_label += (" · 防抖开" if self.rtmw_stabilize.isChecked()
                              else " · 防抖关")
        if tracked:
            self._last_detection = now
            self.status.setText(
                f"{backend_label} · 相机 {self.camera_index} · {fps:.1f} 帧/s · "
                f"已识别 {len(tracked)}/7 个上身部位")
        else:
            self.status.setText(
                f"未检测到可用上身姿态，保持最后画面姿态；请面向摄像头并调整距离")
        for key, label in self.part_labels.items():
            part_name = next(part.label for part in PARTS if part.key == key)
            label.setText(f"{part_name}：{'跟踪中' if key in tracked else '不可见 / 跟随父部位'}")
        if tracked and now - self._last_render >= 1 / 30:
            self._render_pose()

    def zero(self) -> None:
        self.cancel_timed_zero(silent=True)
        if self._pose_is_fresh() and self.engine.zero():
            self._render_pose()
            self.status.setText("已把当前可见部位设为中立姿势")
        else:
            self.status.setText("尚未识别到当前可用的姿态；未修改中立位")

    def _pose_is_fresh(self) -> bool:
        return (self.worker is not None and self.worker.isRunning()
                and self.engine.has_person
                and time.monotonic() - self._last_detection <= 1.5)

    def start_timed_zero(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            self.zero_countdown.setText("相机尚未运行；请先启动相机")
            return
        self._timed_zero_deadline = time.monotonic() + self.zero_delay.value()
        self.cancel_timed_zero_button.setEnabled(True)
        self.zero_timer.start()
        self._update_timed_zero()

    def cancel_timed_zero(self, _checked: bool = False,
                          *, silent: bool = False) -> None:
        if self._timed_zero_deadline is None:
            return
        self._timed_zero_deadline = None
        self.zero_timer.stop()
        self.cancel_timed_zero_button.setEnabled(False)
        self.zero_countdown.setText(
            "定时归零未启动" if silent else "定时归零已取消")

    def _update_timed_zero(self) -> None:
        if self._timed_zero_deadline is None:
            return
        remaining = math.ceil(self._timed_zero_deadline - time.monotonic())
        if remaining > 0:
            self.zero_countdown.setText(
                f"还剩 {remaining} 秒：请走到镜头前并保持中立姿势")
            return
        self._timed_zero_deadline = None
        self.zero_timer.stop()
        self.cancel_timed_zero_button.setEnabled(False)
        if not self._pose_is_fresh() or not self.engine.zero():
            message = "定时归零未执行：到时未识别到当前姿态；原中立位保留"
        else:
            self._render_pose()
            message = f"定时归零完成：{len(self.engine.latest)}/7 个可见部位已设为中立位"
        self.zero_countdown.setText(message)
        self.status.setText(message)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._closing = True
        self.cancel_timed_zero(silent=True)
        if self.worker is not None:
            self.worker.stop()
            if not self.worker.wait(5000):
                self._closing = False
                self.status.setText("相机正在关闭，请稍后再试")
                event.ignore()
                return
        self.plotter.close()
        super().closeEvent(event)
