"""Camera coordination for the fusion window.

State stays on the window for dialog/replay compatibility; no GUI imports.
"""
from __future__ import annotations



class FusionCameraController:
    def __init__(self, window, *, camera_factory, orbbec_type, pose_factory):
        self.window = window
        self.camera_factory = camera_factory
        self.orbbec_type = orbbec_type
        self.pose_factory = pose_factory

    def start_camera(self) -> None:
        window = self.window
        if window.camera_worker is not None or window.replay_mode:
            return
        if window.camera_source == "orbbec":
            worker = self.orbbec_type(window.orbbec_python, fps=window.rgbd_rate.value(),
                                  warmup=window.rgbd_warmup, parent=window)
            window.last_rgbd_observation = None
            window.rgbd_timer.start()
        else:
            worker = self.camera_factory(window.camera_index, window.model_path,
                                  backend=window.backend, parent=window,
                                  capture_observations=True)
            worker.frame_ready.connect(window._on_preview)
            worker.observation_ready.connect(window._on_observation)
        worker.error.connect(window._on_camera_error)
        worker.finished.connect(window._on_camera_finished)
        window.camera_worker = worker
        window.camera_source_combo.setEnabled(False)
        window.rgbd_rate.setEnabled(False)
        window._camera_failed = False
        window.start_camera_button.setEnabled(False)
        window.stop_camera_button.setEnabled(True)
        window.camera_status.setText(
            f"正在打开 Gemini / 预热 {window.rgbd_warmup:g} 秒…" if window.camera_source == "orbbec"
            else f"正在打开摄像头 {window.camera_index} / {window.backend}…")
        worker.start()


    def _change_camera_source(self):
        window = self.window
        window.camera_source = window.camera_source_combo.currentData()
        window.palm_visual_button.setEnabled(not window.replay_mode and window.camera_source=='orbbec')
        window.visual_wearing_button.setEnabled(not window.replay_mode and window.camera_source == 'orbbec')
        window.reuse_calibration_button.setEnabled(not window.replay_mode and window.camera_source == 'orbbec')
        window.combined_calibration_button.setEnabled(not window.replay_mode and window.camera_source == 'orbbec')
        window.rgbd_pose_enabled.setChecked(False)
        window.rgbd_pose_enabled.setEnabled(window.camera_source == "orbbec")


    def _set_rgbd_pose(self, enabled):
        window = self.window
        if window.replay_mode:
            return
        if not enabled:
            if window.pose_worker is not None:
                window.pose_worker.stop()
            window.pose_status.setText("视觉观测已关闭；原始 RGB-D 继续采集")
            window.last_pose_estimate = None
            if window.pose_dialog is not None:
                window.pose_dialog.canvas.estimate = None
                window.pose_dialog.canvas.update()
                window.pose_dialog.status.setText("观测已关闭，表格是最后一次历史结果")
            return
        if window.camera_source != "orbbec":
            window.pose_status.setText("三维观测需要 Gemini RGB-D 数据")
            return
        if window.pose_worker is None:
            worker = self.pose_factory(window,config=window.vision_config,vision_epoch=window.vision_epoch,target=window.person_target)
            worker.error.connect(lambda message: window._on_pose_error(message) if worker is window.pose_worker else None)
            worker.status.connect(lambda message: window.pose_status.setText(message) if worker is window.pose_worker else None)
            worker.finished.connect(window._on_pose_finished)
            window.pose_worker = worker
            worker.start()


    def _on_pose_error(self, message):
        window = self.window
        window.rgbd_pose_enabled.blockSignals(True)
        window.rgbd_pose_enabled.setChecked(False)
        window.rgbd_pose_enabled.blockSignals(False)
        window.pose_status.setText(message+"；原始采集不受影响")
        window.last_pose_estimate = None


    def _on_pose_finished(self):
        window = self.window
        worker, window.pose_worker = window.pose_worker, None
        if worker is not None:
            worker.deleteLater()
        # A quick off/on waits for the old thread to release its GPU sessions.
        if window.rgbd_pose_enabled.isChecked() and window.camera_worker is not None:
            window._set_rgbd_pose(True)


    def stop_camera(self) -> None:
        window = self.window
        if window.camera_worker is not None:
            window.rgbd_pose_enabled.setChecked(False)
            if window.camera_source == "orbbec" and window.session is not None:
                window._stop_session()
            window.camera_worker.stop()
            window.stop_camera_button.setEnabled(False)
            window.camera_status.setText("正在停止相机…")


    def _on_camera_error(self, message: str) -> None:
        window = self.window
        window._camera_failed = True
        window.camera_status.setText(message)
        if window.camera_source == "orbbec" and window.session is not None:
            window.session.error = f"RGB-D 采集中断：{message}"
            window._stop_session()


    def _on_camera_finished(self) -> None:
        window = self.window
        window.live_fusion.invalidate('相机停止/重启，需重新标定')
        window.rgbd_pose_enabled.setChecked(False)
        window.rgbd_timer.stop()
        worker = window.camera_worker
        if isinstance(worker, self.orbbec_type) and window.session is not None:
            window.session.error = window.session.error or "RGB-D 采集进程提前结束"
            window._stop_session()
        window.camera_worker = None
        if worker is not None:
            worker.deleteLater()
        window.start_camera_button.setEnabled(not window.replay_mode)
        window.stop_camera_button.setEnabled(False)
        window.camera_source_combo.setEnabled(not window.replay_mode and window.session is None)
        window.rgbd_rate.setEnabled(not window.replay_mode and window.session is None)
        if not window._camera_failed:
            window.camera_status.setText("相机已停止")

