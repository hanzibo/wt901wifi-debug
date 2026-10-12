"""Recording coordination for the fusion window.

State stays on the window for dialog/replay compatibility; no GUI imports.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np


class FusionRecordingController:
    def __init__(self, window, *, recorder_factory, orbbec_type, warn,
                 clock=time.monotonic, clock_ns=time.monotonic_ns):
        self.window = window
        self.recorder_factory = recorder_factory
        self.orbbec_type = orbbec_type
        self.warn = warn
        self.clock = clock
        self.clock_ns = clock_ns

    def toggle_session(self) -> None:
        window = self.window
        if window.camera_calibration_dialog is not None and window.camera_calibration_dialog.active:
            window.statusBar().showMessage("相机标定正在录制，请在标定窗口中取消本轮。")
            return
        if window.session is not None:
            window._stop_session()
            return
        window._cancel_record_schedule()
        window._start_session()


    def _start_session(self) -> bool:
        window = self.window
        if window.palm_visual_dialog is not None and window.palm_visual_dialog.active:
            window.record_schedule_status.setText('请先完成左手视觉C0采集，再开始联合录制');return False
        if window.engine.natural_hand_previews or (window.natural_hand_dialog is not None and window.natural_hand_dialog.active):
            window.record_schedule_status.setText('请先应用或取消自然掌面预览，并结束采样，再录制');return False
        if window.camera_source == "orbbec" and (
                window.last_rgbd_observation is None or window.camera_worker is None
                or self.clock_ns() - window.last_rgbd_observation.captured_mono_ns > 1_000_000_000):
            window.record_schedule_status.setText("RGB-D 尚未就绪，请先启动相机并等待预热完成")
            return False
        default_parent = Path(__file__).resolve().parent.parent / "recordings"
        try:
            window.session = self.recorder_factory(
                default_parent, max_pending_frames=16 if window.camera_source == "orbbec" else 64,
                assignments=window.engine.assignments,
                axis_mappings=window.engine.axis_mappings,
                algorithm_modes=window.engine.algorithm_modes)
        except OSError as exc:
            self.warn("无法开始录制", str(exc))
            return False
        window.session_button.setText("结束联合录制")
        window.session.continuous_basis_at_start={key:value.tolist() for key,value in window.engine.continuous_basis.items()}
        from copy import deepcopy
        window.session.natural_hand_references_at_start=deepcopy(window.engine.natural_hand_references)
        window.session.calibration_references_at_start=deepcopy(window.engine.calibration_references)
        window.session.camera_epoch=getattr(window.camera_worker,'capture_epoch',None)
        window.session.vision_config=window.vision_metadata;window.session.vision_epoch=window.vision_epoch
        window.session.person_target_at_start=(None if window.last_pose_estimate is None else
            {k:window.last_pose_estimate.diagnostics.get(k) for k in
             ('target_id','target_mode','target_state','selected_index')})
        window.session.heading_max_offset_deg=window.live_fusion.max_offset_deg
        window.session.fusion_at_start={part:{'calibration':data,'offset_deg':float(np.degrees(corrector.offset_rad))} for part,(data,corrector) in window.live_fusion.bindings.items()}
        window._fusion_events_file=(window.session.path/'fusion_events.jsonl').open('w',encoding='utf-8')
        window.camera_source_combo.setEnabled(False)
        window.rgbd_rate.setEnabled(False)
        if isinstance(window.camera_worker, self.orbbec_type):
            window.session.camera_calibration = window.camera_worker.calibration
            window._bridge_skipped_start = window.camera_worker.consumer_skipped
        window.record_schedule_status.setText("录制中；结束时末尾 3 秒不保存")
        window.statusBar().showMessage(f"正在录制：{window.session.path}")
        return True


    def _start_timed_session(self) -> None:
        window = self.window
        if window.camera_calibration_dialog is not None and window.camera_calibration_dialog.active:
            return
        if window.replay_mode or window.session is not None or window._record_start_deadline is not None:
            return
        window._record_start_deadline = self.clock() + window.record_delay.value()
        window.timed_session_button.setEnabled(False)
        window.cancel_session_button.setEnabled(True)
        window.record_delay.setEnabled(False)
        window.record_duration.setEnabled(False)
        window.record_schedule_timer.start()
        window._update_record_schedule()


    def _cancel_record_schedule(self) -> None:
        window = self.window
        if window._record_start_deadline is None:
            return
        window._record_start_deadline = None
        window.record_schedule_timer.stop()
        window.timed_session_button.setEnabled(not window.replay_mode)
        window.cancel_session_button.setEnabled(False)
        window.record_delay.setEnabled(True)
        window.record_duration.setEnabled(True)
        window.record_schedule_status.setText("定时录制已取消；没有创建录制文件")


    def _update_record_schedule(self) -> None:
        window = self.window
        now = self.clock()
        if window._record_start_deadline is not None:
            remaining = window._record_start_deadline - now
            if remaining > 0:
                window.record_schedule_status.setText(
                    f"还有 {remaining:.1f} 秒开始录制；可离开电脑摆好姿势")
                return
            window._record_start_deadline = None
            window.cancel_session_button.setEnabled(False)
            if not window._start_session():
                window.record_schedule_timer.stop()
                window.timed_session_button.setEnabled(not window.replay_mode)
                window.record_delay.setEnabled(True)
                window.record_duration.setEnabled(True)
                window.record_schedule_status.setText("定时录制未能启动")
                return
            window._record_stop_deadline = now + window.record_duration.value()
        if window._record_stop_deadline is not None:
            remaining = window._record_stop_deadline - now
            if remaining <= 0:
                window._stop_session(auto=True)
            else:
                window.record_schedule_status.setText(
                    f"录制中；{remaining:.1f} 秒后自动结束，末尾 3 秒不保存")


    def _stop_session(self, *, auto: bool = False) -> None:
        window = self.window
        window._record_start_deadline = None
        window._record_stop_deadline = None
        window.record_schedule_timer.stop()
        window.timed_session_button.setEnabled(not window.replay_mode)
        window.cancel_session_button.setEnabled(False)
        window.record_delay.setEnabled(True)
        window.record_duration.setEnabled(True)
        session, window.session = window.session, None
        if window._fusion_events_file is not None:
            window._fusion_events_file.close();window._fusion_events_file=None
        if session is None:
            return
        if isinstance(window.camera_worker, self.orbbec_type):
            session.camera_bridge_skipped = window.camera_worker.consumer_skipped - window._bridge_skipped_start
        window.camera_source_combo.setEnabled(not window.replay_mode and window.camera_worker is None)
        window.rgbd_rate.setEnabled(not window.replay_mode and window.camera_worker is None)
        window.session_button.setText("开始联合录制")
        try:
            session.close()
        except OSError as exc:
            window.record_schedule_status.setText(f"录制不完整：{exc}")
            window.statusBar().showMessage(f"录制不完整：{exc}；目录 {session.path}")
        else:
            import json
            manifest_path=session.path/'manifest.json'
            event_path=session.path/'fusion_events.jsonl'
            if manifest_path.is_file() and event_path.is_file():
                try:
                    manifest=json.loads(manifest_path.read_text())
                    with event_path.open() as stream:rows=[json.loads(line) for line in stream]
                    rows=[r for r in rows if manifest['started_mono_ns']<=r['time_ns']<=manifest['retained_until_mono_ns']]
                    event_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
                except (OSError,ValueError) as exc:
                    window.statusBar().showMessage('纠偏事件整理失败，原始数据保留：'+str(exc))
            window.record_schedule_status.setText(
                ("定时录制完成" if auto else "录制已结束")
                + f"；末尾 3 秒已剔除（IMU {session.trimmed_imu_packets} 包、"
                f"相机 {session.trimmed_camera_frames} 帧）")
            window.statusBar().showMessage(
                f"已录制 {session.imu_count} 包 / {session.frame_count} 帧"
                f"（丢 {session.dropped_frames} 帧）：{session.path}", 10000)

