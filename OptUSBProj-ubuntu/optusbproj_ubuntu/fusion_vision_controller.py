"""Vision coordination for the fusion window.

State stays on the window for dialog/replay compatibility; no GUI imports.
"""
from __future__ import annotations

import time
from .pose_2d import Pose2DConfig


class FusionVisionController:
    def __init__(self, window, *, clock_ns=time.monotonic_ns):
        self.window = window
        self.clock_ns = clock_ns

    def _pose_change_error(self):
        window = self.window
        if window.palm_visual_dialog is not None and window.palm_visual_dialog.active:return '请先结束左手视觉C0采集'
        if window.engine.natural_hand_previews or (window.natural_hand_dialog is not None and window.natural_hand_dialog.active):return '请先结束自然掌面采样/临时预览'
        if window.replay_mode:return '回放显示录制时的模型结果，不能切换推理后端'
        if window.session is not None or window._record_start_deadline is not None:return '录制或预约期间不能切换二维模型'
        for dialog in (window.camera_calibration_dialog,window._heading_calibration_dialog,window._visual_wearing_dialog,
                       window._combined_calibration_dialog,window._reuse_calibration_dialog):
            if dialog is not None and (dialog.active or dialog.state=='analyzing'):return '标定或分析期间不能切换二维模型'
        if window.x1_preview_dialog is not None and window.x1_preview_dialog.sender is not None:return '请先停止X1随动再切换二维模型'
        if window.calibration_dialog is not None and window.calibration_dialog.isVisible():return '请先关闭IMU穿戴标定窗口'
        return None


    def _apply_person_target(self,anchor,mode):
        window = self.window
        error=window._pose_change_error()
        if error:window.person_status.setText('人物未切换：'+error);return
        if window.camera_source!='orbbec':return
        import uuid
        window.vision_epoch=uuid.uuid4().hex;window._vision_changed=True
        from .live_rgbd_fusion import PERSON_SELECTION_INVALIDATION
        window.live_fusion.invalidate(PERSON_SELECTION_INVALIDATION)
        window.person_restore_ack.setChecked(False)
        if window.pose_worker is not None:
            window.pose_worker.select_target(anchor,mode,window.vision_epoch)
        else:window.person_target.reset(anchor,mode=mode,selection_id=window.vision_epoch)
        window.last_pose_estimate=None;window._person_candidates=[];window._person_frame_ns=None
        window.person_combo.clear()
        if window.pose_dialog is not None:
            window.pose_dialog.canvas.estimate=None;window.pose_dialog.canvas.update()
        window.person_status.setText('人物选择已更新；等待新帧。穿戴映射保留，原相机纠偏停用，请重新相机标定。')
        window._sync_pose_controls();window._update_live_fusion_status()


    def _person_restore_error(self):
        window = self.window
        error=window._pose_change_error()
        if error:return error
        if window.camera_source!='orbbec' or window.camera_worker is None:return '请先启动原相机采集会话'
        if getattr(window.camera_worker,'capture_epoch',None)!=window._live_camera_epoch:return '相机采集会话已改变，需重新标定'
        if not window.rgbd_pose_enabled.isChecked() or window.pose_worker is None:return '请先开启三维视觉观测'
        item=window.person_combo.currentData()
        if item is None or not 0<=self.clock_ns()-item['captured_mono_ns']<=1_000_000_000:return '请选择当前彩色画面中的原标定者，等待新人物框'
        try:
            parts=window.live_fusion.validate_person_restore()
            if any(not window.engine.is_live(window.live_fusion.bindings[p][0]['device_id'],max_age=1.) for p in parts):
                return '原标定设备数据不新鲜，请先恢复连接'
        except (ValueError,KeyError) as exc:return str(exc)
        return None


    def _restore_original_person(self):
        window = self.window
        error=window._person_restore_error()
        if error or not window.person_restore_ack.isChecked():
            window.person_status.setText('未恢复：'+(error or '请先确认原标定者与安装未移动'));return
        item=window.person_combo.currentData().copy()
        recheck=(window.live_fusion.model_changed or window.vision_metadata.get('signature')!=window._person_restore_vision_signature)
        parts=window.live_fusion.begin_model_recheck() if recheck else window.live_fusion.restore_original_person()
        # Keep the camera calibration but create a new visual selection epoch:
        # pending frames from the prior person must never enter feedback.
        import uuid
        window.vision_epoch=uuid.uuid4().hex;window._vision_changed=True
        window.pose_worker.select_target(item,'manual',window.vision_epoch)
        window.last_pose_estimate=None;window._person_candidates=[];window._person_frame_ns=None
        window.person_combo.clear();window.person_restore_ack.setChecked(False)
        if window.pose_dialog is not None:
            window.pose_dialog.canvas.estimate=None;window.pose_dialog.canvas.update()
        window.person_status.setText(f'已确认原标定者，恢复 {len(parts)} 个部位的标定资格；补偿角保持，等待新骨架后按原质量门控继续纠偏。')
        if recheck:window.person_status.setText('已确认原标定者；请在15秒内斜前举/侧举，使参与骨段清晰可见。逐部位检查新模型方向一致性，通过后恢复纠偏；未通过项保持停用。')
        window._sync_pose_controls();window._update_live_fusion_status()


    def _change_pose_config(self,_index=None,*,model=None):
        window = self.window
        error=window._pose_change_error()
        if error:window._sync_pose_controls();window.pose_status.setText(error);return
        new=Pose2DConfig(window.pose_backend_combo.currentData(),model or window.vision_config.model,window.pose_imgsz_combo.currentData())
        metadata=new.metadata()
        if metadata['signature']==window.vision_metadata['signature']:
            window.vision_config=new;window._sync_pose_controls();return
        import uuid
        from .live_rgbd_fusion import MODEL_SELECTION_INVALIDATION
        window.live_fusion.invalidate(MODEL_SELECTION_INVALIDATION)
        window.person_restore_ack.setChecked(False)
        window._person_candidates=[];window.person_combo.clear()
        window.vision_config=new;window.vision_metadata=metadata;window.vision_epoch=uuid.uuid4().hex;window._vision_changed=True
        window.last_pose_estimate=None
        if window.pose_dialog is not None:
            window.pose_dialog.canvas.estimate=None;window.pose_dialog.canvas.update()
            window.pose_dialog.status.setText('二维模型已切换；表格为旧模型历史结果，等待新观测')
        if window.pose_worker is not None:window.pose_worker.stop()
        elif window.rgbd_pose_enabled.isChecked():window._set_rgbd_pose(True)
        window._sync_pose_controls()
        window.pose_status.setText('模型已切换：保留标定与修正，暂停纠偏；加载完成后请选择原标定者并确认恢复，进行短时方向一致性检查')

