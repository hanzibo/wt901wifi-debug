"""Calibration coordination for the fusion window.

State stays on the window for dialog/replay compatibility; no GUI imports.
"""
from __future__ import annotations

import time
import numpy as np


class FusionCalibrationController:
    def __init__(self, window, *, clock_ns=time.monotonic_ns):
        self.window = window
        self.clock_ns = clock_ns

    def apply_camera_report(self,report):
        window = self.window
        if window.replay_mode or window.session is not None:raise ValueError('回放或录制中不能应用实时标定')
        if window.x1_preview_dialog is not None and window.x1_preview_dialog.sender is not None:raise ValueError('请先停止X1随动，再应用标定')
        if window.camera_worker is None or report.get('camera_epoch') is None or report['camera_epoch']!=getattr(window.camera_worker,'capture_epoch',None):raise ValueError('相机采集会话已变化，需重新标定')
        window._check_vision_report(report)
        window.live_fusion.install(report);window._live_camera_epoch=report['camera_epoch']
        window._person_restore_vision_signature=window.vision_metadata.get('signature')
        window._update_live_fusion_status()


    def prepare_camera_gravity(self,devices,start_ns,end_ns):
        window = self.window
        return window.live_fusion.lock_gravity(devices,start_ns,end_ns)


    def _check_vision_report(self,report):
        window = self.window
        config=report.get('vision_config')
        if config is None:
            if window.vision_config.backend!='rtmw' or window._vision_changed:raise ValueError('旧报告没有二维模型参考信息，需重新标定')
        elif config.get('signature')!=window.vision_metadata['signature'] or report.get('vision_epoch')!=window.vision_epoch:
            raise ValueError('二维模型/权重/输入尺寸或视觉会话已变化，需重新标定')


    def apply_visual_wearing_report(self, report, parts, *, review_acknowledged=False):
        window = self.window
        from .visual_wearing import apply_wearing_report
        if window.replay_mode or window.session is not None:
            raise ValueError('回放或录制期间不能应用穿戴标定')
        if window.x1_preview_dialog is not None and window.x1_preview_dialog.sender is not None:
            raise ValueError('请先停止X1随动')
        epoch = getattr(window.camera_worker, 'capture_epoch', None)
        window._check_vision_report(report)
        backup = {}
        for part in parts:
            device = window.engine.assignments.get(part)
            backup[part] = {'device_id': device, 'epoch': window.engine.epochs.get(device),
                            'basis': window.engine.continuous_basis.get(part),
                            'neutrals': {m: r.copy() for m, r in window.engine.mode_neutral.get(device, {}).items()},
                            'applied_basis': report['parts'][part].get('basis_matrix')}
        apply_wearing_report(window.engine, report, epoch, parts, review_acknowledged=review_acknowledged)
        window.live_fusion.events.append({'type': 'wearing_application', 'time_ns': self.clock_ns(),
            'parts': list(parts), 'source_session': report.get('session'),
            'acceptance_revision': report.get('acceptance_revision', 1),
            'review_acknowledged': bool(review_acknowledged)})
        window._render_dirty = True
        window.statusBar().showMessage('已应用视觉辅助穿戴映射；未启用相机纠偏或机器人', 8000)
        return backup


    def restore_visual_wearing(self, backup):
        window = self.window
        if window.replay_mode or window.session is not None:
            raise ValueError('回放或录制期间不能撤销')
        if window.x1_preview_dialog is not None and window.x1_preview_dialog.sender is not None:
            raise ValueError('请先停止X1随动')
        for part, data in backup.items():
            device = data['device_id']
            if window.engine.assignments.get(part) != device or window.engine.epochs.get(device) != data['epoch']:
                raise ValueError('绑定或积分参考系已变化，不能恢复旧中立位')
            current = window.engine.continuous_basis.get(part)
            if current is None or not np.allclose(current, data['applied_basis']):
                raise ValueError('映射已由其他操作修改，不能覆盖，请重新标定')
        for part, data in backup.items():
            if data['basis'] is None:
                window.engine.clear_calibration(part)
            else:
                window.engine.set_continuous_basis(part, data['basis'])
            window.engine.mode_neutral[data['device_id']] = {m: r.copy() for m, r in data['neutrals'].items()}
            window.engine._sync_active(data['device_id'])
        window._render_dirty = True

