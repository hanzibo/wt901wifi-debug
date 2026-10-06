from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from unittest.mock import patch
import sys
import time
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from PySide6 import QtCore,QtWidgets

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optusbproj_ubuntu.pose import PoseEngine,VENDOR_RPY,GYRO_TILT
from optusbproj_ubuntu.skeleton import PART_BY_KEY
from optusbproj_ubuntu.natural_hand import validate_reference
from optusbproj_ubuntu.natural_hand_gui import NaturalHandDialog
from optusbproj_ubuntu.fusion_gui import FusionWindow
from test_pose import packet
from test_pose_backend_gui import Source
from test_x1_live import prepared_engine,with_manual_left_hand,LiveRetargeter,URDF
from test_recording import packet as raw_packet
from optusbproj_ubuntu.fusion_recording import FusionSessionRecorder
from optusbproj_ubuntu.fusion_replay import FusionSessionReplay


def orientation(engine,r):
    engine.mode_current['A'][VENDOR_RPY]=r.copy();engine._sync_active('A')


class NaturalHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def engine(self):
        e=PoseEngine({'left_hand':'A'});e.feed(packet('A'))
        b=Rotation.from_euler('xyz',[13,24,-35],degrees=True).as_matrix()
        e.apply_calibration('left_hand','A',b,np.eye(3))
        return e,b

    def test_rebased_motion_times_natural_offset_and_zero_restore(self):
        e,b=self.engine();anchor=Rotation.from_euler('xy',[30,10],degrees=True).as_matrix()
        c0=Rotation.from_euler('z',-40,degrees=True).as_matrix();orientation(e,anchor)
        data=e.make_natural_hand_reference('left_hand',anchor,c0);e.set_natural_hand_reference(data)
        index=PART_BY_KEY['left_hand']
        np.testing.assert_allclose(e.transforms()[index][:3,:3],c0,atol=1e-12)
        step=Rotation.from_euler('xy',[20,-15],degrees=True).as_matrix()
        r=anchor@step;orientation(e,r)
        rebased=b@anchor;expected=rebased@step@rebased.T@c0
        np.testing.assert_allclose(e.transforms()[index][:3,:3],expected,atol=1e-12)
        standard=e.transforms(include_natural=False)[index][:3,:3]
        np.testing.assert_allclose(standard,b@r@b.T,atol=1e-12)
        e.zero('left_hand')
        np.testing.assert_allclose(e.transforms()[index][:3,:3],np.eye(3),atol=1e-12)
        self.assertFalse(e.natural_hand_references['left_hand']['enabled'])
        e.set_natural_hand_reference(data)
        np.testing.assert_allclose(e.transforms()[index][:3,:3],expected,atol=1e-12)

    def test_mapping_device_epoch_changes_invalidate_and_invalid_so3_rejected(self):
        for change in (lambda e:e.set_axis_mapping('left_hand',('+X','+Y','+Z')),
                       lambda e:e.clear_calibration('left_hand'),
                       lambda e:e.invalidate_neutral('A'),lambda e:e.assign('left_hand','B')):
            e,_=self.engine();data=e.make_natural_hand_reference('left_hand',np.eye(3),np.eye(3))
            e.set_natural_hand_reference(data);change(e)
            self.assertFalse(e.natural_hand_references)
        e,_=self.engine();data=e.make_natural_hand_reference('left_hand',np.eye(3),np.eye(3))
        data['neutral_pose'][0][0]=2
        with self.assertRaises(ValueError):e.set_natural_hand_reference(data)
        data['part']='right_hand'
        with self.assertRaises(ValueError):validate_reference(data)

    def test_runtime_roundtrip_and_old_rows_compatible(self):
        e,b=self.engine();c0=Rotation.from_euler('z',30,degrees=True).as_matrix()
        e.set_natural_hand_reference(e.make_natural_hand_reference('left_hand',np.eye(3),c0))
        snapshot=e.runtime_snapshot('A',100)
        replay,_=self.engine();replay.restore_runtime(snapshot)
        np.testing.assert_allclose(replay.transforms()[PART_BY_KEY['left_hand']][:3,:3],c0,atol=1e-12)
        snapshot.pop('natural_hand_references');old,_=self.engine();old.restore_runtime(snapshot)
        self.assertFalse(old.natural_hand_references)
        e.algorithm_modes['left_hand']=GYRO_TILT
        self.assertIn('算法',e.natural_hand_error(e.natural_hand_references['left_hand']))

    def test_x1_hand_targets_ignore_display_reference(self):
        e=with_manual_left_hand(prepared_engine());device=e.assignments['left_hand']
        e.epochs[device]='hand-epoch';e.mode_neutral[device]={VENDOR_RPY:np.eye(3)}
        e.capture_manual_standard_hand('left_hand',np.eye(3))
        retarget=LiveRetargeter(e,URDF,follow_left_hand=True)
        relative=retarget.hand_relative_rotation('left');target=retarget.absolute_hand_target('left',np.zeros(3))
        c0=Rotation.from_euler('z',80,degrees=True).as_matrix()
        e.set_natural_hand_reference(e.make_natural_hand_reference('left_hand',np.eye(3),c0))
        np.testing.assert_allclose(retarget.hand_relative_rotation('left'),relative)
        np.testing.assert_allclose(retarget.absolute_hand_target('left',np.zeros(3)),target)
        np.testing.assert_allclose(e.transforms()[PART_BY_KEY['left_hand']][:3,:3],c0,atol=1e-12)

    def test_recorded_files_replay_natural_reference(self):
        e,b=self.engine();c0=Rotation.from_euler('z',-25,degrees=True).as_matrix()
        e.set_natural_hand_reference(e.make_natural_hand_reference('left_hand',np.eye(3),c0))
        with TemporaryDirectory() as tmp:
            recorder=FusionSessionRecorder(Path(tmp),tail_trim_seconds=0,assignments=e.assignments,
                axis_mappings=e.axis_mappings,algorithm_modes=e.algorithm_modes)
            recorder.continuous_basis_at_start={'left_hand':b.tolist()}
            recorder.natural_hand_references_at_start=deepcopy(e.natural_hand_references)
            stamp=recorder.started_mono_ns+10_000_000
            recorder.write_imu(stamp,raw_packet('A',0),e.runtime_snapshot('A',stamp))
            with patch('optusbproj_ubuntu.fusion_recording.time.monotonic_ns',return_value=stamp+1_000_000):recorder.close()
            manifest=json.loads((recorder.path/'manifest.json').read_text())
            self.assertIn('left_hand',manifest['natural_hand_references_at_start'])
            source=FusionSessionReplay(recorder.path)
            try:
                rows=source.poll_timed(now=source._started_at+1)
                self.assertEqual(len(rows),1)
                runtime=source.take_runtime()[(stamp,'A')]
                replay,_=self.engine();replay.restore_runtime(runtime)
                np.testing.assert_allclose(replay.transforms()[PART_BY_KEY['left_hand']][:3,:3],c0,atol=1e-12)
            finally:source.close()

    def test_gui_sample_preview_apply_zero_resume_and_cancel(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(tmp)/'settings.json',{'left_hand':'A'},{},camera_source='orbbec')
            for timer in w.findChildren(QtCore.QTimer):timer.stop()
            e=w.engine;e.feed(packet('A'));d=NaturalHandDialog(w)
            clock=[time.monotonic()];d.confirm.setChecked(True);d.delay.setValue(0)
            with patch('optusbproj_ubuntu.natural_hand_gui.time.monotonic',side_effect=lambda:clock[0]):
                d.standard_confirm.setChecked(True);d.begin(standard=True);d.timer.stop()
                for _ in range(22):
                    clock[0]+=.1;e.received_at['A']=clock[0];d.tick()
                self.assertEqual(e.calibration_references['left_hand']['source'],'explicit_manual_standard')
                d.begin();d.timer.stop()
                for _ in range(22):
                    clock[0]+=.1;e.received_at['A']=clock[0];d.tick()
                self.assertIsNotNone(d.candidate,d.status.text());d.angle.setValue(-35)
                d.show_preview();self.assertIn('left_hand',e.natural_hand_previews)
                self.assertFalse(w._start_session())
                d.apply_reference();self.assertFalse(e.natural_hand_previews)
                self.assertIn('left_hand',e.natural_hand_references)
                e.zero();d.resume_reference();self.assertTrue(e.natural_hand_references['left_hand']['enabled'])
                d.undo_reference();self.assertFalse(e.natural_hand_references)
                d.show_preview();d.close();self.assertFalse(e.natural_hand_previews)
            w.close()

    def test_natural_zero_then_resample_uses_original_standard_pair(self):
        e,b=self.engine()
        n=Rotation.from_euler('xyz',[18,-12,35],degrees=True).as_matrix()
        e.apply_calibration('left_hand','A',b,n)
        rc=n@Rotation.from_euler('z',-40,degrees=True).as_matrix()
        orientation(e,rc);e.zero('left_hand')
        np.testing.assert_allclose(e.neutral_for('A',VENDOR_RPY),rc)
        np.testing.assert_allclose(e.calibration_references['left_hand']['neutral_by_mode'][VENDOR_RPY],n)
        c0=Rotation.from_euler('z',-40,degrees=True).as_matrix()
        ref=e.make_natural_hand_reference('left_hand',rc,c0)
        expected_basis=b@n.T@rc
        np.testing.assert_allclose(ref['basis'],expected_basis,atol=1e-12)
        e.set_natural_hand_reference(ref)
        step=Rotation.from_euler('x',30,degrees=True).as_matrix();orientation(e,rc@step)
        actual=e.transforms()[PART_BY_KEY['left_hand']][:3,:3]
        np.testing.assert_allclose(actual,expected_basis@step@expected_basis.T@c0,atol=1e-12)
        e.zero();again=e.make_natural_hand_reference('left_hand',rc,c0)
        np.testing.assert_allclose(again['basis'],expected_basis,atol=1e-12)
        replay,_=self.engine();replay.restore_runtime(e.runtime_snapshot('A',100))
        np.testing.assert_allclose(replay.calibration_references['left_hand']['neutral_by_mode'][VENDOR_RPY],n)
        np.testing.assert_allclose(replay.make_natural_hand_reference('left_hand',rc,c0)['basis'],expected_basis,atol=1e-12)

    def test_missing_standard_reference_never_uses_last_zero(self):
        e=PoseEngine({'left_hand':'A'});e.feed(packet('A',yaw=40));e.zero()
        with self.assertRaisesRegex(ValueError,'独立标准'):e.make_natural_hand_reference('left_hand',e.current['A'],np.eye(3))
        standard=Rotation.from_euler('xyz',[10,15,20],degrees=True).as_matrix()
        e.capture_manual_standard_hand('left_hand',standard)
        e.zero();ref=e.make_natural_hand_reference('left_hand',np.eye(3),np.eye(3))
        np.testing.assert_allclose(ref['standard_neutral'],standard)


if __name__=='__main__':unittest.main()
