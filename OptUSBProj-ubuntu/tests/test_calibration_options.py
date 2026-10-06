from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from PySide6 import QtWidgets
from test_guided_calibration import _samples, _torso_samples, _rows, MANUAL
from optusbproj_ubuntu.guided_calibration import fit_guided_arm, fit_guided_torso
from optusbproj_ubuntu.calibration_options import validate_prior_weight, capture_prior_weight, add_prior_control
from optusbproj_ubuntu.settings import save_config, load_prior_weight, load_config


class PriorWeightTests(unittest.TestCase):
    def test_zero_ignores_manual_mapping_arm_and_torso(self):
        for fitter, samples in ((lambda s,m:fit_guided_arm('right_upper_arm','D',s,m,prior_weight=0), _samples()[0]),
                                (lambda s,m:fit_guided_torso('D',s,m,prior_weight=0), _torso_samples()[0])):
            a=fitter(samples,MANUAL);b=fitter(samples,('+X','+Y','+Z'))
            np.testing.assert_allclose(a.basis_matrix,b.basis_matrix,atol=1e-12)
            self.assertEqual(a.prior_weight,0)
            with patch('optusbproj_ubuntu.guided_calibration.mapping_matrix',side_effect=AssertionError('prior used')):
                fitter(samples,MANUAL)

    def test_default_is_unchanged(self):
        s,_=_samples()
        a=fit_guided_arm('right_upper_arm','D',s,MANUAL)
        b=fit_guided_arm('right_upper_arm','D',s,MANUAL,prior_weight=.15)
        self.assertEqual(a,b)

    def test_validation_and_recorded_default(self):
        for value in (-1,2.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):validate_prior_weight(value)
        self.assertEqual(capture_prior_weight({}),.15)
        self.assertEqual(capture_prior_weight({'wearing_prior_weight':0}),0)
        self.assertEqual(capture_prior_weight({'wearing_prior_weight':0},.4),.4)

    def test_evidence_prior_cancellation_rejected(self):
        # Right front +X and side -Y yield C=2,S=0; prior yaw pi cancels at lambda=2.
        from scipy.spatial.transform import Rotation
        neutral=np.eye(3)
        front=Rotation.from_euler('y',-90,degrees=True).as_matrix()
        side=Rotation.from_euler('x',-90,degrees=True).as_matrix()
        samples={'neutral':_rows(neutral),'forward':_rows(front),'lateral':_rows(side),
                 'check_forward':_rows(front),'check_neutral':_rows(neutral)}
        with self.assertRaisesRegex(ValueError,'抵消'):
            fit_guided_arm('right_upper_arm','D',samples,('-X','-Y','+Z'),prior_weight=2)

    def test_config_survives_binding_mapping_and_mode_saves(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'local.json'
            self.assertEqual(load_prior_weight(p),.15)
            save_config(p,{}, {},prior_weight=0)
            save_config(p,{'right_upper_arm':'D'}, {'right_upper_arm':MANUAL})
            self.assertEqual(load_prior_weight(p),0)
            self.assertEqual(load_config(p)[0]['right_upper_arm'],'D')

    def test_control_zero_reset_and_frozen_spin(self):
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from types import SimpleNamespace
        owner=SimpleNamespace(wearing_prior_weight=.15)
        d=QtWidgets.QDialog();layout=QtWidgets.QVBoxLayout(d);layout.addWidget(QtWidgets.QLabel('Options'))
        spin=add_prior_control(d,owner)
        buttons=d.findChildren(QtWidgets.QPushButton)
        buttons[0].click();self.assertEqual(spin.value(),0)
        buttons[1].click();self.assertEqual(spin.value(),.15)
        spin.setEnabled(False);buttons[0].click();self.assertEqual(spin.value(),.15)
        d.close()

    def test_recording_snapshot_and_explicit_override_reach_offline_fit(self):
        import json
        from test_combined_calibration import capture
        from optusbproj_ubuntu.combined_calibration import imu_wearing
        with TemporaryDirectory() as tmp:
            p=Path(tmp);capture(p)
            guide=json.loads((p/'camera_guide.json').read_text())
            guide['wearing_prior_weight']=0
            (p/'camera_guide.json').write_text(json.dumps(guide))
            report=imu_wearing(p)
            self.assertEqual(report['prior_weight'],0)
            self.assertTrue(all(r['prior_weight']==0 for r in report['parts'].values()))
            override=imu_wearing(p,prior_weight=.4)
            self.assertEqual(override['prior_weight'],.4)
            self.assertTrue(all(r['prior_weight']==.4 for r in override['parts'].values()))

    def test_visual_primary_fit_unchanged_by_prior_weight(self):
        from test_visual_wearing import fixture
        from optusbproj_ubuntu.visual_wearing import analyze_wearing
        with TemporaryDirectory() as tmp:
            p=Path(tmp);fixture(p)
            a=analyze_wearing(p,prior_weight=0);b=analyze_wearing(p,prior_weight=.4)
            for part in a['parts']:
                np.testing.assert_allclose(a['parts'][part]['basis_matrix'],b['parts'][part]['basis_matrix'])
                self.assertEqual(a['parts'][part]['legacy_fit']['prior_weight'],0)
                self.assertEqual(b['parts'][part]['legacy_fit']['prior_weight'],.4)

    def test_gui_shared_settings_freeze_and_analysis_arguments(self):
        from optusbproj_ubuntu.fusion_gui import FusionWindow
        from optusbproj_ubuntu.calibration_gui import GuidedCalibrationDialog, BilateralCalibrationDialog
        from optusbproj_ubuntu.combined_calibration_gui import CombinedCalibrationDialog
        from optusbproj_ubuntu.visual_wearing_gui import VisualWearingDialog
        from optusbproj_ubuntu.calibration_options import freeze_prior_control, set_prior_enabled
        from test_pose_backend_gui import Source
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with TemporaryDirectory() as tmp:
            w=FusionWindow(Source(),Path(tmp)/'config.json',{}, {},camera_source='orbbec');w.poll_timer.stop();w.rgbd_timer.stop()
            dialogs=[cls(w) for cls in (GuidedCalibrationDialog,BilateralCalibrationDialog,CombinedCalibrationDialog,VisualWearingDialog)]
            try:
                dialogs[0].prior_weight_spin.setValue(0)
                self.assertTrue(all(d.prior_weight_spin.value()==0 for d in dialogs))
                freeze_prior_control(dialogs[0]);w.set_wearing_prior_weight(.4)
                self.assertEqual(dialogs[0].capture_prior_weight,0)
                self.assertEqual(dialogs[0].prior_weight_spin.value(),0)
                self.assertFalse(dialogs[0].prior_weight_spin.prior_buttons[0].isEnabled())
                set_prior_enabled(dialogs[0],True)
                self.assertEqual(dialogs[0].prior_weight_spin.value(),.4)
                for d in dialogs[2:]:
                    d.prior_weight_spin.setValue(0);d.state='complete';d.recording_path=Path(tmp)/'capture'
                    with patch('PySide6.QtCore.QProcess') as factory:
                        d.analyze_recording()
                        args=factory.return_value.setArguments.call_args.args[0]
                        self.assertEqual(args[args.index('--prior-weight')+1],'0.0')
                    d.process=None;d.state='complete'
            finally:
                for d in dialogs:d.close()
                w.close()
