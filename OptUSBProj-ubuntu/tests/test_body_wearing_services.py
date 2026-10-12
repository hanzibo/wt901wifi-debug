"""Headless main-window coordination and wearing phase-transition regression."""
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock
import subprocess
import sys
import numpy as np

from optusbproj_ubuntu.body_device_controller import BodyDeviceController
from optusbproj_ubuntu.body_view import BodyView
from optusbproj_ubuntu.wearing_workflow import GuidedWearingWorkflow
from optusbproj_ubuntu.bilateral_wearing_workflow import BilateralWearingWorkflow
from optusbproj_ubuntu.pose import PoseEngine, GYRO_TILT, VENDOR_RPY


class BodyWearingServiceTests(TestCase):
    def test_workflow_and_device_imports_are_qt_free(self):
        code=('import sys; from optusbproj_ubuntu import body_device_controller, '
              'wearing_workflow, bilateral_wearing_workflow, wearing_protocol; '
              'assert not any(k.startswith("PySide6") for k in sys.modules); '
              'assert "optusbproj_ubuntu.gui" not in sys.modules; '
              'assert "optusbproj_ubuntu.calibration_gui" not in sys.modules')
        subprocess.run([sys.executable,'-c',code],check=True)

    def device(self, host, save=None, mapping=None):
        return BodyDeviceController(host, save=save or Mock(),
            mapping_dialog_factory=mapping or Mock(),warn=Mock(),accepted_code=1)

    def host(self):
        engine=PoseEngine({'right_upper_arm':'A'})
        return NS(engine=engine,settings_path='settings.json',calibration_dialog=None,
                  replay_mode=False,recorder=None,_render_dirty=False,
                  _refresh_choices=Mock(),_update_mapping_button=Mock(),
                  statusBar=Mock(return_value=Mock()))

    def test_failed_assignment_save_does_not_change_binding(self):
        w=self.host();w.combos={'right_upper_arm':Mock(currentData=lambda:'B')}
        c=self.device(w,save=Mock(side_effect=OSError('disk')))
        c._assignment_changed('right_upper_arm')
        self.assertEqual(w.engine.assignments['right_upper_arm'],'A')
        c.warn.assert_called_once();w._refresh_choices.assert_called_once_with(force=True)

    def test_successful_assignment_invalidates_old_calibration_result(self):
        w=self.host();w.combos={'right_upper_arm':Mock(currentData=lambda:'B')}
        w.calibration_dialog=Mock();c=self.device(w)
        c._assignment_changed('right_upper_arm')
        self.assertEqual(w.engine.assignments['right_upper_arm'],'B')
        w.calibration_dialog.invalidate_for.assert_called_once_with('A')
        self.assertTrue(w._render_dirty)

    def test_reconnect_threshold_preserved_and_replay_does_not_invalidate(self):
        w=self.host();w.engine.received_at['A']=10.
        w.engine.continuous_basis['right_upper_arm']=np.eye(3)
        w.calibration_dialog=Mock();c=self.device(w)
        c._check_calibration_reconnect('A',15.)
        self.assertIn('right_upper_arm',w.engine.continuous_basis)
        c._check_calibration_reconnect('A',15.001)
        self.assertNotIn('right_upper_arm',w.engine.continuous_basis)
        w.calibration_dialog.invalidate_for.assert_called_once_with('A')
        w.engine.continuous_basis['right_upper_arm']=np.eye(3);w.replay_mode=True
        c._check_calibration_reconnect('A',30.)
        self.assertIn('right_upper_arm',w.engine.continuous_basis)

    def test_algorithm_switch_during_recording_rolls_back(self):
        w=self.host();combo=Mock(currentData=lambda:GYRO_TILT)
        w.algorithm_combos={'right_upper_arm':combo};w.recorder=object();c=self.device(w)
        c._algorithm_changed('right_upper_arm')
        self.assertEqual(w.engine.algorithm_modes['right_upper_arm'],VENDOR_RPY)
        c.save.assert_not_called();c.warn.assert_called_once()
        combo.blockSignals.assert_any_call(True);combo.blockSignals.assert_any_call(False)

    def test_mapping_cancel_does_not_save_or_apply(self):
        w=self.host();mapping=Mock(return_value=Mock(exec=lambda:0));c=self.device(w,mapping=mapping)
        before=w.engine.axis_mappings['right_upper_arm']
        c._show_axis_mapping('right_upper_arm')
        self.assertEqual(w.engine.axis_mappings['right_upper_arm'],before)
        c.save.assert_not_called();w._update_mapping_button.assert_not_called()

    def workflow(self, dialog, cls=GuidedWearingWorkflow):
        return cls(dialog,recorder_factory=Mock(),fit_arm=Mock(),fit_torso=Mock(),
                   beep=Mock(),warn=Mock(),question=Mock(),yes_button=1,cancel_button=2,
                   clock=lambda:10.,clock_ns=lambda:10_000_000_000)

    def dialog(self):
        return NS(host=NS(engine=Mock(is_live=lambda _:True)),phase='prepare',
                  deadline=9.,index=0,part='right_upper_arm',device_id='A',
                  steps=('neutral','forward'),samples={'neutral':[],'forward':[]},
                  events=[],hold_seconds=Mock(value=lambda:4),
                  prepare_seconds=Mock(value=lambda:3),progress=Mock(),
                  _tick=Mock(),_finish=Mock(),abort=Mock())

    def test_guided_prepare_to_hold_records_same_time_and_beeps(self):
        d=self.dialog();w=self.workflow(d);w._tick()
        self.assertEqual(d.phase,'hold');self.assertEqual(d.deadline,14.)
        self.assertEqual(d.events,[{'step':'neutral','hold_start_mono_ns':10_000_000_000}])
        w.beep.assert_called_once();d._tick.assert_called_once()

    def test_guided_hold_requires_thirty_samples(self):
        d=self.dialog();d.phase='hold';d.events=[{'step':'neutral'}]
        d.samples['neutral']=[{}]*29;w=self.workflow(d);w._tick()
        d.abort.assert_called_once();self.assertEqual(d.index,0);d._finish.assert_not_called()
        d.abort.reset_mock();d.samples['neutral'].append({});w._tick()
        self.assertEqual(d.index,1);self.assertEqual(d.phase,'prepare')
        self.assertEqual(d.deadline,13.);d.abort.assert_not_called()

    def test_guided_final_hold_finishes_without_restart(self):
        d=self.dialog();d.phase='hold';d.index=1;d.events=[{'step':'forward'}]
        d.samples['forward']=[{}]*30;w=self.workflow(d);w._tick()
        d._finish.assert_called_once();d._tick.assert_not_called()

    def test_bilateral_all_streams_lost_aborts_but_one_live_continues(self):
        d=self.dialog();d._participants={'right_upper_arm':'A','right_forearm':'B'}
        d.samples_by_part={key:{'neutral':[]} for key in d._participants}
        d.host.engine.is_live=lambda _:False;w=self.workflow(d,BilateralWearingWorkflow)
        w._tick();d.abort.assert_called_once()
        d.abort.reset_mock();d.host.engine.is_live=lambda device:device=='A'
        w._tick();d.abort.assert_not_called();self.assertEqual(d.phase,'hold')

    def test_bilateral_abort_runs_base_cleanup_once(self):
        d=self.dialog();d.timer=Mock();d.part_combo=Mock();d.read_button=Mock()
        d.start_button=Mock();d.cancel_button=Mock();d.recorder=Mock(path='recording')
        recorder=d.recorder;d._save_events=Mock();d.device_checks={'A':(1,1)}
        d.prior_weight_spin=Mock(value=lambda:.15,prior_buttons=[])
        d.prior_owner=NS(wearing_prior_weight=.15);w=self.workflow(d,BilateralWearingWorkflow)
        w.abort('cancel');w.abort('again')
        self.assertEqual(d.phase,'idle');self.assertIsNone(d.recorder)
        recorder.close.assert_called_once();d._save_events.assert_called_once_with('recording','cancelled')
        d.part_combo.setEnabled.assert_any_call(False)

    def test_view_transforms_and_visibility_still_use_existing_actors(self):
        matrix=np.eye(4);matrix[0,3]=.2;actor=NS();axis=Mock()
        w=NS(actors=[actor],axis_actors=[axis],engine=Mock(transforms=lambda:[matrix]),
             plotter=Mock(),_render_dirty=True)
        view=BodyView(w,model_dir='models',triad_factory=Mock())
        view._render_pose();self.assertIs(actor.user_matrix,matrix)
        self.assertIs(axis.user_matrix,matrix);self.assertFalse(w._render_dirty)
        w.plotter.render.assert_called_once()
        view._toggle_axes(False);axis.SetVisibility.assert_called_once_with(False)
        self.assertTrue(w._render_dirty)
