"""Actual Qt button + observer subprocess; never opens a robot port."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QWidget
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog

APP = QApplication.instance() or QApplication([])


def wait_for(predicate, timeout=8):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        APP.processEvents()
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('observer did not complete in time')


class RuntimeGuiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.viewer=QWidget(); self.viewer.engine=None; self.viewer.replay_mode=False
        self.dialog=RealRobotModesDialog(self.viewer)
        self.dialog.timer.stop()
        self.dialog.mode_select.setCurrentIndex(2)
        self.session=self.dialog.runtime_recording
        self.session.output_root=Path(self.temp.name)

    def tearDown(self):
        self.dialog.worker=None; self.dialog.busy=False
        if self.session.running:
            self.session.stop()
            wait_for(lambda:not self.session.running)
        self.dialog.close(); self.viewer.close(); APP.processEvents()
        self.temp.cleanup()

    def test_click_samples_links_control_and_does_not_change_motion(self):
        d=self.dialog; session=self.session
        before=(d.frequency.value(),d.safety.isChecked(),d.manual.isChecked())
        with patch('optusbproj_ubuntu.real_robot_gui.ElbowSerialWorker') as serial:
            session.button.click()
            wait_for(lambda:session.button.isEnabled() and session.running)
            first=session.output
            path=first/'samples.jsonl'
            wait_for(lambda:path.exists() and '"event": "sample"' in path.read_text())
            d.worker=SimpleNamespace(recorder=SimpleNamespace(path=Path('/fake/control.jsonl')))
            d.busy=True
            session.tick()
            session.button.click()
            wait_for(lambda:not session.running)
            serial.assert_not_called()
        self.assertTrue(d.busy)  # Stopping the observer is NOT stopping motion.
        self.assertEqual(before,(d.frequency.value(),d.safety.isChecked(),d.manual.isChecked()))
        result=json.loads((first/'summary.json').read_text())
        self.assertGreater(result['summary']['root_cpu_percent']['count'],0)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        contexts=[r for r in rows if r['event']=='gui_context']
        self.assertEqual(contexts[-1]['control_log'],'/fake/control.jsonl')
        self.assertIn('左臂仿真同源',contexts[-1]['mode'])
        self.assertEqual(len(contexts),2)  # No new IDs/timestamps on every snapshot.
        self.assertTrue(any(r['event']=='gui_state' and r['busy'] for r in rows))
        self.assertIn('已保存',session.label.text())
        d.worker=None; d.busy=False
        session.button.click()
        wait_for(lambda:session.button.isEnabled() and session.running)
        self.assertNotEqual(session.output,first)
        session.button.click()
        wait_for(lambda:not session.running)
        self.assertTrue((first/'summary.json').exists())

    def test_close_finishes_recording_without_destroying_running_process(self):
        session=self.session
        session.button.click()
        wait_for(lambda:session.button.isEnabled() and session.running)
        output=session.output
        self.dialog.close()
        wait_for(lambda:not session.running)
        self.assertTrue((output/'summary.json').exists())
        self.assertTrue(session.close_pending)

    def test_observer_failure_is_visible_and_not_a_control_failure(self):
        session=self.session
        session.script=Path(self.temp.name)/'missing.py'
        session.button.click()
        wait_for(lambda:not session.running)
        self.assertIn('失败或不完整',session.label.text())
        self.assertIsNone(self.dialog.worker)
        self.assertFalse(self.dialog.busy)
        self.assertTrue(session.button.isEnabled())


if __name__=='__main__': unittest.main()
