from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from PySide6 import QtWidgets

from optusbproj_ubuntu.device_settings import DeviceSettingsDialog, Setting
from wt901.registers import (REG_ACCRANGE, REG_AXIS6, REG_BANDWIDTH,
                             REG_CALSW, REG_GYRORANGE, REG_ORIENT, REG_RATE)


class FakeSource:
    def __init__(self):
        self.reads = []
        self.writes = []
        self.online = {"WT5500011381": ("10.42.0.174", 0),
                       "WT5500011408": ("10.42.0.82", 0)}

    def seen_devices(self, max_age=3.0):
        return self.online

    def device_ip(self, device_id, max_age=3.0):
        if device_id not in self.online:
            raise RuntimeError("offline")
        return self.online[device_id][0]

    def read_registers(self, device_id, address):
        self.reads.append((device_id, address))

    def unlock(self, device_id, max_age=3.0):
        self.writes.append((device_id, "unlock"))

    def write_register(self, device_id, address, value, max_age=3.0):
        self.writes.append((device_id, address, value))


class DeviceSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.source = FakeSource()
        self.dialog = DeviceSettingsDialog(self.source)
        self.dialog.device_combo.setCurrentIndex(1)

    def tearDown(self):
        self.dialog.setting = None
        self.dialog.calibration = None
        self.dialog.close()

    def _finish_reads(self):
        self.dialog.on_replies([("WT5500011381", (0, 0, 0, 0))])
        self.dialog.on_replies([("WT5500011381", (3, 3, 3, 0))])
        self.dialog.on_replies([("WT5500011381", (0, 1, 0, 0))])
        self.dialog.on_replies([("WT5500011381", (20, 0, 0, 0))])

    def test_register_reads_are_sequential_and_target_specific(self):
        self.assertEqual(self.source.reads, [("WT5500011381", REG_CALSW)])
        self.dialog.on_replies([("WT5500011408", (9, 9, 9, 9))])
        self.assertEqual(self.dialog.waiting_address, REG_CALSW)
        self._finish_reads()
        self.assertEqual([address for _, address in self.source.reads],
                         [REG_CALSW, REG_BANDWIDTH, REG_ORIENT, REG_RATE])
        self.assertEqual(self.dialog.registers[REG_GYRORANGE], 3)
        self.assertEqual(self.dialog.registers[REG_ACCRANGE], 3)
        self.assertEqual(self.dialog.registers[REG_AXIS6], 1)
        self.assertEqual(self.dialog.rate.value(), 50.0)
        self.assertTrue(self.dialog.algorithm_button.isEnabled())

    def test_setting_reply_must_match_requested_value(self):
        self._finish_reads()
        self.dialog.setting = Setting(REG_AXIS6, 0, "融合算法")
        self.dialog.waiting_address = REG_AXIS6
        self.dialog.on_replies([("WT5500011381", (1, 0, 0, 0))])
        self.assertIn("未确认", self.dialog.status.text())
        self.assertIsNone(self.dialog.setting)

    def test_busy_device_cannot_be_switched(self):
        self.dialog.device_combo.setCurrentIndex(2)
        self.assertEqual(self.dialog.device_id, "WT5500011381")
        self.assertEqual(self.dialog.device_combo.currentData(), "WT5500011381")

    def test_reference_uses_selected_calsw_code(self):
        self.dialog.waiting_address = None
        self.dialog._reference_value = 0x04
        self.dialog.calibration = "reference_pending"
        self.dialog._enter_calibration("reference")
        self.assertIn(("WT5500011381", REG_CALSW, 0x04), self.source.writes)
        self.assertEqual(self.dialog.calibration, "reference")


if __name__ == "__main__":
    unittest.main()
