from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from PySide6 import QtWidgets

from wt901_viz.device_list import ConnectedDevicesDialog


class FakeSource:
    def __init__(self):
        self.sock = SimpleNamespace(getsockname=lambda: ("0.0.0.0", 1399))
        self.packet = SimpleNamespace(battery_v=3.94, battery_percent=90)

    def seen_endpoints(self, max_age=3.0):
        return {"WT5500011433": ("10.42.0.62", 9250, 0.0)}

    def latest_packet(self, device_id, max_age=3.0):
        return self.packet if device_id == "WT5500011433" else None


class ConnectedDevicesBatteryTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_each_hotspot_device_shows_fresh_voltage_and_estimated_charge(self):
        source = FakeSource()
        with patch.object(ConnectedDevicesDialog, "refresh", lambda self: None):
            dialog = ConnectedDevicesDialog(source)
        try:
            dialog.timer.stop()
            dialog.hotspot_active = True
            dialog.hotspot_ip = "10.42.0.1"
            dialog.stations = ["24:4c:ab:67:5d:46", "24:4c:ab:68:34:c2"]
            dialog.neighbors = {
                "24:4c:ab:67:5d:46": "10.42.0.62",
                "24:4c:ab:68:34:c2": "10.42.0.97"}
            dialog._render()
            self.assertEqual(dialog.table.columnCount(), 6)
            self.assertEqual(dialog.table.item(0, 4).text(), "3.94 V（约 90%）")
            self.assertEqual(dialog.table.item(1, 4).text(), "未知")
            dialog.table.selectRow(0)
            self.assertEqual(dialog._selected_device(),
                             ("WT5500011433", "10.42.0.62"))
            source.packet = None
            dialog._render()
            self.assertEqual(dialog.table.item(0, 4).text(), "未知")
        finally:
            dialog.close()
