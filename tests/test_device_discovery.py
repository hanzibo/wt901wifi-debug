"""Network-independent discovery; all device commands are mocked."""
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets

from wt901_viz.assignment import DeviceAssignmentDialog
from wt901_viz.device_list import ConnectedDevicesDialog


DEVICE = "WT5500011361"
IP = "192.168.5.228"


class FakeSource:
    def __init__(self):
        self.endpoints = {DEVICE: (IP, 9250, time.monotonic())}
        self.sock = SimpleNamespace(getsockname=lambda: ("0.0.0.0", 1399))
        self.switch_to_ap_udp = Mock()

    def seen_endpoints(self, max_age=3.0):
        return {key: value for key, value in self.endpoints.items()
                if time.monotonic() - value[2] <= max_age}

    def latest_packet(self, device_id):
        return SimpleNamespace(battery_v=3.94, battery_percent=90)

    def device_ip(self, device_id):
        return self.seen_endpoints()[device_id][0]


class DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.source = FakeSource()
        self.dialogs = []
        for cls in (ConnectedDevicesDialog, DeviceAssignmentDialog):
            patcher = patch.object(cls, "refresh", lambda self: None)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.close()
        self.app.processEvents()

    def devices(self):
        dialog = ConnectedDevicesDialog(self.source)
        self.dialogs.append(dialog)
        dialog.timer.stop()
        dialog._clear("外部 Wi-Fi：Redmi K60")
        return dialog

    def assignment(self):
        callback = Mock(return_value=True)
        states = {key: SimpleNamespace(name=key, device_id="") for key in ("left", "right")}
        dialog = DeviceAssignmentDialog(self.source, states, callback)
        self.dialogs.append(dialog)
        dialog.timer.stop()
        dialog.connection = "Redmi K60"
        dialog._on_mode(True, "infrastructure", "")
        return dialog, callback

    def choose(self, dialog, key="left"):
        combo = dialog.combos[key]
        combo.setCurrentIndex(combo.findData(DEVICE))

    def test_external_wifi_without_mac_shows_live_device(self):
        dialog = self.devices()
        self.assertEqual(dialog.table.rowCount(), 1)
        self.assertEqual(dialog.table.item(0, 0).text(), DEVICE)
        self.assertEqual(dialog.table.item(0, 3).text(), "未知")
        self.assertIn("90%", dialog.table.item(0, 4).text())
        self.source.endpoints.clear()
        dialog._render()
        self.assertEqual(dialog.table.rowCount(), 0)

    def test_hotspot_metadata_does_not_filter_or_duplicate_streams(self):
        dialog = self.devices()
        dialog.hotspot_active = True
        dialog.stations = ["aa:bb:cc:dd:ee:ff", "aa:bb:cc:dd:ee:00"]
        dialog.neighbors = {dialog.stations[0]: IP, dialog.stations[1]: "10.42.0.99"}
        dialog._render()
        self.assertEqual(dialog.table.rowCount(), 2)
        self.assertEqual(dialog.table.item(1, 0).text(), "未识别")
        dialog.neighbors.clear()
        dialog._render()
        self.assertEqual(dialog.table.item(0, 0).text(), DEVICE)

    def test_external_switch_requires_confirmation_and_targets_selected_id(self):
        dialog = self.devices()
        dialog.table.selectRow(0)
        self.source.switch_to_ap_udp.assert_not_called()
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.No):
            dialog.switch_selected_to_ap()
        self.source.switch_to_ap_udp.assert_not_called()
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            dialog.switch_selected_to_ap()
            dialog.switch_selected_to_ap()  # cooldown
        self.source.switch_to_ap_udp.assert_called_once_with(DEVICE)

    def test_switch_rechecks_endpoint_after_confirmation(self):
        dialog = self.devices()
        dialog.table.selectRow(0)

        def changed(*args):
            self.source.endpoints[DEVICE] = ("192.168.5.229", 9250, time.monotonic())
            return QtWidgets.QMessageBox.Yes

        with patch.object(QtWidgets.QMessageBox, "question", side_effect=changed):
            dialog.switch_selected_to_ap()
        self.source.switch_to_ap_udp.assert_not_called()

    def test_switch_rejects_stale_stream_and_busy_device(self):
        dialog = self.devices()
        dialog.table.selectRow(0)
        dialog.can_switch = lambda device_id: False
        dialog.switch_selected_to_ap()
        self.source.switch_to_ap_udp.assert_not_called()
        self.source.endpoints[DEVICE] = (IP, 9250, time.monotonic() - 4)
        self.assertIsNone(dialog._eligible_device())

    def test_hotspot_switch_still_requires_fresh_association(self):
        dialog = self.devices()
        dialog.hotspot_active = True
        dialog.table.selectRow(0)
        dialog._last_scan = time.monotonic()
        self.assertIsNone(dialog._eligible_device())
        dialog.stations = ["aa:bb:cc:dd:ee:ff"]
        dialog.neighbors = {dialog.stations[0]: IP}
        self.assertEqual(dialog._eligible_device(), (DEVICE, IP))
        dialog._last_scan -= 6
        self.assertIsNone(dialog._eligible_device())

    def test_switch_rejects_changed_routing_identity(self):
        dialog = self.devices()
        dialog.table.selectRow(0)
        with patch.object(self.source, "device_ip", return_value="192.168.5.229"), \
                patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            dialog.switch_selected_to_ap()
        self.source.switch_to_ap_udp.assert_not_called()

    def test_external_wifi_assignment_applies(self):
        dialog, callback = self.assignment()
        self.choose(dialog)
        dialog._apply()
        callback.assert_called_once_with({"left": DEVICE, "right": ""})

    def test_metadata_failure_does_not_hide_stream(self):
        dialog, callback = self.assignment()
        dialog._clear("nmcli 不可用")
        self.assertIn(DEVICE, dialog.online_ids)
        self.choose(dialog)
        dialog._apply()
        callback.assert_called_once()

    def test_assignment_rechecks_staleness_at_apply(self):
        dialog, callback = self.assignment()
        self.choose(dialog)
        self.source.endpoints.clear()
        dialog._apply()
        callback.assert_not_called()
        self.assertIn("已不在线", dialog.status.text())
        dialog._fill_choices()
        self.assertEqual(dialog.combos["left"].currentData(), DEVICE)
        self.assertIn("已离线", dialog.combos["left"].currentText())
        dialog._apply()
        callback.assert_not_called()

    def test_assignment_rejects_duplicates(self):
        dialog, callback = self.assignment()
        self.choose(dialog, "left")
        self.choose(dialog, "right")
        dialog._apply()
        callback.assert_not_called()
        self.assertIn("不能同时", dialog.status.text())


if __name__ == "__main__":
    unittest.main()
