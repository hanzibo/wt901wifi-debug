import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets
from wrist_demo import WristWindow
from wt901.receive_stats import ReceiveStats


class WristReceiveDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_assigned_unassigned_warmup_and_stale(self):
        source = Mock()
        source.receive_stats.return_value = {
            "WT1": ReceiveStats(50, 10, 100),
            "WT2": ReceiveStats(None, 5, 1),
            "old": ReceiveStats(0, 4000, 0),
        }
        window = SimpleNamespace(source=source, _last_rate_refresh=float("-inf"),
            receive_rate_label=QtWidgets.QLabel(), part_states={
                "left": SimpleNamespace(device_id="WT1", name="左手"),
                "right": SimpleNamespace(device_id="absent", name="右手")})
        with patch("wrist_demo.time.monotonic", return_value=10):
            WristWindow._update_receive_rates(window)
            WristWindow._update_receive_rates(window)
        text = window.receive_rate_label.text()
        self.assertIn("左手 WT1：50.0 Hz", text)
        self.assertIn("未分配 WT2：采样中", text)
        self.assertIn("右手 absent：尚未收包", text)
        self.assertNotIn("old", text)
        source.receive_stats.assert_called_once()
        source.receive_stats.return_value["WT1"] = ReceiveStats(0, 5000, 0)
        with patch("wrist_demo.time.monotonic", return_value=11):
            WristWindow._update_receive_rates(window)
        self.assertIn("0.0 Hz · 距末包 5000 ms", window.receive_rate_label.text())

    def test_unsupported_source(self):
        window = SimpleNamespace(source=object(), _last_rate_refresh=float("-inf"),
                                 receive_rate_label=QtWidgets.QLabel())
        WristWindow._update_receive_rates(window)
        self.assertIn("不支持", window.receive_rate_label.text())
