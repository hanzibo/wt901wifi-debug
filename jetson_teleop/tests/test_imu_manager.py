"""Shared-source management and window IPC; AP writes are always mocked."""
import json
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from test_jetson import APP, Window, GuiUdpSource, load, raw_config, checkpoint, MODES, PARTS, packet
from jetson_teleop.imu_manager import ImuManager
from jetson_teleop.single_instance import InstanceServer, forward
from PySide6 import QtCore, QtWidgets


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for mode in MODES:
            (root/(mode+'.json')).write_text(json.dumps(checkpoint(mode)))
        (root/'config.json').write_text(json.dumps(raw_config()))
        self.source = GuiUdpSource(0,'127.0.0.1')
        self.addCleanup(self.source.close)
        self.window = Window(self.source, load(root/'config.json'), background=False)
        self.window.timer.stop(); self.window.control.timer.stop()
        self.addCleanup(self.window.close)
        with patch.object(ImuManager, 'refresh_network'):
            self.manager = ImuManager(self.window)
        self.manager.timer.stop(); self.manager.network_timer.stop()
        self.addCleanup(self.manager.close)
        self.key = self.window.config['assignments'][PARTS[0]]

    def feed(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(packet(self.key), self.source.sock.getsockname())
        time.sleep(.01); self.window.poll(); self.manager.render()
        self.manager.table.selectRow(0)

    def test_one_shared_source_and_live_receive_display(self):
        self.feed()
        self.assertIs(self.manager.source, self.window.source)
        self.assertTrue(self.manager.ap_button.isEnabled())
        self.assertEqual(self.manager.table.item(0,0).text(), self.key)
        self.assertEqual(self.manager.table.item(0,1).text(), '左上臂')
        self.assertEqual(self.manager.table.item(0,7).text(), '在线')
        self.assertEqual(self.window.engine.counts[self.key], 1)

    def test_ap_confirmation_cancel_and_exact_selected_device(self):
        self.feed()
        with patch.object(self.source, 'switch_to_ap_udp') as send, \
                patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.No):
            self.manager.switch_ap(); send.assert_not_called()
        with patch.object(self.source, 'switch_to_ap_udp') as send, \
                patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.Yes):
            self.manager.switch_ap(); send.assert_called_once_with(self.key)
        self.assertIn('没有设备确认回执', self.manager.status.text())

    def test_uart_connection_blocks_ap_even_when_not_following(self):
        self.feed()
        self.window.control.worker = Mock()
        try:
            self.manager.controls()
            self.assertFalse(self.manager.ap_button.isEnabled())
            with patch.object(self.source,'switch_to_ap_udp') as send:
                self.manager.switch_ap(); send.assert_not_called()
        finally:
            self.window.control.worker = None

    def test_recheck_after_confirmation(self):
        self.feed()
        def confirm(*args):
            self.window.control.busy = True
            return QtWidgets.QMessageBox.Yes
        try:
            with patch.object(self.source,'switch_to_ap_udp') as send, \
                    patch.object(QtWidgets.QMessageBox,'question',side_effect=confirm):
                self.manager.switch_ap(); send.assert_not_called()
        finally:
            self.window.control.busy = False

    def test_stale_device_and_disappearance_not_success(self):
        self.feed()
        self.manager.pending = (self.key, time.monotonic()-5)
        ip, port, stamp = self.source._endpoints[self.key]
        self.source._endpoints[self.key] = (ip,port,stamp-4)
        self.manager.render()
        self.assertFalse(self.manager.ap_button.isEnabled())
        self.assertIn('不等于 AP 已验证', self.manager.status.text())
        with patch.object(self.source,'switch_to_ap_udp') as send:
            self.manager.switch_ap(); send.assert_not_called()

    def test_ipv4_inventory_uses_all_interfaces(self):
        raw = json.dumps([dict(ifname='eth0',addr_info=[dict(family='inet',local='192.168.3.8',prefixlen=24)])])
        with patch.object(self.manager.process,'readAllStandardOutput',return_value=raw.encode()):
            self.manager.network_ready(0)
        self.assertIn('eth0: 192.168.3.8/24',self.manager.network.text())

    def test_closing_robot_window_keeps_manager_and_input_alive(self):
        self.window.imu_manager = self.manager
        self.window.timer.start()
        self.manager.show(); self.window.show(); APP.processEvents()
        self.window.close()
        self.assertFalse(self.window.isVisible())
        self.assertTrue(self.manager.isVisible())
        self.assertTrue(self.window.timer.isActive())

    def test_second_desktop_only_requests_window_not_new_receiver(self):
        name = str(Path(self.temp.name)/'instance.sock')
        server = InstanceServer(name)
        self.addCleanup(server.close)
        seen = []
        server.requested.connect(seen.append)
        self.assertTrue(forward(name,'imu'))
        for _ in range(5): APP.processEvents()
        self.assertEqual(seen,['imu'])
        self.assertTrue(forward(name,'control'))
        for _ in range(5): APP.processEvents()
        self.assertEqual(seen,['imu','control'])
        with self.assertRaises(ValueError): forward(name,'switch_ap')
