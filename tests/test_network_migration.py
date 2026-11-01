"""Offline migration tests: mocked sends only, never configure a real IMU."""
import os
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtCore, QtWidgets
from wt901.gui_udp import GuiUdpSource
from wt901.network_config import build_sta_command
from wt901_viz.network_migration import NetworkMigrationDialog

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
SECRET = 'offline-test-password'


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.source = Mock()
        self.source.seen_endpoints.return_value = {
            f'WT{i}': (f'192.168.3.{10+i}', 9250, time.monotonic()) for i in range(6)}
        self.guard = Mock(return_value=True)
        self.dialog = NetworkMigrationDialog(self.source, can_switch=self.guard)
        self.addCleanup(self.dialog.close)
        self.dialog.password.setText(SECRET)
        self.dialog.password_confirm.setText(SECRET)
        self.dialog.ready.setChecked(True)

    def start(self, answer=QtWidgets.QMessageBox.Yes):
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=answer) as question:
            self.dialog.start()
        self.dialog.timer.stop()
        return question

    def test_six_device_snapshot_survives_source_disappearance(self):
        question = self.start()
        self.source.seen_endpoints.return_value = {}
        # Transport does the actual fresh-ID/IP check, mocked here.
        for _ in range(6):
            self.dialog.send_next()
        self.assertEqual(self.source.configure_network.call_count, 6)
        self.source.configure_network.assert_any_call(
            'WT5', '192.168.3.15', 'HUAWEI-509', SECRET, '192.168.3.8', 1399)
        self.assertIsNone(self.dialog.target)
        self.assertEqual(self.dialog.password.text(), '')
        self.assertNotIn(SECRET, question.call_args.args[2])
        self.assertNotIn(SECRET, self.dialog.status.text())
        self.assertIn('不等于接入成功', self.dialog.status.text())

    def test_unchecked_device_excluded_and_profile_target(self):
        self.dialog.profile.setCurrentIndex(1)
        self.dialog.table.item(2, 0).setCheckState(QtCore.Qt.Unchecked)
        self.start()
        for _ in range(5):
            self.dialog.send_next()
        calls = self.source.configure_network.call_args_list
        self.assertEqual(len(calls), 5)
        self.assertNotIn('WT2', [call.args[0] for call in calls])
        self.assertEqual(calls[0].args[2:], ('WT901-LAN', SECRET, '10.42.0.1', 1399))

    def test_cancel_confirmation_and_missing_consent_never_send(self):
        self.start(QtWidgets.QMessageBox.No)
        self.assertIsNone(self.dialog.target)
        self.dialog.ready.setChecked(False)
        self.start()
        self.source.configure_network.assert_not_called()

    def test_guard_rechecked_after_modal_and_between_devices(self):
        with patch.object(QtWidgets.QMessageBox, 'question', side_effect=lambda *args: (
                self.guard.configure_mock(return_value=False) or QtWidgets.QMessageBox.Yes)):
            self.dialog.start()
        self.assertIsNone(self.dialog.target)
        self.guard.return_value = True
        self.start()
        self.dialog.send_next()
        self.guard.return_value = False
        self.dialog.send_next()
        self.assertEqual(self.source.configure_network.call_count, 1)
        self.assertIsNone(self.dialog.target)
        self.assertIn('控制状态改变', self.dialog.status.text())

    def test_failure_continues_without_retry_or_secret_in_status(self):
        self.source.configure_network.side_effect = [OSError(SECRET)] + [None]*5
        self.start()
        for _ in range(6): self.dialog.send_next()
        self.assertEqual(self.dialog.failed, 1)
        self.assertEqual(self.dialog.sent, 5)
        self.assertNotIn(SECRET, self.dialog.status.text())
        self.assertNotIn(SECRET, self.dialog.table.item(0, 3).text())

    def test_close_cancels_remaining_and_forgets_password(self):
        self.start()
        self.dialog.send_next()
        self.dialog.reject()
        self.dialog.send_next()
        self.assertEqual(self.source.configure_network.call_count, 1)
        self.assertFalse(self.dialog.timer.isActive())
        self.assertIsNone(self.dialog.target)

    def test_validation_rejects_invalid_targets_and_password_mismatch(self):
        for ip in ('0.0.0.0', '127.0.0.1', '255.255.255.255', '224.0.0.1', 'not-ip'):
            with self.assertRaises(ValueError): build_sta_command('test', SECRET, ip, 1399)
        self.dialog.password_confirm.setText('different')
        self.start()
        self.assertIsNone(self.dialog.target)


class SourceMigrationTests(unittest.TestCase):
    def setUp(self):
        # Avoid even binding a real port. Only the sendto mock is used.
        self.source = GuiUdpSource.__new__(GuiUdpSource)
        self.source.sock = Mock()
        stamp = time.monotonic()
        self.source._devices = {'WT1': ('192.168.3.10', stamp)}
        self.source._endpoints = {'WT1': ('192.168.3.10', 12345, stamp)}

    def send(self, expected='192.168.3.10'):
        self.source.configure_network('WT1', expected, 'HUAWEI-509', SECRET, '192.168.3.8', 1399)

    def test_exact_unicast_to_config_port_not_packet_source_port(self):
        self.send()
        self.source.sock.sendto.assert_called_once_with(
            build_sta_command('HUAWEI-509', SECRET, '192.168.3.8', 1399), ('192.168.3.10', 9250))

    def test_stale_reassigned_or_ambiguous_ip_never_sends(self):
        with self.assertRaises(RuntimeError): self.send('192.168.3.11')
        self.source._endpoints['WT2'] = ('192.168.3.10', 9250, time.monotonic())
        with self.assertRaises(RuntimeError): self.send()
        self.source._endpoints.pop('WT2')
        self.source._devices['WT1'] = ('192.168.3.10', time.monotonic()-4)
        with self.assertRaises(RuntimeError): self.send()
        self.source.sock.sendto.assert_not_called()
