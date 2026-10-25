"""Offline wizard tests: mock nmcli and writes; never change host networking."""
import ipaddress
import os
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets
from wt901_viz.onboarding import DeviceSetupDialog, build_sta_command


class OnboardingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.refresh = patch.object(DeviceSetupDialog, 'refresh').start()
        self.nmcli = patch.object(DeviceSetupDialog, '_run_nmcli').start()
        self.addCleanup(patch.stopall)
        self.source = Mock()
        self.source.sock.getsockname.return_value = ('0.0.0.0', 1399)
        self.source.seen_devices.return_value = {}
        self.dialog = DeviceSetupDialog(self.source)
        self.dialog.refresh_timer.stop()
        self.dialog.join_timer.stop()
        self.addCleanup(self.dialog.close)

    def external(self):
        self.dialog.network_mode.setCurrentIndex(1)
        self.dialog.external_ssid.setText('Redmi K60')
        self.dialog.external_ip.setText('192.168.5.126')

    def ap(self):
        self.dialog.interface = 'wlan0'
        self.dialog.mode = 'infrastructure'
        self.dialog.current_ssid = 'WT5500011381'
        self.dialog.local_ip = '192.168.4.2'
        self.dialog.local_network = ipaddress.IPv4Network('192.168.4.0/24')

    def password(self):
        self.dialog.hotspot_password.setText('test-secret-123')
        self.dialog.confirm_password.setText('test-secret-123')

    def send(self, answer=QtWidgets.QMessageBox.StandardButton.Yes):
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=answer) as confirm:
            with patch('wt901_viz.onboarding.socket.socket') as sock:
                self.dialog.send_sta_configuration()
        return confirm, sock

    def test_external_mode_autofill_and_keep_target_while_on_device_ap(self):
        self.dialog.current_ssid = 'Redmi K60'
        self.dialog.mode = 'infrastructure'
        self.dialog._on_local_ip(True, '192.168.5.126/24\n', '')
        self.dialog.network_mode.setCurrentIndex(1)
        self.assertEqual(self.dialog.external_ssid.text(), 'Redmi K60')
        self.assertEqual(self.dialog.external_ip.text(), '192.168.5.126')
        self.ap()
        self.dialog._on_local_ip(True, '192.168.4.2/24\n', '')
        self.dialog._on_hotspot_profile(False, '', '')
        self.assertEqual(self.dialog.external_ip.text(), '192.168.5.126')
        self.assertTrue(self.dialog.send_button.isEnabled())
        self.assertFalse(self.dialog.hotspot_up_button.isEnabled())
        self.assertFalse(self.dialog.hotspot_down_button.isEnabled())
        self.assertTrue(self.dialog.sta_form.isRowVisible(self.dialog.external_ip))
        self.assertFalse(self.dialog.sta_form.isRowVisible(self.dialog.hotspot_ssid))

    def test_external_send_without_pc_hotspot_and_no_secret_in_log(self):
        self.external()
        self.ap()
        self.password()
        confirm, sock = self.send()
        sock.return_value.__enter__.return_value.sendto.assert_called_once_with(
            b'IPWIFI:"Redmi K60","test-secret-123";UDP192.168.5.126,1399\r\n',
            ('192.168.4.1', 9250))
        self.assertIn('192.168.5.126:1399', confirm.call_args.args[2])
        self.assertNotIn('test-secret', confirm.call_args.args[2])
        self.assertNotIn('test-secret', self.dialog.log.toPlainText())
        self.assertEqual(self.dialog.hotspot_password.text(), '')
        self.assertEqual(self.dialog.expected_id, 'WT5500011381')
        self.nmcli.assert_not_called()

    def test_cancel_never_writes(self):
        self.external()
        self.ap()
        self.password()
        _, sock = self.send(QtWidgets.QMessageBox.StandardButton.No)
        sock.assert_not_called()
        self.assertIsNone(self.dialog.expected_id)

    def test_remote_receiver_uses_selected_port_not_pc_listener(self):
        self.dialog.network_mode.setCurrentIndex(2)
        self.dialog.external_ssid.setText('HUAWEI-509')
        self.dialog.external_ip.setText('192.168.3.8')
        self.dialog.external_port.setValue(1400)
        self.source.sock.getsockname.return_value = ('10.42.0.1',1399)
        self.ap(); self.password()
        confirm, sock = self.send()
        sock.return_value.__enter__.return_value.sendto.assert_called_once_with(
            b'IPWIFI:"HUAWEI-509","test-secret-123";UDP192.168.3.8,1400\r\n',
            ('192.168.4.1',9250))
        self.assertTrue(self.dialog.join_target['remote'])
        self.assertIn('板子',confirm.call_args.args[2])
        self.source.seen_devices.return_value = {'WT5500011381': ('192.168.3.6',time.monotonic())}
        self.dialog.check_join()
        self.assertFalse(self.dialog.join_reported)
        self.assertIn('尚未在本机验证',self.dialog.result_label.text())

    def test_remote_mode_does_not_overwrite_board_ip_with_pc_wifi(self):
        self.dialog.network_mode.setCurrentIndex(2)
        self.dialog.external_ip.setText('192.168.3.8')
        self.dialog.external_ssid.setText('HUAWEI-509')
        self.dialog.current_ssid = 'bjut_wifi'
        self.dialog.mode = 'infrastructure'
        self.dialog._on_local_ip(True,'10.22.17.185/24\n','')
        self.dialog._capture_external_address()
        self.assertEqual(self.dialog.external_ip.text(),'192.168.3.8')
        self.assertEqual(self.dialog.external_ssid.text(),'HUAWEI-509')

    def test_invalid_targets_password_and_listener_never_write(self):
        self.external()
        self.ap()
        self.password()
        for address in ('', 'bad', '0.0.0.0', '127.0.0.1', '224.0.0.1',
                        '169.254.1.1', '255.255.255.255', '192.168.4.1', '192.168.4.2'):
            with self.subTest(address=address):
                self.dialog.external_ip.setText(address)
                confirm, sock = self.send()
                confirm.assert_not_called()
                sock.assert_not_called()
        self.dialog.external_ip.setText('192.168.5.126')
        self.dialog.confirm_password.setText('mismatch')
        _, sock = self.send()
        sock.assert_not_called()
        self.password()
        self.source.sock.getsockname.return_value = ('10.42.0.1', 1399)
        _, sock = self.send()
        sock.assert_not_called()

    def test_only_selected_new_device_on_returned_wifi_is_success(self):
        self.external()
        self.ap()
        self.password()
        self.send()
        self.source.seen_devices.return_value = {
            'WT5500011381': ('192.168.5.10', time.monotonic())}
        self.dialog.check_join()
        self.assertFalse(self.dialog.join_reported)  # Still on device AP.
        self.dialog.current_ssid = 'Redmi K60'
        self.dialog.local_ip = '192.168.5.126'
        self.dialog.local_network = ipaddress.IPv4Network('192.168.5.0/24')
        self.source.seen_devices.return_value = {
            'WT5500011381': ('192.168.5.10', self.dialog.sent_at - 1)}
        self.dialog.check_join()
        self.assertFalse(self.dialog.join_reported)  # Stale pre-write sample.
        self.source.seen_devices.return_value = {
            'WT5500011408': ('192.168.5.10', time.monotonic())}
        self.dialog.check_join()
        self.assertFalse(self.dialog.join_reported)  # Another device.
        self.source.seen_devices.return_value = {
            'WT5500011381': ('192.168.4.1', time.monotonic())}
        self.dialog.check_join()
        self.assertFalse(self.dialog.join_reported)  # Device AP subnet.
        self.source.seen_devices.return_value = {
            'WT5500011381': ('192.168.5.10', time.monotonic())}
        self.dialog.check_join()
        self.assertTrue(self.dialog.join_reported)

    def test_changed_pc_ip_is_not_success_and_target_edit_does_not_rewrite_expectation(self):
        self.external()
        self.ap()
        self.password()
        self.send()
        self.dialog.current_ssid = 'Redmi K60'
        self.dialog.local_ip = '192.168.5.127'
        self.dialog.external_ip.setText('192.168.5.127')
        self.dialog.check_join()
        self.assertFalse(self.dialog.join_reported)
        self.assertIn('不一致', self.dialog.result_label.text())
        self.assertEqual(self.dialog.join_target['ip'], '192.168.5.126')

    def test_legacy_hotspot_checks_and_exact_payload_remain(self):
        self.ap()
        self.password()
        _, sock = self.send()
        sock.assert_not_called()
        self.dialog._on_hotspot_profile(True, 'ap\nWT901-LAN\nbg\nshared\n10.42.0.1/24\n', '')
        _, sock = self.send()
        sock.return_value.__enter__.return_value.sendto.assert_called_once_with(
            b'IPWIFI:"WT901-LAN","test-secret-123";UDP10.42.0.1,1399\r\n',
            ('192.168.4.1', 9250))
        self.dialog.mode = 'ap'
        self.dialog.current_ssid = 'WT901-LAN'
        self.source.seen_devices.return_value = {
            'WT5500011381': ('10.42.0.10', time.monotonic())}
        self.dialog.check_join()
        self.assertTrue(self.dialog.join_reported)

    def test_external_mode_does_not_start_or_stop_pc_hotspot(self):
        self.external()
        self.dialog.start_hotspot()
        self.dialog.stop_hotspot()
        self.nmcli.assert_not_called()

    def test_capture_ignores_device_ap(self):
        self.external()
        self.ap()
        self.dialog._capture_external_address()
        self.assertEqual(self.dialog.external_ip.text(), '192.168.5.126')

    def test_command_validation(self):
        for ssid, password, ip, port in (
                ('x;bad', 'password', '192.168.5.1', 1399),
                ('Redmi', 'short', '192.168.5.1', 1399),
                ('Redmi', 'password', 'bad', 1399),
                ('Redmi', 'password', '192.168.5.1', 0)):
            with self.subTest(ssid=ssid, ip=ip, port=port):
                with self.assertRaises(ValueError):
                    build_sta_command(ssid, password, ip, port)


if __name__ == '__main__':
    unittest.main()
