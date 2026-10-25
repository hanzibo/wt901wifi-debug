"""No robot port: strict config, synthetic UDP, Qt gates and shared offline IK."""
import copy
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'OptUSBProj-ubuntu')]

from jetson_teleop.config import load, PARTS, MODES
from jetson_teleop.run import smoke
from wt901.gui_udp import GuiUdpSource
from optusbproj_ubuntu.real_robot_checkpoints import create, MODES as SIDES
from optusbproj_ubuntu.real_robot_arm import ArmLimits
from PySide6.QtWidgets import QApplication
from jetson_teleop.window import Window

APP = QApplication.instance() or QApplication([])


def raw_config():
    return dict(version=1, mode=MODES[0], frequency=20,
                assignments={key: f'WT{5500000000+i:010d}' for i, key in enumerate(PARTS)},
                axis_mappings={key: ['+Y', '-X', '+Z'] for key in PARTS},
                signs={s:[1]*6 for s in ('left', 'right')},
                palm_offsets={s:dict(axis='z', degrees=90) for s in ('left', 'right')},
                checkpoints={m: m+'.json' for m in MODES})


def checkpoint(mode):
    joints = {}
    for side in SIDES[mode]:
        limits = ArmLimits(side=side, lower=(0,90,0,0,0,50) if side=='right' else ArmLimits().lower,
                           upper=(235,240,180,118,180,121) if side=='right' else ArmLimits().upper)
        for i in range(6):
            joints[side[0].upper()+str(i+1)] = dict(lower=limits.lower[i], upper=limits.upper[i],
                                                   speed=limits.speed[i], acceleration=limits.acceleration[i])
    gesture = {side[0].upper()+str(i): dict(angle=90, tolerance=5)
               for side in SIDES[mode] for i in (2,4)}
    return create(mode, joints, gesture, input_settings=dict(manual_mapping=True,
                          palm_offset_enabled={side:True for side in SIDES[mode]}))


def packet(device, tick=0):
    frame = bytearray(54)
    frame[:12] = device.encode('ascii')
    frame[12:18] = bytes([26,10,25,12,0,0])
    struct.pack_into('<H', frame, 18, tick)
    struct.pack_into('<h', frame, 24, 2048)
    frame[52:] = b'\r\n'
    return bytes(frame)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'config.json'

    def read(self, value):
        self.path.write_text(json.dumps(value), encoding='utf-8')
        return load(self.path)

    def test_valid_and_relative_paths(self):
        result = self.read(raw_config())
        self.assertEqual(result['checkpoints'][MODES[0]].parent, self.path.parent)

    def test_rejects_invalid_or_unsafe_values(self):
        for name, value in [('version',True), ('mode','relative'), ('frequency',float('nan')),
                            ('frequency',101), ('assignments',[]), ('checkpoints',None),
                            ('signs',None), ('palm_offsets',None)]:
            with self.subTest(name=name, value=value):
                raw = raw_config(); raw[name] = value
                with self.assertRaises(ValueError): self.read(raw)
        for raw in ([], None):
            with self.assertRaises(ValueError): self.read(raw)
        raw = raw_config(); raw['assignments'][PARTS[1]] = raw['assignments'][PARTS[0]]
        with self.assertRaises(ValueError): self.read(raw)
        raw = raw_config(); raw['axis_mappings'][PARTS[0]] = ['+X','+Y','-Z']
        with self.assertRaises(ValueError): self.read(raw)

    def test_offline_ik_zero_fingers(self):
        self.assertEqual(smoke(), 0)


class GuiTests(ConfigTests):
    def setUp(self):
        super().setUp()
        for mode in MODES:
            (self.path.parent/(mode+'.json')).write_text(json.dumps(checkpoint(mode)), encoding='utf-8')
        self.source = GuiUdpSource(0, '127.0.0.1')
        self.addCleanup(self.source.close)
        self.window = Window(self.source, self.read(raw_config()))
        self.window.timer.stop(); self.window.control.timer.stop()
        self.addCleanup(self.window.close)

    def feed(self, keys):
        # Test only on loopback, random UDP port. Never an IMU configuration command.
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for key in keys:
                sender.sendto(packet(self.window.config['assignments'][key]), self.source.sock.getsockname())
            time.sleep(.02)
            self.window.poll()
        finally:
            sender.close()

    def test_no_auto_uart_and_explicit_fresh_zero(self):
        w = self.window; d = w.control
        self.assertEqual(w.preset_error, '')
        self.assertIsNone(d.hand_session)
        self.assertIsNone(d.worker)
        self.assertFalse(d.connect_button.isEnabled())
        w.review.setChecked(True); w.zero()
        self.assertFalse(w.zero_confirmed)
        self.feed(PARTS[:3]); w.zero()
        self.assertTrue(w.zero_confirmed)
        self.assertTrue(d.connect_button.isEnabled())
        self.assertTrue(all(w.engine.algorithm_modes[k]=='gyro_tilt' for k in PARTS))
        w.review.setChecked(False)
        self.assertFalse(w.zero_confirmed)
        with patch('optusbproj_ubuntu.real_robot_gui.ElbowSerialWorker') as uart:
            d._connect()
            uart.assert_not_called()

    def test_bilateral_requires_all_six_and_invalidates_reference(self):
        w = self.window; d = w.control
        self.feed(PARTS[:3]); w.review.setChecked(True); w.zero()
        self.assertTrue(w.zero_confirmed)
        d.mode_select.setCurrentIndex(3)
        self.assertFalse(w.zero_confirmed)
        self.assertFalse(w.review.isChecked())
        w.review.setChecked(True); w.zero()
        self.assertFalse(w.zero_confirmed)
        self.feed(PARTS); w.zero()
        self.assertTrue(w.zero_confirmed)

    def test_stale_input_does_not_allow_zero(self):
        self.feed(PARTS)
        w = self.window; w.review.setChecked(True)
        for key in PARTS:
            w.engine.received_at[w.engine.assignments[key]] -= 1
        w.zero(); self.assertFalse(w.zero_confirmed)

    def test_missing_preset_fails_closed(self):
        w = self.window; d = w.control
        w.config['checkpoints'][MODES[1]] = self.path.parent/'missing.json'
        d.mode_select.setCurrentIndex(3)
        self.assertTrue(w.preset_error)
        w.review.setChecked(True); self.feed(PARTS); w.zero()
        self.assertFalse(w.zero_confirmed)
        self.assertFalse(d.connect_button.isEnabled())

    def test_lightweight_imports_and_render(self):
        self.window.show(); self.window.poll(); APP.processEvents()
        image = self.window.grab()
        self.assertGreater(image.width(), 0)
        for name in ('vtk', 'pyvista', 'cv2', 'torch', 'mediapipe', 'pyqtgraph'):
            self.assertNotIn(name, sys.modules)


if __name__ == '__main__': unittest.main()
