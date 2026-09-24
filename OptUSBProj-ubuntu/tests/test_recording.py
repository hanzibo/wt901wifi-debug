from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from wt901.protocol import AttitudePacket
from optusbproj_ubuntu.recording import PacketRecorder, ReplaySource
from optusbproj_ubuntu.settings import load_assignments, save_assignments


def packet(device_id: str, roll: float) -> AttitudePacket:
    return AttitudePacket(
        device_id=device_id, year=2026, month=9, day=24, hour=10, minute=0,
        second=0, millisecond=0, acc_x=0, acc_y=0, acc_z=1,
        gyro_x=0, gyro_y=0, gyro_z=0, mag_x=20, mag_y=0, mag_z=40,
        roll=roll, pitch=0, yaw=0, temperature=25, battery_v=3.9,
        rssi=-50, version=1)


class RecordingTests(unittest.TestCase):
    def test_csv_roundtrip_preserves_device_id_and_timing(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            recorder = PacketRecorder(path)
            recorder.write(packet("WT5500011381", 10), host_time_ns=1_000_000_000)
            recorder.write(packet("WT5500011408", 25), host_time_ns=1_050_000_000)
            recorder.close()
            replay = ReplaySource(path)
            try:
                start = replay._started_at
                first = replay.poll(now=start)
                self.assertEqual([(p.device_id, p.roll) for p in first],
                                 [("WT5500011381", 10)])
                second = replay.poll(now=start + 0.1)
                self.assertEqual([(p.device_id, p.roll) for p in second],
                                 [("WT5500011408", 25)])
            finally:
                replay.close()

    def test_assignment_store_rejects_duplicate_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "local_config.json"
            save_assignments(path, {"torso": "WT5500011381"})
            self.assertEqual(load_assignments(path)["torso"], "WT5500011381")
            path.write_text('{"assignments":{"torso":"WT5500011381",'
                            '"left_hand":"WT5500011381"}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_assignments(path)


if __name__ == "__main__":
    unittest.main()
