"""Read-only single-device experiment recorder and phase schedule."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT.parent), str(PROJECT)]

from imu_axis_experiment import (AxisExperimentRecorder, DEFAULT_DEVICE_ID,
                                 STAGES, TOTAL_SECONDS, stage_at)  # noqa: E402
from wt901.protocol import AttitudePacket  # noqa: E402


def packet(device_id: str) -> AttitudePacket:
    return AttitudePacket(
        device_id=device_id,
        year=2026, month=9, day=29, hour=10, minute=0, second=0, millisecond=0,
        acc_x=0.0, acc_y=0.0, acc_z=1.0,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        mag_x=30.0, mag_y=20.0, mag_z=10.0,
        roll=1.0, pitch=2.0, yaw=3.0,
        temperature=25.0, battery_v=3.8, rssi=-50, version=13038,
    )


class AxisExperimentTests(unittest.TestCase):
    def test_stage_boundaries(self):
        boundary = 0
        for name, _, duration in STAGES:
            self.assertEqual(stage_at(boundary)[0], name)
            self.assertEqual(stage_at(boundary + duration - 0.001)[0], name)
            boundary += duration
        self.assertEqual(boundary, TOTAL_SECONDS)
        self.assertIsNone(stage_at(TOTAL_SECONDS))
        with self.assertRaises(ValueError):
            stage_at(-0.1)

    def test_recorder_filters_target_and_saves_raw_values(self):
        with tempfile.TemporaryDirectory() as directory:
            start = 1_000_000_000
            recorder = AxisExperimentRecorder(Path(directory), DEFAULT_DEVICE_ID, start)
            recorder.append(packet(DEFAULT_DEVICE_ID), start - 1)
            recorder.append(packet("WT5500011408"), start + 500_000_000)
            recorder.append(packet(DEFAULT_DEVICE_ID), start + 500_000_000)
            recorder.append(packet(DEFAULT_DEVICE_ID),
                            start + (TOTAL_SECONDS + 1) * 1_000_000_000)
            recorder.close()
            with (recorder.path / "samples.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["device_id"], DEFAULT_DEVICE_ID)
            self.assertEqual(rows[0]["phase"], "neutral_before")
            self.assertEqual(rows[0]["host_recv_mono_ns"], str(start + 500_000_000))
            self.assertEqual(float(rows[0]["acc_z"]), 1.0)
            self.assertEqual(float(rows[0]["yaw"]), 3.0)
            manifest = json.loads((recorder.path / "manifest.json").read_text())
            self.assertEqual(manifest["samples"], 1)
            self.assertEqual(manifest["phase_counts"]["neutral_before"], 1)
            self.assertEqual(manifest["stop_reason"], "completed")


if __name__ == "__main__":
    unittest.main()
