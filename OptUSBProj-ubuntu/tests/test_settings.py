from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from optusbproj_ubuntu.settings import load_config, save_config
from optusbproj_ubuntu.skeleton import SENSOR_KEYS


class SettingsTests(unittest.TestCase):
    def test_old_seven_slot_config_loads_and_new_slots_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local_config.json"
            path.write_text(json.dumps({
                "version": 2,
                "assignments": {"left_hand": "WT5500011381"},
                "axis_mappings": {"left_hand": ["+Y", "+Z", "+X"]},
            }), encoding="utf-8")
            assignments, mappings = load_config(path)
            self.assertEqual(assignments["left_hand"], "WT5500011381")
            self.assertEqual(mappings["left_hand"], ("+Y", "+Z", "+X"))
            assignments["head"] = "WT5500011408"
            assignments["right_foot"] = "WT5500011624"
            mappings["right_foot"] = ("+Y", "+Z", "+X")
            save_config(path, assignments, mappings)
            restored_assignments, restored_mappings = load_config(path)
            self.assertEqual({key: value for key, value in restored_assignments.items()
                              if value}, assignments)
            self.assertEqual(restored_mappings, mappings)
            self.assertEqual(set(json.loads(path.read_text(encoding="utf-8"))
                                 ["assignments"]), set(SENSOR_KEYS))


if __name__ == "__main__":
    unittest.main()
