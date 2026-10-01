"""Neutral-preserving segment scaling and live/recording integration."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_x1_live import prepared_engine, URDF
from optusbproj_ubuntu.x1_retarget import retarget_bone_points, retarget_scales
from optusbproj_ubuntu.x1_live import LiveRetargeter
from optusbproj_ubuntu.x1_live_recording import X1DiagnosticRecorder


class ScalingTests(unittest.TestCase):
    def setUp(self):
        self.h = tuple(np.array(x, dtype=float) for x in ([0, 0, 0], [0, 0, -2], [0, 0, -3]))
        self.r = tuple(np.array(x, dtype=float) for x in ([0, 0, 0], [0, 0, -1], [0, 0, -3]))

    def test_scales_and_neutral(self):
        self.assertEqual(retarget_scales(self.h, self.r), (1., 1.))
        self.assertEqual(retarget_scales(self.h, self.r, "segments"), (.5, 2.))
        for mode in ("uniform", "segments"):
            e, w, bend = retarget_bone_points(self.h, self.h, self.r, scale_mode=mode)
            np.testing.assert_array_equal(e, self.r[1])
            np.testing.assert_array_equal(w, self.r[2])
            self.assertEqual(bend, 0.)

    def test_independent_displacements_and_translation_invariance(self):
        current = tuple(np.array(x, dtype=float) for x in ([0, 0, 0], [2, 0, 0], [2, 1, 0]))
        e, w, bend = retarget_bone_points(self.h, current, self.r, scale_mode="segments")
        np.testing.assert_allclose(e, [1, 0, 0])
        np.testing.assert_allclose(w, [1, 2, 0])
        self.assertAlmostEqual(bend, 90.)
        shifted = tuple(p + [7, 8, 9] for p in current)
        e2, w2, _ = retarget_bone_points(self.h, shifted, self.r, scale_mode="segments")
        np.testing.assert_allclose(e2, e)
        np.testing.assert_allclose(w2, w)

    def test_invalid_mode_and_lengths(self):
        with self.assertRaises(ValueError):
            retarget_scales(self.h, self.r, "bad")
        with self.assertRaises(ValueError):
            retarget_scales((self.h[0], self.h[0], self.h[2]), self.r, "segments")

    def test_live_status_and_recording(self):
        for mode in ("uniform", "segments"):
            bridge = LiveRetargeter(prepared_engine(), URDF, scale_mode=mode)
            _, status = bridge.step()
            self.assertEqual(status["left"]["scale_mode"], mode)
            with tempfile.TemporaryDirectory() as directory:
                recorder = X1DiagnosticRecorder(Path(directory), bridge, URDF)
                manifest = json.loads((recorder.path / "manifest.json").read_text())
                self.assertEqual(manifest["solver"]["scale_mode"], mode)
                self.assertEqual(manifest["solver"]["length_scales"]["left"]["upper"],
                                 bridge.length_scales["left"][0])
                recorder.close()
