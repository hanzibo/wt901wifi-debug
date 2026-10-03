from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optusbproj_ubuntu.rgbd_rate import PhaseRateLimiter


class RateTests(unittest.TestCase):
    def test_equal_source_rate_with_jitter_keeps_frames(self):
        limiter = PhaseRateLimiter(10, 0)
        times = np.arange(600)/10 + .004*np.sin(np.arange(600)*1.3)
        self.assertEqual(sum(limiter.accept(float(t)) for t in times), 600)

    def test_thirty_to_ten_and_no_burst_after_stall(self):
        limiter = PhaseRateLimiter(10, 0)
        self.assertEqual(sum(limiter.accept(t/30) for t in range(300)), 100)
        self.assertTrue(limiter.accept(30.01))
        self.assertFalse(limiter.accept(30.012))
        self.assertTrue(limiter.accept(30.101))

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            PhaseRateLimiter(0, 0)
