"""Stopping-boundary regression: pure planning and synthetic guards, no IO."""
from dataclasses import replace
import math
import random
import unittest
from unittest.mock import Mock

import numpy as np
from optusbproj_ubuntu.real_robot_arm import ArmControl, ArmLimits
from optusbproj_ubuntu.real_robot_bilateral import BilateralLimits, right_defaults
from optusbproj_ubuntu.real_robot_control import ElbowControl, Limits
from optusbproj_ubuntu.real_robot_protocol import encode, decode
from optusbproj_ubuntu.real_robot_shared import SharedArmControl
from optusbproj_ubuntu.real_robot_trajectory import stopping_boundary
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker

BASE = encode([90.] * 15, [0] * 10)
Q, V, A = 114.5638103375399, 47.23194784273835, 150.


class StopTests(unittest.TestCase):
    def test_recorded_l6_boundary_and_negative_direction(self):
        for direction in (1, -1):
            q, v = direction * Q, direction * V
            target = q + v * abs(v) / (2 * A)
            path = stopping_boundary(q, v, -direction * A, target, 100., A)
            self.assertIsNotNone(path)
            for t in np.linspace(0, path.duration, 31):
                (p,), (speed,), (acc,) = path.at_time(float(t))
                self.assertLessEqual(abs(speed), abs(v))
                self.assertLessEqual(abs(acc), A)
                self.assertGreaterEqual(p, path.position_extrema[0].min)
                self.assertLessEqual(p, path.position_extrema[0].max)
            self.assertEqual(path.at_time(path.duration + 1), ([target], [0.], [0.]))

    def test_only_numerically_equal_valid_states_use_analytic_branch(self):
        stop = Q + V * V / (2 * A)
        self.assertIsNone(stopping_boundary(Q, V, -A, stop + 1e-7, 100., A))
        self.assertIsNone(stopping_boundary(Q, 101., -A, stop, 100., A))
        self.assertIsNone(stopping_boundary(Q, V, -151., stop, 100., A))
        self.assertIsNone(stopping_boundary(Q, V, -A, stop, 100., 0.))
        self.assertIsNone(stopping_boundary(Q, V, -A, float('nan'), 100., A))
        path = stopping_boundary(90., 0., 0., 90., 100., A)
        self.assertEqual(path.at_time(.05), ([90.], [0.], [0.]))

    def test_roundoff_endpoint_is_exact_but_extrema_are_not_hidden(self):
        path = stopping_boundary(.1, -1., 0., 0., 100., 5.)
        self.assertEqual(path.at_time(1.), ([0.], [0.], [0.]))
        stop = Q + V * V / (2 * A)
        path = stopping_boundary(Q, V, -A, stop - 5e-13, 100., A)
        self.assertGreaterEqual(path.position_extrema[0].max, stop)

    def test_unrelated_ruckig_failures_are_not_ignored(self):
        c = self.control('arm')
        c.target[0] = 100.
        c.planner = Mock()
        c.planner.calculate.side_effect = c.r.RuckigError('synthetic failure')
        with self.assertRaisesRegex(RuntimeError, 'L1 轨迹规划失败'):
            c._plan(0)

    def test_random_stop_states_are_continuous_and_acceleration_bounded(self):
        rng = random.Random(16)
        for _ in range(2000):
            q, v, acc = rng.uniform(0., 240.), rng.uniform(-250., 250.), rng.uniform(.1, 250.)
            end = q + v * abs(v) / (2 * acc)
            path = stopping_boundary(q, v, 0., end, 250., acc)
            self.assertIsNotNone(path)
            dt = min(.01, path.duration / 2)
            (p,), (speed,), _ = path.at_time(dt)
            self.assertAlmostEqual(p, q + v * dt - math.copysign(.5 * acc * dt**2, v))
            self.assertLessEqual(abs(speed - v), acc * dt + 1e-10)

    def control(self, kind, frequency=20.):
        limits = ArmLimits(enabled=(True,) * 6, frequency=frequency)
        if kind == 'shared':
            c = SharedArmControl(BASE, limits, [90.] * 6)
            c.phase = 'following'
        elif kind == 'bilateral':
            c = ArmControl(BASE, BilateralLimits(replace(right_defaults(), enabled=(True,)*6,
                                                        frequency=frequency), limits))
        else:
            c = ArmControl(BASE, limits)
        c.first = False
        c.q[-1], c.v[-1], c.a[-1] = Q, V, -A
        return c

    def test_logged_l6_gap_braking_all_arm_modes_and_frequencies(self):
        for kind in ('arm', 'shared', 'bilateral'):
            for frequency in (1., 20., 100.):
                with self.subTest(kind=kind, frequency=frequency):
                    c = self.control(kind, frequency)
                    c.brake('input gap'); c.mode = 'gap'
                    stop = c.target[-1]
                    previous_q, previous_v = c.q[-1], c.v[-1]
                    for _ in range(int(frequency) + 2):
                        frame = c.step()
                        self.assertGreaterEqual(c.q[-1], previous_q - 1e-10)
                        self.assertLessEqual(c.q[-1], 122. + 1e-10)
                        self.assertLessEqual(abs(c.v[-1] - previous_v), A / frequency + 1e-10)
                        self.assertEqual(decode(frame)[0][:6], [90.] * 6)
                        previous_q, previous_v = c.q[-1], c.v[-1]
                    self.assertAlmostEqual(c.q[-1], stop)
                    self.assertEqual(c.v[-1], 0.)
                    self.assertEqual(c.mode, 'gap')
                    self.assertEqual(c.status()['planning_methods'][-1], 'analytic_stop')

    def test_guard_recovery_and_hard_timeout_keep_existing_semantics(self):
        for recovery in (True, False):
            c = self.control('shared')
            w = ElbowSerialWorker('NEVER_OPEN', Mock(error=None))
            target = [90.] * 5 + [100.]
            w._reserve(c, target, 10.25)
            w._input_guard(c, 10.26, 10.25, None)
            self.assertEqual(c.mode, 'gap')
            c.step()
            previous = c.q.copy()
            w.latest = (target, 10.8)
            w._input_guard(c, 10.3 if recovery else 10.51, 10.8, None)
            np.testing.assert_array_equal(c.q, previous)
            self.assertEqual(c.mode, 'active' if recovery else 'braking')
            for _ in range(80): c.step()
            if not recovery:
                self.assertEqual(c.mode, 'paused')
                w._input_guard(c, 10.55, 10.8, None)
                self.assertEqual(c.mode, 'paused')

    def test_elbow_and_out_of_bounds_still_rejected(self):
        c = ElbowControl(BASE, 90., Limits(speed=100., acceleration=A))
        c.first = False
        c.q, c.v, c.a = Q - 30., V, -A
        c.brake('pause')
        for _ in range(20): c.step()
        self.assertEqual(c.mode, 'paused')
        self.assertEqual(c.status()['planning_method'], 'analytic_stop')
        c = self.control('arm')
        c.q[-1] = 121.
        with self.assertRaisesRegex(RuntimeError, '制动轨迹越界'):
            c.brake('unsafe stop')


if __name__ == '__main__':
    unittest.main()
